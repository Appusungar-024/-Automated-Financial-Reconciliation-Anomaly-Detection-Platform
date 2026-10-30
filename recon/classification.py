"""
Stage 5 -- exception classification and suggested matches.

Whatever the matching cascade could not resolve is explained here: is it a
DUPLICATE posting, an AMOUNT_MISMATCH, a DATE_MISMATCH, a PROBABLE_MATCH that
just fell outside the auto-match tolerances, or genuinely MISSING?

Design (and why it differs from v1)
-----------------------------------
v1 picked each leftover row's "best" counterpart by a blended score of vendor,
date and amount similarity. For an AMOUNT_MISMATCH that is backwards: the true
counterpart is *defined* by having the wrong amount, so an unrelated
transaction with a similar amount and date could outscore it. Measured, v1
labelled only ~19% of large amount mismatches correctly.

v2 treats identity and attributes separately:

1. **Identity** -- a counterpart is only eligible if its vendor is similar
   (>= CLASSIFY_VENDOR_GATE) or, where references are comparable between the two
   systems, if it shares the reference.
2. **Attributes** -- among eligible candidates, date and amount decide *which*
   field is off, and so which label applies.
3. **Mutual best** -- a mismatch label is only given if the pairing is mutual:
   row A's best candidate is B *and* B's best candidate is A. Otherwise two
   unrelated leftovers from a busy vendor would be paired up and mislabelled.
   Rows without a mutual partner are MISSING, with no misleading suggestion.

Candidates are found by blocking on a vendor prefix (bank narrations truncate
names, so prefixes survive where whole names do not) rather than by an
arbitrary "nearest amount" cap, which is what silently discarded true
counterparts in v1.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import config

EXCEPTION_COLUMNS = ["row_id", "source_system", "pair_name", "txn_date", "vendor_raw", "reference",
                     "amount", "direction", "exception_type", "suggested_match_row_id",
                     "suggested_match_confidence", "suggested_match_reason"]

PREFIX_LEN = 3
CLOSE_SCORE = 70             # date_score / amount_score at or above this counts as "aligned"
MAX_CANDIDATES = 500         # safety valve for pathological blocks


def _relations(pairs):
    counterparts: dict[str, set] = defaultdict(set)
    ref_comparable: set = set()
    pair_names: dict[str, list] = defaultdict(list)
    for p in pairs:
        a, b = p["source_a"], p["source_b"]
        counterparts[a].add(b)
        counterparts[b].add(a)
        pair_names[a].append(p["name"])
        pair_names[b].append(p["name"])
        if p["use_reference_check"]:
            ref_comparable.update({(a, b), (b, a)})
    return counterparts, ref_comparable, pair_names


def _arrays(df: pd.DataFrame) -> dict:
    return {
        "row_id": df["row_id"].to_numpy(), "source": df["source_system"].to_numpy(),
        "day": df["txn_date"].to_numpy().astype("datetime64[D]").astype(np.int64),
        "amount": df["amount"].to_numpy(dtype=float), "vendor": df["vendor_clean"].to_numpy(),
        "ref": df["reference"].to_numpy(),
        "prefix": np.array([v[:PREFIX_LEN] for v in df["vendor_clean"]], dtype=object),
    }


def _duplicate_ids(sources, remaining, matched_ids) -> set:
    """A leftover row is a DUPLICATE if its fingerprint matches a row that DID match (its twin
    was reconciled, this extra copy was not), or if the fingerprint repeats among the leftovers."""
    dups: set = set()
    for s, left in remaining.items():
        if left.empty:
            continue
        full = sources[s]
        twin_hashes = set(full.loc[full["row_id"].isin(matched_ids.get(s, ())), "dup_hash"])
        flagged = left["dup_hash"].isin(twin_hashes) | left.duplicated("dup_hash", keep=False)
        dups.update(left.loc[flagged, "row_id"])
    return dups


def _find_best(source, live, counterparts, ref_comparable) -> dict:
    """For each live leftover row of `source`: its best eligible counterpart and the scores."""
    pool_parts = [live[t] for t in counterparts[source] if t in live and not live[t].empty]
    best: dict = {}
    rows = live.get(source)
    if rows is None or rows.empty:
        return best
    if not pool_parts:
        return {rid: None for rid in rows["row_id"]}
    pool = pd.concat(pool_parts, ignore_index=True)
    P, R = _arrays(pool), _arrays(rows)

    prefix_index: dict = defaultdict(list)
    ref_index: dict = defaultdict(list)
    for j in range(len(pool)):
        if P["prefix"][j]:
            prefix_index[P["prefix"][j]].append(j)
        if P["ref"][j] and (source, P["source"][j]) in ref_comparable:
            ref_index[P["ref"][j]].append(j)

    for i in range(len(rows)):
        rid = R["row_id"][i]
        by_ref = ref_index.get(R["ref"][i], []) if R["ref"][i] else []
        by_vendor = prefix_index.get(R["prefix"][i], []) if R["prefix"][i] else []
        if not by_ref and not by_vendor:
            best[rid] = None
            continue
        ref_set = set(by_ref)
        idx = np.fromiter(ref_set | set(by_vendor), dtype=np.int64)
        day_gap = np.abs(P["day"][idx] - R["day"][i])
        amt_gap = np.abs(P["amount"][idx] - R["amount"][i])
        band = max(R["amount"][i] * config.CLASSIFY_AMOUNT_BAND_PCT, config.CLASSIFY_MIN_AMOUNT_BAND)
        is_ref = np.fromiter((j in ref_set for j in idx), dtype=bool, count=len(idx))
        keep = is_ref | ((day_gap <= config.CLASSIFY_DATE_WINDOW_DAYS) & (amt_gap <= band))
        idx, day_gap, amt_gap, is_ref = idx[keep], day_gap[keep], amt_gap[keep], is_ref[keep]
        if len(idx) > MAX_CANDIDATES:
            order = np.argsort(day_gap)[:MAX_CANDIDATES]
            idx, day_gap, amt_gap, is_ref = idx[order], day_gap[order], amt_gap[order], is_ref[order]

        top = None
        denom = max(R["amount"][i], 1.0)
        for k, j in enumerate(idx):
            if is_ref[k]:
                vendor_score = 100.0
            else:
                if not R["vendor"][i] or not P["vendor"][j]:
                    continue
                vendor_score = float(max(fuzz.token_sort_ratio(R["vendor"][i], P["vendor"][j]),
                                         fuzz.partial_ratio(R["vendor"][i], P["vendor"][j])))
                if vendor_score < config.CLASSIFY_VENDOR_GATE:
                    continue
            date_score = max(0.0, 100.0 - 5.0 * day_gap[k])
            amount_score = max(0.0, 100.0 - 2000.0 * amt_gap[k] / max(denom, P["amount"][j]))
            composite = 0.4 * vendor_score + 0.3 * date_score + 0.3 * amount_score
            if top is None or composite > top["composite"]:
                top = {"cand": P["row_id"][j], "composite": composite, "date_score": date_score,
                       "amount_score": amount_score, "via_ref": bool(is_ref[k]),
                       "day_gap": int(day_gap[k]), "cand_amount": float(P["amount"][j]), "amount": float(R["amount"][i])}
        best[rid] = top
    return best


def _label(b: dict) -> tuple[str, str, float]:
    """(exception_type, reason, confidence) for a mutual best pairing."""
    sym = config.CURRENCY_SYMBOL
    diff = b["cand_amount"] - b["amount"]
    pct = abs(diff) / max(b["amount"], 1.0) * 100
    amount_txt = f"amount differs by {sym}{abs(diff):,.2f} ({pct:.1f}%)"
    date_txt = f"date differs by {b['day_gap']} day{'s' if b['day_gap'] != 1 else ''}"
    close_date, close_amount = b["date_score"] >= CLOSE_SCORE, b["amount_score"] >= CLOSE_SCORE
    basis = "reference matches" if b["via_ref"] else "vendor matches"
    confidence = round(b["composite"], 1)
    if b["via_ref"]:
        confidence = max(confidence, 85.0)
    if close_date and close_amount:
        return "PROBABLE_MATCH", f"{basis}; date and amount are close but outside auto-match tolerance", confidence
    if close_amount:
        return "DATE_MISMATCH", f"{basis} and amount aligns; {date_txt}", confidence
    if close_date:
        return "AMOUNT_MISMATCH", f"{basis} and date aligns; {amount_txt}", confidence
    if b["via_ref"]:
        return "PROBABLE_MATCH", f"{basis}; {date_txt} and {amount_txt}", confidence
    return "MISSING", "", 0.0


def classify_exceptions(sources: dict, remaining: dict, matched_ids: dict, pairs: list) -> pd.DataFrame:
    """Classify every leftover row. ``sources`` are the full per-source frames, ``remaining`` the
    final unmatched rows per source, ``matched_ids`` the row ids each source matched in any pair."""
    counterparts, ref_comparable, pair_names = _relations(pairs)
    dup_ids = _duplicate_ids(sources, remaining, matched_ids)
    live = {s: left[~left["row_id"].isin(dup_ids)] for s, left in remaining.items()}

    best: dict = {}
    for s in live:
        best.update(_find_best(s, live, counterparts, ref_comparable))

    out = []
    for s, left in remaining.items():
        if left.empty:
            continue
        label_pairs = " | ".join(pair_names.get(s, []))
        for row in left.itertuples(index=False):
            etype, sugg_id, sugg_conf, sugg_reason = "MISSING", None, None, None
            if row.row_id in dup_ids:
                etype, sugg_reason = "DUPLICATE", "same date, amount, vendor and reference already posted"
            else:
                mine = best.get(row.row_id)
                theirs = best.get(mine["cand"]) if mine else None
                if mine and theirs and theirs["cand"] == row.row_id:      # mutual best
                    etype, reason, conf = _label(mine)
                    if etype != "MISSING":
                        sugg_id, sugg_conf, sugg_reason = mine["cand"], conf, reason
            out.append((row.row_id, s, label_pairs, row.txn_date, row.vendor_raw, row.reference, row.amount,
                        row.direction, etype, sugg_id, sugg_conf, sugg_reason))
    return pd.DataFrame(out, columns=EXCEPTION_COLUMNS)
