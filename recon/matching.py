"""
The matching cascade:

    Stage 0  REFERENCE       (only where a shared reference is expected) reference + date + amount
    Stage 1  EXACT           date + amount + normalized vendor all identical
    Stage 2  DATE_AMOUNT     date + amount identical, vendor text ignored
    Stage 3  AMOUNT_TOLERANCE  same date, amount within tolerance (bank fees/rounding)
    Stage 4  FUZZY_VENDOR    date within a window + amount within a wider tolerance,
                              best vendor-text similarity above threshold wins

Each stage only looks at what the previous stage left unmatched, and every
match is enforced 1-to-1 (a row can be claimed at most once) so duplicate
postings can't silently multiply into a many-to-many join -- a classic bug
in naive pandas-merge reconciliation scripts.

What's left after stage 4 is handed to classification.py (stage 5) to be classified
as MISSING / DUPLICATE / AMOUNT_MISMATCH / DATE_MISMATCH with a suggested counterpart.

Every match also records *why* it was made -- amount difference, date difference,
vendor similarity and the rule that fired -- so each decision can be audited later.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import config

MATCH_COLUMNS = ["row_id_a", "row_id_b", "source_a", "source_b", "pair_name",
                  "date_a", "date_b", "amount_a", "amount_b", "vendor_a", "vendor_b",
                  "reference_a", "reference_b", "match_type", "match_confidence",
                  "reference_mismatch", "amount_diff", "date_diff_days", "vendor_similarity",
                  "rule_applied", "decided_by"]


def rule_text(match_type: str) -> str:
    """Plain-English statement of the rule that produced a match (stored with each decision)."""
    return {
        "REFERENCE": "Shared reference, date and amount all agree",
        "EXACT": "Date, amount and normalised vendor are identical",
        "DATE_AMOUNT": "Date and amount are identical; vendor text differs or was ignored",
        "AMOUNT_TOLERANCE": (f"Same date; amount within tolerance "
                             f"(max of {config.CURRENCY_SYMBOL}{config.AMOUNT_TOLERANCE_ABS:g} or "
                             f"{config.AMOUNT_TOLERANCE_PCT:.1%})"),
        "FUZZY_VENDOR": (f"Vendor similarity >= {config.FUZZY_VENDOR_THRESHOLD}; date within "
                         f"+/-{config.FUZZY_DATE_WINDOW_DAYS} days; amount within {config.FUZZY_AMOUNT_TOLERANCE_PCT:.0%}"),
        "MANUAL": "Matched manually by a reviewer",
    }.get(match_type, match_type)


@dataclass
class StageStats:
    stage: str
    matched_pairs: int
    remaining_a: int
    remaining_b: int


@dataclass
class ReconciliationResult:
    pair_name: str
    matches: pd.DataFrame
    exceptions: pd.DataFrame
    stage_stats: list = field(default_factory=list)


def _rank_within_keys(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    df = df.copy()
    df["_rank"] = df.groupby(keys).cumcount()
    return df


def _exact_key_stage(a: pd.DataFrame, b: pd.DataFrame, keys: list, match_type: str, confidence: float):
    """Vectorized 1-to-1 merge on exact keys. Duplicate keys within a source
    are paired by occurrence order (1st with 1st, 2nd with 2nd, ...) via
    groupby().cumcount(), so a genuine duplicate posting can match at most
    one counterpart instead of exploding into a cartesian join."""
    ar = _rank_within_keys(a, keys)
    br = _rank_within_keys(b, keys)
    merged = ar.merge(br[keys + ["_rank", "row_id"]], on=keys + ["_rank"], suffixes=("", "_b"))
    if merged.empty:
        empty = pd.DataFrame(columns=["row_id_a", "row_id_b"])
        return empty, a, b

    matched_a_ids = merged["row_id"].tolist()
    matched_b_ids = merged["row_id_b"].tolist()
    pairs = pd.DataFrame({"row_id_a": matched_a_ids, "row_id_b": matched_b_ids,
                           "match_type": match_type, "match_confidence": confidence})

    remaining_a = a[~a["row_id"].isin(matched_a_ids)]
    remaining_b = b[~b["row_id"].isin(matched_b_ids)]
    return pairs, remaining_a, remaining_b


def match_exact(a: pd.DataFrame, b: pd.DataFrame):
    return _exact_key_stage(a, b, ["txn_date", "amount", "vendor_clean"], "EXACT", 100.0)


def match_date_amount(a: pd.DataFrame, b: pd.DataFrame):
    return _exact_key_stage(a, b, ["txn_date", "amount"], "DATE_AMOUNT", 90.0)


def match_amount_tolerance(a: pd.DataFrame, b: pd.DataFrame):
    """Same date, amount within tolerance (max of an absolute rupee amount
    or a percentage, to comfortably cover both small flat bank fees and
    proportional deductions on larger transactions).

    Works on plain numpy arrays / dicts inside the loop rather than slicing
    the DataFrame on every iteration -- pandas' per-call overhead (index
    alignment, new object construction) dominates at this row count even
    though each individual slice is "fast" in isolation."""
    if a.empty or b.empty:
        return pd.DataFrame(columns=["row_id_a", "row_id_b"]), a, b

    b_dates, b_amounts, b_row_ids = b["txn_date"].values, b["amount"].values, b["row_id"].values
    b_by_date = {}
    for i, d in enumerate(b_dates):
        b_by_date.setdefault(d, []).append(i)

    used_b_idx = set()
    rows = []
    a_sorted = a.sort_values(["txn_date", "amount"])
    for a_row in a_sorted.itertuples():
        idxs = b_by_date.get(np.datetime64(a_row.txn_date))
        if not idxs:
            continue
        tol = max(config.AMOUNT_TOLERANCE_ABS, a_row.amount * config.AMOUNT_TOLERANCE_PCT)
        best_i, best_diff = None, None
        for i in idxs:
            if i in used_b_idx:
                continue
            diff = abs(b_amounts[i] - a_row.amount)
            if diff <= tol and (best_diff is None or diff < best_diff):
                best_i, best_diff = i, diff
        if best_i is not None:
            used_b_idx.add(best_i)
            confidence = round(max(60.0, 95.0 - (best_diff / max(a_row.amount, 1e-6)) * 100 * 5), 1)
            rows.append({"row_id_a": a_row.row_id, "row_id_b": b_row_ids[best_i],
                          "match_type": "AMOUNT_TOLERANCE", "match_confidence": confidence})

    pairs = pd.DataFrame(rows, columns=["row_id_a", "row_id_b", "match_type", "match_confidence"])
    remaining_a = a[~a["row_id"].isin(pairs.get("row_id_a", []))]
    remaining_b = b[~b["row_id"].isin(pairs.get("row_id_b", []))]
    return pairs, remaining_a, remaining_b


def match_fuzzy_vendor(a: pd.DataFrame, b: pd.DataFrame):
    """Last resort before giving up: widen the date window and amount
    tolerance, then require strong vendor-text similarity to accept a match.
    This is the stage that rescues transactions where BOTH the date drifted
    (processing delay) AND the amount shifted slightly.

    Performance note: this stays on numpy arrays throughout the loop rather
    than slicing the DataFrame per row -- pandas' per-call overhead was the
    actual bottleneck at 50k-transaction scale, not the O(n log n) search
    itself. searchsorted on the pre-sorted date array gives the window in
    O(log n); amount filtering happens before any rapidfuzz call, since
    it's a cheap way to rule out the vast majority of candidates."""
    if a.empty or b.empty:
        return pd.DataFrame(columns=["row_id_a", "row_id_b"]), a, b

    b_sorted = b.sort_values("txn_date").reset_index(drop=True)
    b_dates = b_sorted["txn_date"].values
    b_amounts = b_sorted["amount"].values
    b_vendors = b_sorted["vendor_clean"].values
    b_row_ids = b_sorted["row_id"].values
    window = pd.Timedelta(days=config.FUZZY_DATE_WINDOW_DAYS).to_timedelta64()

    used_b_idx = set()
    rows = []
    for a_row in a.itertuples():
        if not a_row.vendor_clean:
            continue  # no vendor text to compare -- fuzzy matching has no signal here
        a_date = np.datetime64(a_row.txn_date)
        lo = b_dates.searchsorted(a_date - window, side="left")
        hi = b_dates.searchsorted(a_date + window, side="right")
        if lo >= hi:
            continue

        tol = max(config.AMOUNT_TOLERANCE_ABS, a_row.amount * config.FUZZY_AMOUNT_TOLERANCE_PCT)
        best_i, best_score = None, -1.0
        for i in range(lo, hi):
            if i in used_b_idx or not b_vendors[i]:
                continue
            if abs(b_amounts[i] - a_row.amount) > tol:
                continue
            # token_sort_ratio catches reordered multi-word names;
            # partial_ratio catches truncated bank-narration prefixes that
            # token_sort_ratio alone under-scores. Take whichever fits better.
            score = max(fuzz.token_sort_ratio(a_row.vendor_clean, b_vendors[i]),
                         fuzz.partial_ratio(a_row.vendor_clean, b_vendors[i]))
            if score > best_score:
                best_i, best_score = i, score

        if best_i is not None and best_score >= config.FUZZY_VENDOR_THRESHOLD:
            used_b_idx.add(best_i)
            rows.append({"row_id_a": a_row.row_id, "row_id_b": b_row_ids[best_i],
                          "match_type": "FUZZY_VENDOR", "match_confidence": round(float(best_score), 1)})

    pairs = pd.DataFrame(rows, columns=["row_id_a", "row_id_b", "match_type", "match_confidence"])
    remaining_a = a[~a["row_id"].isin(pairs.get("row_id_a", []))]
    remaining_b = b[~b["row_id"].isin(pairs.get("row_id_b", []))]
    return pairs, remaining_a, remaining_b


def _enrich_matches(pairs: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, pair_cfg: dict) -> pd.DataFrame:
    """Pulls full row detail back onto a bare (row_id_a, row_id_b) pair list, adds the
    decision rationale, and applies the reference-mismatch soft flag where reference
    checking is meaningful for this pair."""
    if pairs.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    cols = ["txn_date", "amount", "vendor_raw", "vendor_clean", "reference"]
    a_cols = a.set_index("row_id")[cols]
    b_cols = b.set_index("row_id")[cols]
    out = pairs.merge(a_cols, left_on="row_id_a", right_index=True) \
               .merge(b_cols, left_on="row_id_b", right_index=True, suffixes=("_a", "_b"))
    out["source_a"] = pair_cfg["source_a"]
    out["source_b"] = pair_cfg["source_b"]
    out["pair_name"] = pair_cfg["name"]
    if pair_cfg["use_reference_check"]:
        out["reference_mismatch"] = out["reference_a"].astype(str).str.strip() != out["reference_b"].astype(str).str.strip()
    else:
        out["reference_mismatch"] = False

    out["amount_diff"] = (out["amount_b"] - out["amount_a"]).round(2)
    out["date_diff_days"] = (out["txn_date_b"] - out["txn_date_a"]).dt.days
    out["vendor_similarity"] = [
        float(max(fuzz.token_sort_ratio(x, y), fuzz.partial_ratio(x, y))) if x and y else np.nan
        for x, y in zip(out["vendor_clean_a"], out["vendor_clean_b"])
    ]
    out["rule_applied"] = out["match_type"].map(rule_text)
    out["decided_by"] = "system"
    out = out.rename(columns={"txn_date_a": "date_a", "txn_date_b": "date_b",
                                "vendor_raw_a": "vendor_a", "vendor_raw_b": "vendor_b"})
    return out[MATCH_COLUMNS]


def match_by_reference(a: pd.DataFrame, b: pd.DataFrame):
    """For pairs where the same reference number is expected to carry
    through both systems (a payment gateway settlement UTR appearing
    verbatim on the bank statement; an ERP invoice number referenced in the
    ledger voucher), reference is a strong extra key for resolving
    same-date/same-amount collisions that vendor-text truncation can create
    (multiple candidates can share a date+amount; at most one should also
    share the reference). Requires date AND amount to also match exactly --
    reference agreeing while either genuinely differs is exactly what
    AMOUNT_MISMATCH / DATE_MISMATCH should catch, not something this stage
    should silently wave through. Only called for pairs where reference
    correspondence is actually meaningful (see config.RECON_PAIRS)."""
    has_ref_a = a["reference"].astype(str).str.strip() != ""
    has_ref_b = b["reference"].astype(str).str.strip() != ""
    pairs, still_a, still_b = _exact_key_stage(
        a[has_ref_a], b[has_ref_b], ["reference", "txn_date", "amount"], "REFERENCE", 99.0
    )
    remaining_a = pd.concat([still_a, a[~has_ref_a]])
    remaining_b = pd.concat([still_b, b[~has_ref_b]])
    return pairs, remaining_a, remaining_b


def reconcile(df_a: pd.DataFrame, df_b: pd.DataFrame, pair_cfg: dict) -> ReconciliationResult:
    """Runs the full matching cascade for one reconciliation pair and
    returns matched pairs plus whatever's left over as raw (unclassified)
    leftovers -- classification into MISSING/DUPLICATE/etc. happens in
    classification.py, since it needs the duplicate flags and cross-source
    context to do it well."""
    stage_stats = []
    all_pairs = []
    remaining_a, remaining_b = df_a, df_b

    stages = []
    if pair_cfg.get("use_reference_check"):
        stages.append(("REFERENCE", match_by_reference))
    stages += [
        ("EXACT", match_exact),
        ("DATE_AMOUNT", match_date_amount),
        ("AMOUNT_TOLERANCE", match_amount_tolerance),
        ("FUZZY_VENDOR", match_fuzzy_vendor),
    ]
    for name, fn in stages:
        pairs, remaining_a, remaining_b = fn(remaining_a, remaining_b)
        all_pairs.append(pairs)
        stage_stats.append(StageStats(name, len(pairs), len(remaining_a), len(remaining_b)))

    combined_pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    matches = _enrich_matches(combined_pairs, df_a, df_b, pair_cfg)

    leftover_a = remaining_a.assign(source_role="a", pair_name=pair_cfg["name"])
    leftover_b = remaining_b.assign(source_role="b", pair_name=pair_cfg["name"])
    exceptions_raw = pd.concat([leftover_a, leftover_b], ignore_index=True)

    return ReconciliationResult(pair_cfg["name"], matches, exceptions_raw, stage_stats)
