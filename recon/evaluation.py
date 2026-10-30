"""
Grades a pipeline run against the labels ``simulate`` held out.

Three things are measured, because "does it work" has three honest answers:

1. **Matching and classification** -- did each row end up where the injected defect says it
   should (matched, missing, duplicate, amount / date mismatch)? Equally important: how often did
   a row that *should* have been flagged get silently auto-matched?
2. **Ingestion validation** -- were the injected junk rows (bad dates, bad amounts, TOTAL lines,
   blanks) caught, and were any genuine rows wrongly quarantined?
3. **Anomaly detection** -- ranking quality (ROC-AUC, average precision) and flag-level
   precision / recall for rules alone, the Isolation Forest alone, and the blend, at transaction
   level. A transaction is one real-world payment; its Bank and Ledger rows are not two samples.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config

EXPECTED = {
    "MATCH": {"MATCHED"},
    "MISSING": {"MISSING"},
    "DUPLICATE": {"MATCHED", "DUPLICATE"},          # either copy may be the one that reconciles
    "AMOUNT_MISMATCH_SMALL": {"MATCHED"},
    "AMOUNT_MISMATCH_LARGE": {"AMOUNT_MISMATCH"},
    "DATE_MISMATCH_MODERATE": {"MATCHED"},
    "DATE_MISMATCH_LARGE": {"DATE_MISMATCH"},
    "REFERENCE_MISMATCH": {"MATCHED"},
}
SHOULD_BE_FLAGGED = ("MISSING", "AMOUNT_MISMATCH_LARGE", "DATE_MISMATCH_LARGE")
ANOMALY_PREFIX = "SUSPICIOUS_"


def load_row_labels(raw_dir) -> pd.DataFrame:
    frames = []
    for source, stem in config.SOURCE_FILE_STEMS.items():
        path = raw_dir / f"_debug_{stem}_mapping.csv"
        if path.exists():
            m = pd.read_csv(path)
            m["row_id"] = [f"{source}-{i:07d}" for i in m["row_index"]]
            m["source_system"] = source
            frames.append(m[["row_id", "source_system", "true_txn_id", "discrepancy"]])
    if not frames:
        raise FileNotFoundError(f"no _debug_*_mapping.csv ground-truth files in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def _expected(label: str):
    return {"MATCHED"} if label.startswith(ANOMALY_PREFIX) else EXPECTED.get(label)


def evaluate_matching(result, labels: pd.DataFrame) -> dict:
    rows = labels[labels["true_txn_id"] >= 0].copy()
    rows = rows[rows["source_system"].isin(result.raw["source_system"].unique())]
    matched = set(result.matches["row_id_a"]) | set(result.matches["row_id_b"])
    exc = dict(zip(result.exceptions["row_id"], result.exceptions["exception_type"]))
    seen = set(result.raw["row_id"])
    rows["actual"] = [("MATCHED" if r in matched else exc.get(r, "UNSEEN")) if r in seen else "QUARANTINED"
                      for r in rows["row_id"]]
    rows["expected_set"] = rows["discrepancy"].map(_expected)
    rows = rows[rows["expected_set"].notna()]
    rows["correct"] = [a in e for a, e in zip(rows["actual"], rows["expected_set"])]

    by_type = (rows.groupby("discrepancy")["correct"].agg(rows="size", correct="sum").assign(
        accuracy=lambda d: (d["correct"] / d["rows"] * 100).round(1)).reset_index())
    flagged_rows = rows[rows["discrepancy"].isin(SHOULD_BE_FLAGGED)]
    return {
        "rows": len(rows), "overall_accuracy": round(rows["correct"].mean() * 100, 2),
        "by_type": by_type, "confusion": pd.crosstab(rows["discrepancy"], rows["actual"]),
        "should_be_flagged_rows": len(flagged_rows),
        "silently_auto_matched_pct": round((flagged_rows["actual"] == "MATCHED").mean() * 100, 2) if len(flagged_rows) else None,
        "clean_match_pct": round((rows.loc[rows["discrepancy"] == "MATCH", "actual"] == "MATCHED").mean() * 100, 2),
    }


def evaluate_pairs(result, labels: pd.DataFrame) -> dict:
    """Pair-level precision: of the pairs the cascade matched, how many link two rows of the SAME real
    transaction? Row-level accuracy cannot see this -- two clean rows wrongly paired with each other
    both look 'correctly matched' -- so a pipeline can post a high match rate while pairing the wrong
    things. This is the stricter, more honest number."""
    m = result.matches
    if m.empty:
        return {"pairs": 0, "precision_pct": None, "by_stage": pd.DataFrame(columns=["match_type", "pairs", "precision_pct"])}
    truth = labels.set_index("row_id")["true_txn_id"]
    ok = (m["row_id_a"].map(truth) == m["row_id_b"].map(truth)).fillna(False)
    by_stage = (m.assign(correct=ok).groupby("match_type")["correct"]
                .agg(pairs="size", precision_pct=lambda x: round(x.mean() * 100, 1)).reset_index())
    return {"pairs": len(m), "precision_pct": round(ok.mean() * 100, 2), "by_stage": by_stage,
            "correctly_paired_rows_pct": round(2 * ok.sum() / result.total * 100, 2)}


def evaluate_ingestion(result, labels: pd.DataFrame) -> dict:
    junk = labels[labels["true_txn_id"] < 0]
    f = result.failures
    caught = {}
    if not f.empty and "row_id" in f:
        sev_by_row = f.dropna(subset=["row_id"]).groupby("row_id")["severity"].agg(lambda s: "ERROR" if "ERROR" in set(s) else s.iloc[0])
        caught = sev_by_row.to_dict()
    from .simulate import JUNK_EXPECTED_SEVERITY
    junk = junk.assign(expected=junk["discrepancy"].map(JUNK_EXPECTED_SEVERITY), actual=junk["row_id"].map(caught))
    junk["detected"] = junk["expected"] == junk["actual"]
    genuine = labels[labels["true_txn_id"] >= 0]
    wrongly = f[f["row_id"].isin(genuine["row_id"]) & f["row_excluded"].astype(bool)] if not f.empty else f
    return {
        "junk_rows": len(junk), "detected": int(junk["detected"].sum()),
        "detection_rate": round(junk["detected"].mean() * 100, 2) if len(junk) else None,
        "by_kind": junk.groupby("discrepancy")["detected"].agg(rows="size", detected="sum").reset_index(),
        "genuine_rows_wrongly_quarantined": int(len(wrongly)),
    }


def _safe(fn, y, s):
    return round(float(fn(y, s)), 4) if 0 < y.sum() < len(y) else None


def evaluate_anomalies(result, labels: pd.DataFrame) -> dict:
    rows = result.raw.merge(labels[["row_id", "true_txn_id", "discrepancy"]], on="row_id", how="inner")
    rows = rows[rows["true_txn_id"] >= 0]
    rows["is_anomaly"] = rows["discrepancy"].str.startswith(ANOMALY_PREFIX)
    pattern = rows[rows["is_anomaly"]].groupby("true_txn_id")["discrepancy"].first()
    txn = rows.groupby("true_txn_id").agg(is_anomaly=("is_anomaly", "max"), risk=("risk_score", "max"),
                                          rule=("rule_score", "max"), model=("ml_raw", "max"),
                                          flagged=("is_suspicious", "max"))
    y = txn["is_anomaly"].astype(int).to_numpy()
    rules_flag = txn["rule"] >= config.RISK_FLAG_THRESHOLD
    clean = ~txn["is_anomaly"]
    n_pos = int(y.sum())

    def flag_stats(flag):
        return {"recall_pct": round(flag[txn["is_anomaly"]].mean() * 100, 1) if n_pos else None,
                "precision_pct": round(txn["is_anomaly"][flag].mean() * 100, 1) if flag.any() else None,
                "false_positive_rate_pct": round(flag[clean].mean() * 100, 3), "flagged_txns": int(flag.sum())}

    top_k = txn.sort_values("risk", ascending=False).head(n_pos)
    ranking = pd.DataFrame([
        {"method": "rules only", "roc_auc": _safe(roc_auc_score, y, txn["rule"]), "avg_precision": _safe(average_precision_score, y, txn["rule"])},
        {"method": "Isolation Forest only", "roc_auc": _safe(roc_auc_score, y, txn["model"]), "avg_precision": _safe(average_precision_score, y, txn["model"])},
        {"method": "blended risk score", "roc_auc": _safe(roc_auc_score, y, txn["risk"]), "avg_precision": _safe(average_precision_score, y, txn["risk"])},
    ])
    per_pattern = pd.DataFrame({
        "transactions": pattern.value_counts(),
        "rules_recall_pct": (rules_flag.reindex(pattern.index).groupby(pattern).mean() * 100).round(1),
        "blended_recall_pct": (txn["flagged"].astype(bool).reindex(pattern.index).groupby(pattern).mean() * 100).round(1),
    }).reset_index().rename(columns={"index": "pattern"})
    return {"anomalous_txns": n_pos, "txns": len(txn), "ranking": ranking, "per_pattern": per_pattern,
            "flag_rules_only": flag_stats(rules_flag), "flag_blended": flag_stats(txn["flagged"].astype(bool)),
            "precision_at_k_pct": round(top_k["is_anomaly"].mean() * 100, 1) if n_pos else None, "k": n_pos}


def evaluate(result, raw_dir=None) -> dict:
    labels = load_row_labels(raw_dir or config.RAW_DIR)
    return {"matching": evaluate_matching(result, labels), "pairs": evaluate_pairs(result, labels), "ingestion": evaluate_ingestion(result, labels),
            "anomaly": evaluate_anomalies(result, labels)}


def format_report(rep: dict) -> str:
    m, p, i, a = rep["matching"], rep["pairs"], rep["ingestion"], rep["anomaly"]
    out = [f"MATCHING & CLASSIFICATION  ({m['rows']:,} labelled rows)",
           f"  overall row-level accuracy      {m['overall_accuracy']}%",
           f"  clean transactions auto-matched {m['clean_match_pct']}%",
           f"  should-be-flagged rows silently auto-matched  {m['silently_auto_matched_pct']}%  (of {m['should_be_flagged_rows']:,})",
           m["by_type"].to_string(index=False), "",
           f"PAIR-LEVEL PRECISION  ({p['pairs']:,} matched pairs)",
           f"  pairs linking the SAME real transaction: {p['precision_pct']}%   "
           f"(rows correctly paired: {p.get('correctly_paired_rows_pct')}% of all rows)",
           p["by_stage"].to_string(index=False), "",
           f"INGESTION VALIDATION  ({i['junk_rows']} injected junk rows)",
           f"  detected {i['detected']} ({i['detection_rate']}%)   genuine rows wrongly quarantined: {i['genuine_rows_wrongly_quarantined']}",
           i["by_kind"].to_string(index=False), "",
           f"ANOMALY DETECTION  ({a['anomalous_txns']} anomalous of {a['txns']:,} transactions)",
           a["ranking"].to_string(index=False),
           f"  flagged at risk >= {config.RISK_FLAG_THRESHOLD}:  rules only {a['flag_rules_only']}",
           f"                         blended    {a['flag_blended']}",
           f"  precision@{a['k']} (top-k by blended risk): {a['precision_at_k_pct']}%",
           a["per_pattern"].to_string(index=False)]
    return "\n".join(out)
