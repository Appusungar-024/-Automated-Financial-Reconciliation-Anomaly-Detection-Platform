"""
Anomaly detection: rules + Isolation Forest -> one explainable 0-100 risk score.

Why both?
* **Rules** encode known red flags (reference reuse, structuring just under a
  reporting threshold, round-number payments, ...). They are precise and every
  flag carries a reason an auditor can read -- but they only find what someone
  thought to write down.
* **Isolation Forest** (scikit-learn, unsupervised -- it never sees labels)
  scores how *unusual* each transaction is across ~13 engineered features
  (amount, vendor-relative amount, timing, vendor history, bursts ...), so it
  can surface odd combinations no rule anticipated.

The two are combined with a noisy-OR -- risk = 1 - (1 - rules)(1 - model) -- so
either strong signal is enough, and agreement raises the score further. A
transaction can reconcile perfectly and still be suspicious (the same fabricated
payment entered in both books matches cleanly), which is why this runs on every
transaction, not just on exceptions.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config

log = logging.getLogger(__name__)
SYM = config.CURRENCY_SYMBOL

FEATURES = ["log_amount", "z_loo", "log_ratio_to_baseline", "is_round_10k", "round_level", "is_weekend",
            "dow", "day_of_month", "vendor_rows_log", "burst_count", "near_threshold", "ref_reuse"]

# features worth quoting when explaining a model flag: (label, direction, percentile cut-off)
_EXPLAIN = {
    "log_amount": ("amount is in the top {p:.1f}% of all transactions", "high", 0.99),
    "z_loo": ("amount is far above this vendor's usual range", "high", 0.99),
    "log_ratio_to_baseline": ("amount is many times the usual for this vendor (or for the book, if the vendor is new)", "high", 0.99),
    "burst_count": ("unusual burst of payments to the same vendor", "high", 0.995),
    "vendor_rows_log": ("vendor has very little history", "low", 0.03),
}


def _day_numbers(df: pd.DataFrame) -> np.ndarray:
    return df["txn_date"].to_numpy().astype("datetime64[D]").astype(np.int64)


def loo_zscore(df: pd.DataFrame) -> pd.Series:
    """Leave-one-out z-score of each amount against its own vendor's history.

    Including the outlier in its own baseline inflates the std and masks it: a
    60,000 payment against a history of 0-8,000 scores only z~2.6 when it drags
    its own mean/std upward, but z~19 when judged against the other rows."""
    amount = df["amount"].astype(float)
    grp = df.groupby("vendor_clean")["amount"]
    n = grp.transform("count")
    s = grp.transform("sum")
    ss = amount.pow(2).groupby(df["vendor_clean"]).transform("sum")
    loo_n = (n - 1).clip(lower=1)
    loo_mean = (s - amount) / loo_n
    loo_var = (((ss - amount ** 2) - loo_n * loo_mean ** 2) / (loo_n - 1).clip(lower=1)).clip(lower=0)
    z = (amount - loo_mean) / np.sqrt(loo_var).replace(0, np.nan)
    z[(n < 4) | (df["vendor_clean"] == "")] = np.nan       # need >= 3 other rows for a meaningful std
    return z


def _burst_counts(df: pd.DataFrame, window: int, mask: pd.Series | None = None) -> np.ndarray:
    """For every row: how many OTHER rows of the same (source, vendor) fall within +/- window days.
    One global sort + searchsorted -- no per-vendor Python loop."""
    rows = df if mask is None else df[mask]
    out = np.zeros(len(df), dtype=int)
    if rows.empty:
        return out
    gid = pd.factorize(rows["source_system"] + "|" + rows["vendor_clean"])[0].astype(np.int64)
    key = gid * 1_000_000 + _day_numbers(rows)
    ordered = np.sort(key)
    counts = np.searchsorted(ordered, key + window, side="right") - np.searchsorted(ordered, key - window, side="left") - 1
    counts = np.where(rows["vendor_clean"].to_numpy() == "", 0, counts)
    if mask is None:
        return counts
    out[np.flatnonzero(mask.to_numpy())] = counts
    return out


def _reference_reuse(df: pd.DataFrame) -> pd.Series:
    """Same reference, SAME source, different amounts: a reused or tampered instrument.
    (Scoped to one source on purpose: a gateway and a bank legitimately share a settlement
    reference, and if their amounts differ that is a mismatch exception, not tampering.)"""
    has_ref = df["reference"] != ""
    distinct = df[has_ref].groupby(["source_system", "reference"])["amount"].transform("nunique")
    flag = pd.Series(False, index=df.index)
    flag[has_ref] = distinct > 1
    return flag


def rule_scores(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    """-> (rule_score 0-100, reasons text, individual boolean masks)."""
    pts = config.RULE_POINTS
    amount = df["amount"].astype(float)
    masks = {
        "round_amount": (amount >= config.ROUND_AMOUNT_DIVISOR) & (amount % config.ROUND_AMOUNT_DIVISOR == 0),
        "weekend": df["txn_date"].dt.weekday >= 5,
        "vendor_outlier": loo_zscore(df).abs() >= config.ZSCORE_OUTLIER_THRESHOLD,
        "reference_reuse": _reference_reuse(df),
    }
    near = (amount >= config.STRUCTURING_THRESHOLD * 0.9) & (amount < config.STRUCTURING_THRESHOLD)
    masks["structuring"] = pd.Series(_burst_counts(df, config.STRUCTURING_WINDOW_DAYS, near) >= 1, index=df.index) & near
    vendor_rows = df.groupby("vendor_clean")["amount"].transform("count")
    if len(df) >= config.NEW_VENDOR_MIN_DATASET_ROWS:       # "top 0.5% of amounts" needs a real distribution
        masks["new_vendor_large"] = ((vendor_rows <= config.NEW_VENDOR_MAX_ROWS) & (df["vendor_clean"] != "") &
                                     (amount >= amount.quantile(config.NEW_VENDOR_AMOUNT_QUANTILE)))
    else:
        masks["new_vendor_large"] = pd.Series(False, index=df.index)
    masks = {k: v.fillna(False).astype(bool) for k, v in masks.items()}

    text = {
        "round_amount": f"round amount (multiple of {SYM}{config.ROUND_AMOUNT_DIVISOR:,})",
        "weekend": "dated on a weekend",
        "vendor_outlier": f"amount is a statistical outlier for this vendor (|z| >= {config.ZSCORE_OUTLIER_THRESHOLD:g}, leave-one-out)",
        "reference_reuse": "same reference used with different amounts in the same source",
        "structuring": f"repeated payments just under {SYM}{config.STRUCTURING_THRESHOLD:,} to the same vendor within {config.STRUCTURING_WINDOW_DAYS} days",
        "new_vendor_large": "unusually large payment to a vendor with almost no history",
    }
    score = sum(masks[k].astype(int) * pts[k] for k in masks).clip(upper=100)
    reasons = pd.Series("", index=df.index, dtype=object)
    any_flag = np.zeros(len(df), dtype=bool)
    for m in masks.values():
        any_flag |= m.to_numpy()
    for i in np.flatnonzero(any_flag):
        reasons.iloc[i] = "; ".join(text[k] for k, m in masks.items() if m.iloc[i])
    return score.astype(float), reasons, masks


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    amount = df["amount"].astype(float)
    grp = df.groupby("vendor_clean")["amount"]
    vendor_n = grp.transform("count")
    median = grp.transform("median")
    is_int = amount % 1 == 0
    f = pd.DataFrame(index=df.index)
    f["log_amount"] = np.log1p(amount)
    f["z_loo"] = loo_zscore(df).fillna(0).clip(-10, 10)
    # Compare each amount to its vendor's median -- or, for a vendor with too little history to
    # have a baseline, to the whole book's median. Defaulting to "no deviation" there (as an earlier
    # version did) hid exactly the case that matters most: a very large payment to a brand-new vendor.
    baseline = median.where(vendor_n >= 3, amount.median())
    f["log_ratio_to_baseline"] = np.log((amount + 1) / (baseline + 1))
    f["is_round_10k"] = ((amount >= config.ROUND_AMOUNT_DIVISOR) & (amount % config.ROUND_AMOUNT_DIVISOR == 0)).astype(int)
    f["round_level"] = sum(((amount % 10 ** k == 0) & is_int & (amount >= 10 ** k)).astype(int) for k in range(1, 6))
    weekday = df["txn_date"].dt.weekday
    f["is_weekend"] = (weekday >= 5).astype(int)
    f["dow"] = weekday
    f["day_of_month"] = df["txn_date"].dt.day
    f["vendor_rows_log"] = np.log1p(vendor_n)
    f["burst_count"] = _burst_counts(df, config.STRUCTURING_WINDOW_DAYS)
    f["near_threshold"] = ((amount >= config.STRUCTURING_THRESHOLD * 0.9) & (amount < config.STRUCTURING_THRESHOLD)).astype(int)
    f["ref_reuse"] = _reference_reuse(df).astype(int)
    return f[FEATURES].astype(float)


def _model_reasons(features: pd.DataFrame, ml_score: np.ndarray) -> pd.Series:
    reasons = pd.Series("", index=features.index, dtype=object)
    flagged = np.flatnonzero(ml_score > 0)
    if len(flagged) == 0:
        return reasons
    pct = features[list(_EXPLAIN)].rank(pct=True)
    for i in flagged:
        parts = []
        for name, (label, direction, cut) in _EXPLAIN.items():
            p = pct.iloc[i][name]
            if (direction == "high" and p >= cut) or (direction == "low" and p <= cut):
                parts.append(label.format(p=(1 - p) * 100 + 0.05))
        reasons.iloc[i] = "model: " + ("; ".join(parts[:3]) if parts else "unusual combination of amount, timing and vendor-history features")
    return reasons


def ml_scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Isolation Forest -> (raw anomaly score, 0-100 tail score, features, model info).

    The tail score is zero for everything below ML_TAIL_START (the 97th percentile by
    default), then rises linearly to 100. Ordinary transactions therefore contribute
    nothing to the blended risk; only the model's most unusual picks do."""
    features = build_features(df)
    model = IsolationForest(n_estimators=config.ML_N_ESTIMATORS, contamination=config.ML_CONTAMINATION,
                            random_state=config.ML_RANDOM_STATE, n_jobs=1)
    model.fit(features.to_numpy())
    raw = -model.score_samples(features.to_numpy())                 # higher = more anomalous
    pct = pd.Series(raw).rank(method="average", pct=True).to_numpy()
    tail = np.clip((pct - config.ML_TAIL_START) / (1 - config.ML_TAIL_START), 0, 1) * 100
    info = {"model": "IsolationForest", "n_rows": int(len(df)), "n_features": len(FEATURES), "features": FEATURES,
            "n_estimators": config.ML_N_ESTIMATORS, "random_state": config.ML_RANDOM_STATE,
            "tail_start_percentile": config.ML_TAIL_START}
    return raw, tail, features, info


def blend(rule: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Noisy-OR of two 0-100 scores."""
    r, m = np.clip(rule, 0, 100) / 100, np.clip(model, 0, 100) / 100
    return 100 * (1 - (1 - r) * (1 - m))


def detect(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Score every transaction. Adds rule_score, ml_raw, ml_score, risk_score, is_suspicious,
    suspicion_reasons. Returns (scored frame, model info for the audit log)."""
    out = df.copy()
    if out.empty:
        for c in ("rule_score", "ml_raw", "ml_score", "risk_score"):
            out[c] = pd.Series(dtype=float)
        out["is_suspicious"] = pd.Series(dtype=bool)
        out["suspicion_reasons"] = pd.Series(dtype=object)
        return out, {"model": "IsolationForest", "n_rows": 0}
    rule, rule_reasons, _ = rule_scores(out)
    raw, tail, features, info = ml_scores(out)
    model_reasons = _model_reasons(features, tail)
    out["rule_score"] = rule.to_numpy()
    out["ml_raw"] = raw
    out["ml_score"] = tail
    out["risk_score"] = blend(rule.to_numpy(), tail).round(1)
    out["is_suspicious"] = out["risk_score"] >= config.RISK_FLAG_THRESHOLD
    out["suspicion_reasons"] = [("; ".join(x for x in (a, b) if x)) for a, b in zip(rule_reasons, model_reasons)]
    info["flagged"] = int(out["is_suspicious"].sum())
    log.info("anomaly detection: %d of %d transactions flagged (risk >= %d)", info["flagged"], len(out),
             config.RISK_FLAG_THRESHOLD)
    return out, info
