"""
Review prioritisation.

Thousands of exceptions are no use as a flat list: an analyst has hours, not
days. Each exception gets a 0-100 priority built from four transparent parts
(weights and severities live in ``config`` so they can be argued with):

    amount    percentile rank of its value among all exceptions       (40%)
    severity  how serious this *kind* of break is (policy judgement)  (25%)
    risk      the transaction's anomaly risk score                    (20%)
    age       days outstanding, capped at AGING_CAP_DAYS              (15%)

Percentile-ranking the amount keeps the score comparable whether a run holds a
few thousand rupees or a few hundred crore.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def band_for(score: float) -> str:
    for name, floor in config.PRIORITY_BANDS:
        if score >= floor:
            return name
    return config.PRIORITY_BANDS[-1][0]


def aging_label(days: float) -> str:
    for lo, hi, label in config.AGING_BUCKETS:
        if lo <= days <= hi:
            return label
    return config.AGING_BUCKETS[-1][2]


def score_exceptions(exceptions: pd.DataFrame, risk_by_row: pd.Series, as_of) -> pd.DataFrame:
    """Add risk_score, age_days, priority_score and priority_band to a classified exceptions frame."""
    out = exceptions.copy()
    if out.empty:
        for c, dt in (("risk_score", float), ("age_days", int), ("priority_score", float), ("priority_band", object)):
            out[c] = pd.Series(dtype=dt)
        return out
    w = config.PRIORITY_WEIGHTS
    out["risk_score"] = out["row_id"].map(risk_by_row).fillna(0.0).astype(float)
    out["age_days"] = (pd.Timestamp(as_of) - pd.to_datetime(out["txn_date"])).dt.days.clip(lower=0).astype(int)

    amount_c = out["amount"].rank(pct=True) * 100
    severity_c = out["exception_type"].map(config.EXCEPTION_SEVERITY).fillna(50.0)
    age_c = (out["age_days"] / config.AGING_CAP_DAYS * 100).clip(upper=100)
    out["priority_score"] = (w["amount"] * amount_c + w["severity"] * severity_c +
                             w["risk"] * out["risk_score"] + w["age"] * age_c).round(1)
    out["priority_band"] = out["priority_score"].map(band_for)
    return out
