import math

import pandas as pd

from recon import config, prioritization
from recon.prioritization import aging_label, band_for, score_exceptions


def frame(rows):
    return pd.DataFrame(rows, columns=["row_id", "exception_type", "amount", "txn_date"])


def test_weights_sum_to_one():
    assert math.isclose(sum(config.PRIORITY_WEIGHTS.values()), 1.0)


def test_large_old_risky_amount_mismatch_outranks_small_fresh_date_mismatch():
    ex = frame([("a", "AMOUNT_MISMATCH", 900_000.0, "2026-06-01"), ("b", "DATE_MISMATCH", 500.0, "2026-08-28")])
    out = score_exceptions(ex, pd.Series({"a": 80.0, "b": 0.0}), "2026-08-29").set_index("row_id")
    assert out.loc["a", "priority_score"] > out.loc["b", "priority_score"] + 40
    assert out.loc["a", "priority_band"] == "CRITICAL" and out.loc["b", "priority_band"] == "LOW"


def test_each_component_moves_the_score_in_the_right_direction():
    base = ("MISSING", 10_000.0, "2026-08-01")
    ex = frame([("base",) + base, ("bigger", "MISSING", 99_000_000.0, "2026-08-01"),
                ("worse", "AMOUNT_MISMATCH", 10_000.0, "2026-08-01"), ("older", "MISSING", 10_000.0, "2026-05-01")])
    s = score_exceptions(ex, pd.Series({"base": 0.0}), "2026-08-29").set_index("row_id")["priority_score"]
    assert s["bigger"] > s["base"] and s["worse"] > s["base"] and s["older"] > s["base"]
    risky = score_exceptions(ex, pd.Series({"base": 90.0}), "2026-08-29").set_index("row_id")["priority_score"]
    assert risky["base"] > s["base"]


def test_age_is_measured_from_the_as_of_date_and_never_negative():
    ex = frame([("a", "MISSING", 1.0, "2026-08-19"), ("b", "MISSING", 1.0, "2026-09-10")])
    out = score_exceptions(ex, pd.Series(dtype=float), "2026-08-29").set_index("row_id")
    assert out.loc["a", "age_days"] == 10 and out.loc["b", "age_days"] == 0


def test_scores_are_bounded_and_complete():
    ex = frame([(f"r{i}", t, float(10 ** (i % 7)), "2026-08-01")
                for i, t in enumerate(["MISSING", "DUPLICATE", "AMOUNT_MISMATCH", "DATE_MISMATCH", "PROBABLE_MATCH"] * 8)])
    out = score_exceptions(ex, pd.Series(dtype=float), "2026-08-29")
    assert out["priority_score"].between(0, 100).all() and out["priority_band"].notna().all()


def test_unknown_exception_type_gets_a_neutral_severity_not_a_crash():
    out = score_exceptions(frame([("a", "SOMETHING_NEW", 5.0, "2026-08-01")]), pd.Series(dtype=float), "2026-08-29")
    assert out["priority_score"].notna().all()


def test_bands_and_aging_labels():
    assert [band_for(x) for x in (95, 70, 69.9, 55, 40, 39.9, 0)] == ["CRITICAL", "CRITICAL", "HIGH", "HIGH", "MEDIUM", "LOW", "LOW"]
    assert [aging_label(d) for d in (0, 7, 8, 30, 31, 61, 400)] == ["0-7 days", "0-7 days", "8-14 days", "15-30 days",
                                                                  "31-60 days", "60+ days", "60+ days"]


def test_empty_input():
    out = score_exceptions(frame([]), pd.Series(dtype=float), "2026-08-29")
    assert out.empty and "priority_score" in out.columns
