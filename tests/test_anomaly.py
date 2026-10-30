import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from recon import anomaly, config
from tests.helpers import make_frame


def book(seed=0, n=3000, n_anom=30):
    """A believable ledger (60 vendors, weekday payments, lognormal amounts) with a handful of
    injected anomalies: large, round, weekend payments many times a vendor's norm."""
    rng = np.random.default_rng(seed)
    vendors = [f"Vendor{i:03d} Ltd" for i in range(60)]
    scale = {v: int(rng.choice([2_000, 8_000, 30_000])) for v in vendors}
    days = pd.bdate_range("2026-06-01", "2026-08-28")
    rows = [(rng.choice(days), v, f"R{i:05d}", round(float(scale[v] * rng.lognormal(0, 0.4)), 2))
            for i, v in enumerate(rng.choice(vendors, n))]
    labels = [0] * n
    saturdays = pd.date_range("2026-06-06", "2026-08-22", freq="7D")
    for k in range(n_anom):
        v = str(rng.choice(vendors))
        amount = float(round(scale[v] * rng.uniform(8, 15) / 10_000) * 10_000) or 100_000.0
        rows.append((rng.choice(saturdays), v, f"A{k:05d}", amount))
        labels.append(1)
    return make_frame("BANK", rows), np.array(labels)


# ------------------------------------------------------------------ leave-one-out z-score

def test_leave_one_out_z_catches_an_outlier_that_a_naive_z_score_masks():
    history = [0, 1000, 1000, 1111.42, 1699.54, 1699.54, 8000, 8000]
    rows = [("2026-08-01", "Mohanty PLC", f"R{i}", a) for i, a in enumerate(history + [60_000.0])]
    df = make_frame("BANK", rows)
    naive = (df["amount"] - df["amount"].mean()) / df["amount"].std()
    assert naive.iloc[-1] < config.ZSCORE_OUTLIER_THRESHOLD          # the outlier inflates its own std...
    assert anomaly.loo_zscore(df).iloc[-1] > 10                       # ...leave-one-out sees it plainly


def test_z_score_is_undefined_without_enough_vendor_history():
    df = make_frame("BANK", [("2026-08-01", "Rare Co", "R1", 100.0), ("2026-08-02", "Rare Co", "R2", 90.0)])
    assert anomaly.loo_zscore(df).isna().all()


# ------------------------------------------------------------------ individual rules

def test_reference_reuse_flags_the_same_source_but_not_a_legitimate_cross_system_share():
    reuse = make_frame("BANK", [("2026-08-01", "A Co", "UTR1", 100.0), ("2026-08-02", "B Co", "UTR1", 250.0)])
    _, _, m = anomaly.rule_scores(reuse)
    assert m["reference_reuse"].all()

    # a gateway and a bank share a settlement UTR by design; differing amounts there is a mismatch,
    # not tampering, so it must not look like reference reuse
    both = pd.concat([make_frame("PAYMENT_GATEWAY", [("2026-08-01", "A Co", "UTR1", 100.0)]),
                      make_frame("BANK", [("2026-08-01", "A Co", "UTR1", 90.0)])], ignore_index=True)
    _, _, m = anomaly.rule_scores(both)
    assert not m["reference_reuse"].any()


def test_identical_repostings_are_not_reference_reuse():
    dup = make_frame("BANK", [("2026-08-01", "A Co", "UTR1", 100.0)] * 2)
    assert not anomaly.rule_scores(dup)[2]["reference_reuse"].any()


def test_structuring_needs_repeat_payments_in_one_source_just_under_the_threshold():
    t = config.STRUCTURING_THRESHOLD
    split = make_frame("BANK", [("2026-08-03", "Acme", "R1", t - 400), ("2026-08-05", "Acme", "R2", t - 900)])
    assert anomaly.rule_scores(split)[2]["structuring"].all()

    # one payment recorded once in the bank and once in the ledger is NOT structuring
    twice = pd.concat([make_frame("BANK", [("2026-08-03", "Acme", "R1", t - 400)]),
                       make_frame("LEDGER", [("2026-08-03", "Acme", "V1", t - 400)])], ignore_index=True)
    assert not anomaly.rule_scores(twice)[2]["structuring"].any()

    spread = make_frame("BANK", [("2026-08-03", "Acme", "R1", t - 400), ("2026-08-20", "Acme", "R2", t - 900)])
    assert not anomaly.rule_scores(spread)[2]["structuring"].any()


def test_round_amount_and_weekend_points_alone_do_not_cross_the_flag_threshold():
    sat = make_frame("BANK", [("2026-08-01", "Acme", "R1", 120_000.0)])            # a Saturday, round
    score, reasons, _ = anomaly.rule_scores(sat)
    assert score.iloc[0] == config.RULE_POINTS["round_amount"] + config.RULE_POINTS["weekend"]
    assert score.iloc[0] < config.RISK_FLAG_THRESHOLD
    assert "round amount" in reasons.iloc[0] and "weekend" in reasons.iloc[0]


def test_large_payment_to_a_first_time_vendor_is_flagged():
    rows = [("2026-08-03", f"Vendor{i % 20}", f"R{i}", 1000.0 + i) for i in range(400)]
    rows.append(("2026-08-04", "Brand New Supplier", "RX", 2_500_000.0))
    _, _, m = anomaly.rule_scores(make_frame("BANK", rows))
    assert m["new_vendor_large"].iloc[-1] and not m["new_vendor_large"].iloc[:-1].any()


# ------------------------------------------------------------------ blending and the model

def test_noisy_or_blend_properties():
    assert anomaly.blend(np.array([0.0]), np.array([0.0]))[0] == 0
    assert anomaly.blend(np.array([60.0]), np.array([0.0]))[0] == pytest.approx(60)
    assert anomaly.blend(np.array([0.0]), np.array([70.0]))[0] == pytest.approx(70)
    assert anomaly.blend(np.array([50.0]), np.array([50.0]))[0] == pytest.approx(75)
    assert anomaly.blend(np.array([100.0]), np.array([100.0]))[0] == pytest.approx(100)


def test_isolation_forest_ranks_injected_anomalies_far_above_normal_rows():
    df, y = book()
    scored, info = anomaly.detect(df)
    assert roc_auc_score(y, scored["ml_raw"]) > 0.95
    assert info["model"] == "IsolationForest" and info["n_features"] == len(anomaly.FEATURES)


def test_blended_risk_flags_the_anomalies_without_flooding_the_queue():
    df, y = book()
    scored, _ = anomaly.detect(df)
    flagged = scored["is_suspicious"].to_numpy()
    assert flagged[y == 1].mean() >= 0.5                  # recalls at least half of the planted anomalies
    assert flagged.mean() < 0.05                           # ...while flagging under 5% of all rows
    assert roc_auc_score(y, scored["risk_score"]) > 0.9


def test_model_flags_come_with_a_reason_a_human_can_read():
    df, y = book()
    scored, _ = anomaly.detect(df)
    top = scored[scored["is_suspicious"]].iloc[0]
    assert top["suspicion_reasons"] != ""
    assert all(c in scored.columns for c in ("rule_score", "ml_score", "risk_score", "ml_raw"))


def test_detection_is_deterministic():
    df, _ = book(n=800, n_anom=10)
    a, _ = anomaly.detect(df)
    b, _ = anomaly.detect(df)
    assert np.allclose(a["risk_score"], b["risk_score"])


def test_empty_frame_is_handled():
    empty = make_frame("BANK", [])
    out, info = anomaly.detect(empty)
    assert out.empty and info["n_rows"] == 0 and "risk_score" in out.columns
