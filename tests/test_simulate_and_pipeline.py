"""Simulator contract, and an end-to-end regression test that locks in measured quality."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from recon import config, evaluation, ingestion, pipeline, repository, simulate

END = date(2026, 8, 29)
VALIDATION_AS_OF = "2026-12-31"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    simulate.main(2500, 5, d, end_date=END, excel_sources=("BANK", "ERP"))
    return d


@pytest.fixture(scope="module")
def result(dataset):
    return pipeline.run(directory=dataset, persist=False, validation_as_of=VALIDATION_AS_OF)


# ---------------------------------------------------------------- simulator

def test_simulation_is_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    simulate.main(600, 3, a, end_date=END)
    simulate.main(600, 3, b, end_date=END)
    for name in ("bank.csv", "ledger.csv", "_ground_truth.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_a_different_seed_gives_different_data(tmp_path):
    simulate.main(600, 3, tmp_path / "a", end_date=END)
    simulate.main(600, 4, tmp_path / "b", end_date=END)
    assert (tmp_path / "a" / "bank.csv").read_bytes() != (tmp_path / "b" / "bank.csv").read_bytes()


def test_requested_formats_and_manifest(dataset):
    m = json.loads((dataset / "_manifest.json").read_text())
    assert {s: f["format"] for s, f in m["files"].items()} == {"BANK": "xlsx", "LEDGER": "csv", "PAYMENT_GATEWAY": "csv", "ERP": "xlsx"}
    assert (dataset / "bank.xlsx").exists() and not (dataset / "bank.csv").exists()


def test_switching_format_removes_the_stale_twin_file(tmp_path):
    simulate.main(600, 3, tmp_path, end_date=END, excel_sources=("BANK",))
    simulate.main(600, 3, tmp_path, end_date=END)
    assert (tmp_path / "bank.csv").exists() and not (tmp_path / "bank.xlsx").exists()


def test_no_simulated_transaction_has_a_zero_amount():
    """Regression: rounding a sub-500 amount to the nearest thousand used to produce 0.00."""
    rng = np.random.default_rng(0)
    assert min(simulate.random_amount(rng) for _ in range(20_000)) >= 100


def test_dates_are_weekday_heavy():
    rng = np.random.default_rng(0)
    days = simulate.sample_dates(rng, date(2026, 6, 1), date(2026, 8, 29), 20_000)
    weekend = np.mean([d.weekday() >= 5 for d in days])
    assert weekend < 0.08                     # v1's uniform dates put ~29% on weekends


def test_junk_rows_are_present_and_labelled(dataset):
    labels = evaluation.load_row_labels(dataset)
    junk = labels[labels["true_txn_id"] < 0]
    assert set(junk["discrepancy"]) == set(simulate.JUNK_KINDS) and len(junk) > 5


# ---------------------------------------------------------------- end to end

def test_every_source_ingested_including_excel_with_preamble(result):
    assert {s: r.file_name for s, r in result.ingestion.items()} == {
        "BANK": "bank.xlsx", "LEDGER": "ledger.csv", "PAYMENT_GATEWAY": "payment_gateway.csv", "ERP": "erp.xlsx"}
    assert result.ingestion["BANK"].header_row == 4 and result.ingestion["ERP"].header_row == 2
    assert all(not r.fatal for r in result.ingestion.values())


def test_control_totals_balance(result):
    matched = set(result.matches["row_id_a"]) | set(result.matches["row_id_b"])
    assert len(matched) + len(result.exceptions) == result.total
    assert not matched & set(result.exceptions["row_id"])
    assert result.matches["row_id_a"].is_unique and result.matches["row_id_b"].is_unique


def test_matching_quality_meets_the_bar(result, dataset):
    m = evaluation.evaluate_matching(result, evaluation.load_row_labels(dataset))
    assert m["overall_accuracy"] >= 97.0
    assert m["clean_match_pct"] >= 98.0
    assert m["silently_auto_matched_pct"] <= 8.0         # defects that slipped through as "matches"
    by = m["by_type"].set_index("discrepancy")["accuracy"]
    assert by["AMOUNT_MISMATCH_LARGE"] >= 80 and by["DATE_MISMATCH_LARGE"] >= 80 and by["MISSING"] >= 85
    assert by["DUPLICATE"] >= 95 and by["REFERENCE_MISMATCH"] >= 95


def test_ingestion_validation_catches_every_junk_row_and_no_genuine_one(result, dataset):
    i = evaluation.evaluate_ingestion(result, evaluation.load_row_labels(dataset))
    assert i["detection_rate"] == 100.0
    assert i["genuine_rows_wrongly_quarantined"] == 0


def test_anomaly_detection_quality_meets_the_bar(result, dataset):
    a = evaluation.evaluate_anomalies(result, evaluation.load_row_labels(dataset))
    ranking = a["ranking"].set_index("method")
    assert ranking.loc["blended risk score", "roc_auc"] >= 0.97
    assert ranking.loc["blended risk score", "avg_precision"] > ranking.loc["rules only", "avg_precision"]
    assert a["flag_blended"]["recall_pct"] >= 85 and a["flag_blended"]["false_positive_rate_pct"] <= 2.0
    # the model's job: patterns the rules only partly cover
    spike = a["per_pattern"].set_index("discrepancy").loc["SUSPICIOUS_ROUND_SPIKE"]
    assert spike["blended_recall_pct"] >= spike["rules_recall_pct"] + 30


def test_reference_mismatch_flag_is_set_only_where_references_are_comparable(result):
    m = result.matches
    assert m.loc[m["pair_name"] == "BANK_VS_LEDGER", "reference_mismatch"].eq(False).all()
    assert m.loc[m["pair_name"] != "BANK_VS_LEDGER", "reference_mismatch"].any()


def test_priority_queue_is_populated_and_banded(result):
    ex = result.exceptions
    assert ex["priority_score"].between(0, 100).all()
    assert set(ex["priority_band"]) <= {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    assert ex["priority_band"].nunique() >= 3             # the bands actually separate the queue


def test_persisting_a_full_run_round_trips(tmp_db, dataset):
    r = pipeline.run(directory=dataset, validation_as_of=VALIDATION_AS_OF, notes="e2e")
    k = repository.kpis(r.run_id)
    assert k["total_rows"] == r.total and k["matched_rows"] == r.matched_rows
    assert k["exceptions_open"] == r.exception_count
    assert len(repository.validation_failures_df(r.run_id, severity=["ERROR"])) > 0
    assert repository.exception_summary(r.run_id)["exception_count"].sum() == r.exception_count


def test_pair_level_precision_is_reported_and_exact_stages_are_trustworthy(result, dataset):
    """Row-level accuracy cannot see two clean rows wrongly paired with each other; this can."""
    p = evaluation.evaluate_pairs(result, evaluation.load_row_labels(dataset))
    assert p["pairs"] == len(result.matches) and 0 < p["precision_pct"] <= 100
    by = p["by_stage"].set_index("match_type")["precision_pct"]
    assert by["REFERENCE"] == 100.0 and by["EXACT"] == 100.0       # shared reference / identical fields never mis-pair
