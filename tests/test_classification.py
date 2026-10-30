import pandas as pd

from recon import classification, config
from tests.helpers import make_frame, pair

BL = [pair("BANK_VS_LEDGER")]
GB = [pair("GATEWAY_VS_BANK")]


def classify(frames, pairs, matched=None):
    """Treat every row as unmatched unless listed; returns {row_id: (type, suggested)}."""
    remaining = dict(frames)
    matched_ids = matched or {s: set() for s in frames}
    if matched:
        remaining = {s: f[~f["row_id"].isin(matched[s])] for s, f in frames.items()}
    out = classification.classify_exceptions(frames, remaining, matched_ids, pairs)
    return out.set_index("row_id")


def test_amount_mismatch_is_named_and_points_at_its_counterpart():
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-1", "B1", 45000.0)])
    ledger = make_frame("LEDGER", [("2026-08-12", "TCS", "V1", 40000.0)])
    out = classify({"BANK": bank, "LEDGER": ledger}, BL)
    assert out.loc["BANK-0000000", "exception_type"] == "AMOUNT_MISMATCH"
    assert out.loc["BANK-0000000", "suggested_match_row_id"] == "LEDGER-0000000"
    assert out.loc["LEDGER-0000000", "suggested_match_row_id"] == "BANK-0000000"
    assert "5,000.00" in out.loc["BANK-0000000", "suggested_match_reason"]


def test_date_mismatch_is_named():
    bank = make_frame("BANK", [("2026-08-01", "NEFT-TCS-1", "B1", 45000.0)])
    ledger = make_frame("LEDGER", [("2026-08-20", "TCS", "V1", 45000.0)])
    out = classify({"BANK": bank, "LEDGER": ledger}, BL)
    assert set(out["exception_type"]) == {"DATE_MISMATCH"}
    assert "19 days" in out.loc["BANK-0000000", "suggested_match_reason"]


def test_unrelated_vendor_is_missing_even_when_date_and_amount_are_close():
    """Regression for v1's core flaw: with a blended score, an unrelated transaction with a
    similar amount outscored the true counterpart. Vendor is the identity gate."""
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-1", "B1", 45000.0)])
    ledger = make_frame("LEDGER", [("2026-08-12", "Zenith Foods", "V1", 43000.0)])
    out = classify({"BANK": bank, "LEDGER": ledger}, BL)
    assert set(out["exception_type"]) == {"MISSING"}
    assert out["suggested_match_row_id"].isna().all()


def test_true_counterpart_beats_an_unrelated_row_with_a_closer_amount():
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-1", "B1", 45000.0)])
    ledger = make_frame("LEDGER", [("2026-08-12", "TCS", "V1", 40500.0),            # right vendor, 10% off
                                    ("2026-08-12", "Zenith Foods", "V2", 44990.0)])  # wrong vendor, near-identical amount
    out = classify({"BANK": bank, "LEDGER": ledger}, BL)
    assert out.loc["BANK-0000000", "suggested_match_row_id"] == "LEDGER-0000000"
    assert out.loc["LEDGER-0000001", "exception_type"] == "MISSING"


def test_no_counterpart_at_all_is_missing():
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-1", "B1", 45000.0)])
    out = classify({"BANK": bank, "LEDGER": make_frame("LEDGER", [])}, BL)
    assert out.loc["BANK-0000000", "exception_type"] == "MISSING"


def test_only_mutual_pairings_are_suggested():
    """Two bank rows chase one ledger row: only the closer one is paired, the other is MISSING."""
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-1", "B1", 45000.0),
                                ("2026-08-12", "NEFT-TCS-2", "B2", 38000.0)])
    ledger = make_frame("LEDGER", [("2026-08-12", "TCS", "V1", 44000.0)])
    out = classify({"BANK": bank, "LEDGER": ledger}, BL)
    assert out.loc["BANK-0000000", "exception_type"] == "AMOUNT_MISMATCH"
    assert out.loc["BANK-0000001", "exception_type"] == "MISSING"


def test_extra_copy_of_a_reconciled_row_is_a_duplicate_not_missing():
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-1", "B1", 100.0)] * 2)
    ledger = make_frame("LEDGER", [("2026-08-12", "TCS", "V1", 100.0)])
    out = classify({"BANK": bank, "LEDGER": ledger}, BL,
                   matched={"BANK": {"BANK-0000000"}, "LEDGER": {"LEDGER-0000000"}})
    assert list(out.index) == ["BANK-0000001"] and out.iloc[0]["exception_type"] == "DUPLICATE"


def test_a_repeated_row_that_never_reconciled_is_still_a_duplicate():
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-1", "B1", 100.0)] * 2)
    out = classify({"BANK": bank, "LEDGER": make_frame("LEDGER", [])}, BL)
    assert set(out["exception_type"]) == {"DUPLICATE"}


def test_shared_reference_makes_a_counterpart_eligible_despite_a_different_vendor_name():
    gateway = make_frame("PAYMENT_GATEWAY", [("2026-08-10", "Rajan Bava and Dhora", "UTR111", 7000.0)])
    bank = make_frame("BANK", [("2026-08-10", "NEFT-SOMETHINGELSE-1", "UTR111", 6000.0)])
    out = classify({"PAYMENT_GATEWAY": gateway, "BANK": bank}, GB)
    assert set(out["exception_type"]) == {"AMOUNT_MISMATCH"}
    assert "reference matches" in out.iloc[0]["suggested_match_reason"]


def test_shared_reference_is_ignored_where_schemes_are_unrelated():
    bank = make_frame("BANK", [("2026-08-10", "NEFT-AAAA-1", "REF777", 7000.0)])
    ledger = make_frame("LEDGER", [("2026-08-10", "Totally Different", "REF777", 6000.0)])
    out = classify({"BANK": bank, "LEDGER": ledger}, BL)
    assert set(out["exception_type"]) == {"MISSING"}


def test_exceptions_carry_the_pairs_their_source_belongs_to():
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-1", "B1", 45000.0)])
    out = classify({"BANK": bank, "LEDGER": make_frame("LEDGER", [])}, BL)
    assert out.loc["BANK-0000000", "pair_name"] == "BANK_VS_LEDGER"
