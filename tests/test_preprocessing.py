import numpy as np
import pandas as pd

from recon.preprocessing import finalize_frame, normalize_vendor


def test_bank_narration_collapses_to_vendor_token():
    assert normalize_vendor("NEFT-TCS-563349") == "TCS"
    assert normalize_vendor("RTGS-RELIANCERETA-294419") == "RELIANCERETA"


def test_company_suffixes_are_stripped_as_whole_words():
    assert normalize_vendor("Zoho Corporation") == "ZOHO"
    assert normalize_vendor("Dutta Ltd") == "DUTTA"
    assert normalize_vendor("Pingle PLC") == "PINGLE"


def test_hyphenated_surname_is_not_mistaken_for_a_reference_suffix():
    """Regression: a trailing-reference regex once ate '-IYENGAR', so the ledger said GUHA
    while the bank narration said GUHAIYENGAR and the two never matched exactly."""
    assert normalize_vendor("Guha-Iyengar") == "GUHAIYENGAR"
    assert normalize_vendor("RTGS-GUHAIYENGAR-294419") == "GUHAIYENGAR"
    assert normalize_vendor("Guha-Iyengar") == normalize_vendor("RTGS-GUHAIYENGAR-294419")


def test_missing_values_normalise_to_empty_string():
    assert normalize_vendor(None) == ""
    assert normalize_vendor(float("nan")) == ""
    assert normalize_vendor("   ") == ""


def _rows(**over):
    base = {"source_row": [0, 1, 2], "txn_date": pd.to_datetime(["2026-08-12"] * 3),
            "vendor_raw": ["TCS", "TCS", "Infosys"], "reference": ["R1", "R1", "R2"],
            "amount": [45000.0, 45000.0, 100.0], "direction": ["DEBIT"] * 3}
    base.update(over)
    return pd.DataFrame(base)


def test_finalize_builds_row_ids_from_source_position():
    out = finalize_frame(_rows(source_row=[7, 8, 9]), "BANK")
    assert list(out["row_id"]) == ["BANK-0000007", "BANK-0000008", "BANK-0000009"]
    assert out["txn_date"].dtype == np.dtype("datetime64[ns]")


def test_exact_repeat_is_flagged_as_duplicate_but_first_occurrence_is_not():
    out = finalize_frame(_rows(), "LEDGER")
    assert list(out["is_duplicate"]) == [False, True, False]


def test_same_amount_different_reference_is_not_a_duplicate():
    out = finalize_frame(_rows(reference=["R1", "R9", "R2"]), "LEDGER")
    assert not out["is_duplicate"].any()


def test_empty_frame_finalises_cleanly():
    empty = pd.DataFrame({"source_row": pd.Series(dtype=int), "txn_date": pd.Series(dtype="datetime64[ns]"),
                          "vendor_raw": pd.Series(dtype=str), "reference": pd.Series(dtype=str),
                          "amount": pd.Series(dtype=float), "direction": pd.Series(dtype=str)})
    out = finalize_frame(empty, "ERP")
    assert out.empty and "row_id" in out.columns
