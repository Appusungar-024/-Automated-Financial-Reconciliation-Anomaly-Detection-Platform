import pandas as pd

from recon import matching
from tests.helpers import make_frame, pair

BANK_LEDGER = pair("BANK_VS_LEDGER")
GATEWAY_BANK = pair("GATEWAY_VS_BANK")


def stages(result):
    return {s.stage: s.matched_pairs for s in result.stage_stats}


def test_identical_rows_match_exactly():
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-563349", "B1", 45000.0)])
    ledger = make_frame("LEDGER", [("2026-08-12", "TCS", "V1", 45000.0)])
    res = matching.reconcile(bank, ledger, BANK_LEDGER)
    assert stages(res)["EXACT"] == 1 and res.exceptions.empty
    assert res.matches.loc[0, "match_type"] == "EXACT" and res.matches.loc[0, "match_confidence"] == 100.0


def test_the_brief_example_amount_mismatch_does_not_match():
    """Bank 45,000 vs Ledger 40,000 on the same day for the same vendor: must NOT auto-match."""
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-563349", "B1", 45000.0)])
    ledger = make_frame("LEDGER", [("2026-08-12", "TCS", "V1", 40000.0)])
    res = matching.reconcile(bank, ledger, BANK_LEDGER)
    assert res.matches.empty and len(res.exceptions) == 2


def test_each_row_is_claimed_at_most_once():
    """A genuine duplicate posting must pair 1-to-1, not explode into a cartesian join."""
    bank = make_frame("BANK", [("2026-08-12", "NEFT-TCS-1", "B1", 100.0)] * 2)
    ledger = make_frame("LEDGER", [("2026-08-12", "TCS", "V1", 100.0)])
    res = matching.reconcile(bank, ledger, BANK_LEDGER)
    assert len(res.matches) == 1
    assert len(res.exceptions) == 1 and res.exceptions.iloc[0]["source_role"] == "a"
    assert res.matches["row_id_a"].is_unique and res.matches["row_id_b"].is_unique


def test_vendor_text_is_ignored_at_the_date_amount_stage():
    bank = make_frame("BANK", [("2026-08-12", "NEFT-RELIANCERETA-1", "B1", 7000.0)])
    ledger = make_frame("LEDGER", [("2026-08-12", "Reliance Retail Limited", "V1", 7000.0)])
    res = matching.reconcile(bank, ledger, BANK_LEDGER)
    assert stages(res)["DATE_AMOUNT"] == 1


def test_amount_tolerance_is_max_of_absolute_and_percentage():
    ledger = make_frame("LEDGER", [("2026-08-12", "Acme", "V1", 45000.0)])
    inside = make_frame("BANK", [("2026-08-12", "NEFT-ACME-1", "B1", 45200.0)])     # 0.44% < 0.5%
    assert stages(matching.reconcile(inside, ledger, BANK_LEDGER))["AMOUNT_TOLERANCE"] == 1
    tiny_l = make_frame("LEDGER", [("2026-08-12", "Acme", "V1", 100.0)])
    tiny_b = make_frame("BANK", [("2026-08-12", "NEFT-ACME-1", "B1", 104.0)])        # within the Rs.5 floor
    assert stages(matching.reconcile(tiny_b, tiny_l, BANK_LEDGER))["AMOUNT_TOLERANCE"] == 1


def test_gap_beyond_every_stage_tolerance_is_not_matched():
    ledger = make_frame("LEDGER", [("2026-08-12", "Acme", "V1", 45000.0)])
    beyond = make_frame("BANK", [("2026-08-12", "NEFT-ACME-1", "B1", 45800.0)])     # 1.8%: > 0.5% and > 1%
    res = matching.reconcile(beyond, ledger, BANK_LEDGER)
    assert res.matches.empty and len(res.exceptions) == 2


def test_stage_four_accepts_a_slightly_wider_amount_gap_when_vendor_and_date_are_strong():
    """0.67% fails the 0.5% tolerance stage, but a same-vendor same-day row is rescued by the fuzzy stage (1%)."""
    ledger = make_frame("LEDGER", [("2026-08-12", "Acme", "V1", 45000.0)])
    bank = make_frame("BANK", [("2026-08-12", "NEFT-ACME-1", "B1", 45300.0)])
    assert stages(matching.reconcile(bank, ledger, BANK_LEDGER))["FUZZY_VENDOR"] == 1


def test_fuzzy_stage_rescues_date_drift_with_a_truncated_vendor_prefix():
    bank = make_frame("BANK", [("2026-08-12", "NEFT-KRISHNANGROU-1", "B1", 9000.0)])   # bank truncates the name
    ledger = make_frame("LEDGER", [("2026-08-16", "Krishnan Group", "V1", 9000.0)])    # 4 days later
    res = matching.reconcile(bank, ledger, BANK_LEDGER)
    assert stages(res)["FUZZY_VENDOR"] == 1 and res.matches.loc[0, "date_diff_days"] == 4


def test_fuzzy_stage_refuses_unrelated_vendors_and_wide_date_gaps():
    ledger = make_frame("LEDGER", [("2026-08-16", "Krishnan Group", "V1", 9000.0)])
    other_vendor = make_frame("BANK", [("2026-08-12", "NEFT-ZENITHFOODS-1", "B1", 9000.0)])
    far_date = make_frame("BANK", [("2026-08-06", "NEFT-KRISHNANGROU-1", "B1", 9000.0)])  # 10 days
    assert matching.reconcile(other_vendor, ledger, BANK_LEDGER).matches.empty
    assert matching.reconcile(far_date, ledger, BANK_LEDGER).matches.empty


def test_reference_stage_matches_through_vendor_noise_where_references_are_comparable():
    gateway = make_frame("PAYMENT_GATEWAY", [("2026-08-10", "Rajan Bava and Dhora", "UTR111", 7000.0)])
    bank = make_frame("BANK", [("2026-08-10", "NEFT-SOMETHINGELSE-1", "UTR111", 7000.0)])
    res = matching.reconcile(gateway, bank, GATEWAY_BANK)
    assert res.matches.loc[0, "match_type"] == "REFERENCE" and not res.matches.loc[0, "reference_mismatch"]


def test_reference_agreement_does_not_wave_through_a_different_amount_or_date():
    """Regression: matching on reference alone swallowed injected amount/date mismatches,
    silently turning real discrepancies into 'matches'."""
    gateway = make_frame("PAYMENT_GATEWAY", [("2026-08-10", "Acme Traders", "UTR111", 7000.0)])
    bank_amount = make_frame("BANK", [("2026-08-10", "NEFT-ACMETRADERS-1", "UTR111", 6000.0)])
    bank_date = make_frame("BANK", [("2026-08-25", "NEFT-ACMETRADERS-1", "UTR111", 7000.0)])
    assert "REFERENCE" not in set(matching.reconcile(gateway, bank_amount, GATEWAY_BANK).matches["match_type"])
    assert matching.reconcile(gateway, bank_amount, GATEWAY_BANK).matches.empty
    assert matching.reconcile(gateway, bank_date, GATEWAY_BANK).matches.empty


def test_reference_mismatch_is_flagged_only_where_references_are_comparable():
    gw = make_frame("PAYMENT_GATEWAY", [("2026-08-10", "Acme", "UTR111", 7000.0)])
    bk = make_frame("BANK", [("2026-08-10", "NEFT-ACME-1", "UTR999", 7000.0)])
    flagged = matching.reconcile(gw, bk, GATEWAY_BANK).matches
    assert flagged.loc[0, "reference_mismatch"]                                     # real signal here
    led = make_frame("LEDGER", [("2026-08-10", "Acme", "V555", 7000.0)])
    unrelated = matching.reconcile(bk, led, BANK_LEDGER).matches
    assert not unrelated.loc[0, "reference_mismatch"]                               # unrelated schemes: no noise


def test_every_match_records_its_decision_rationale():
    bank = make_frame("BANK", [("2026-08-12", "NEFT-ACME-1", "B1", 45200.0)])
    ledger = make_frame("LEDGER", [("2026-08-12", "Acme", "V1", 45000.0)])
    m = matching.reconcile(bank, ledger, BANK_LEDGER).matches.iloc[0]
    assert m["amount_diff"] == -200.0 and m["date_diff_days"] == 0
    assert m["vendor_similarity"] == 100.0 and m["decided_by"] == "system"
    assert "tolerance" in m["rule_applied"].lower()


def test_empty_inputs_are_handled():
    empty = make_frame("BANK", [])
    ledger = make_frame("LEDGER", [("2026-08-12", "Acme", "V1", 1.0)])
    res = matching.reconcile(empty, ledger, BANK_LEDGER)
    assert res.matches.empty and len(res.exceptions) == 1
