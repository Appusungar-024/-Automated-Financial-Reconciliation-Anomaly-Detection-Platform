"""Shared fixtures. Every test that touches the database gets its own throwaway SQLite file."""
import csv
import os

import pytest

from recon import config, db, pipeline

VALIDATION_AS_OF = "2026-12-31"      # keeps FUTURE_DATE checks independent of the day the tests run

BANK_HEADER = ["Txn_Date", "Value_Date", "Narration", "Chq_Ref_No", "Withdrawal_Amt", "Deposit_Amt"]
LEDGER_HEADER = ["Entry_Date", "Vendor_Name", "Voucher_No", "Debit", "Credit", "Account_Head"]


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return path


@pytest.fixture
def tmp_db(tmp_path):
    """An isolated database per test. SQLite by default; set TEST_DATABASE_URL to run the very same
    suite against PostgreSQL (the schema is dropped and rebuilt for every test)."""
    url = os.getenv("TEST_DATABASE_URL")
    db.configure(url or f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    if url:
        db.drop_all()
    db.init_db()
    yield db
    db.get_engine().dispose()
    db._engine = None


@pytest.fixture
def mini_files(tmp_path):
    """Two sources, four rows each, with a known story:
         0  TCS 45,000      matches exactly
         1  Acme            amount mismatch (bank 10,000 vs ledger 8,000)
         2  Zenith / Gamma  each missing from the other side
         3  Beta 700        matches (via date+amount; the bank narration differs)"""
    bank = write_rows(tmp_path / "bank.csv", [BANK_HEADER,
        ["12/08/2026", "12/08/2026", "NEFT-TCS-111111", "B1", "45000.00", ""],
        ["13/08/2026", "13/08/2026", "NEFT-ACME-222222", "B2", "10000.00", ""],
        ["14/08/2026", "14/08/2026", "NEFT-ZENITH-333333", "B3", "500.00", ""],
        ["15/08/2026", "15/08/2026", "IMPS-BETA-444444", "B4", "700.00", ""]])
    ledger = write_rows(tmp_path / "ledger.csv", [LEDGER_HEADER,
        ["2026-08-12", "TCS", "V1", "45000.00", "", "Purchases"],
        ["2026-08-13", "Acme", "V2", "8000.00", "", "Purchases"],
        ["2026-08-15", "Beta Traders", "V3", "700.00", "", "Purchases"],
        ["2026-08-16", "Gamma", "V4", "900.00", "", "Purchases"]])
    return {"BANK": bank, "LEDGER": ledger}


@pytest.fixture
def mini_run(tmp_db, mini_files):
    return pipeline.run(mini_files, validation_as_of=VALIDATION_AS_OF, notes="mini scenario", actor="tester")
