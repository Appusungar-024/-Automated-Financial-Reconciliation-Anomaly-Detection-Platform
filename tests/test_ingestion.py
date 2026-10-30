import csv
import hashlib
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from recon import config
from recon.ingestion import (find_source_file, ingest_file, parse_amounts, parse_dates)

AS_OF = "2026-09-30"
EXCEL_SERIAL_2026_08_29 = (date(2026, 8, 29) - date(1899, 12, 30)).days


def write_csv(path, rows, delimiter=","):
    with open(path, "w", newline="", encoding="utf-8") as fh:      # real CSV quoting ("1,200.50")
        csv.writer(fh, delimiter=delimiter).writerows(rows)
    return path


def write_xlsx(path, sheets):
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(path)
    return path


BANK_HEADER = ["Txn_Date", "Value_Date", "Narration", "Chq_Ref_No", "Withdrawal_Amt", "Deposit_Amt"]


def codes(res, **where):
    f = res.failures
    for k, v in where.items():
        f = f[f[k] == v]
    return list(f["code"])


# ---------------------------------------------------------------- happy paths

def test_csv_roundtrip_maps_columns_and_signs(tmp_path):
    p = write_csv(tmp_path / "bank.csv", [
        BANK_HEADER,
        ["12/08/2026", "12/08/2026", "NEFT-TCS-563349", "REF1", "45000.00", ""],
        ["13/08/2026", "13/08/2026", "IMPS-INFOSYS-111111", "REF2", "", "1,200.50"],
    ])
    res = ingest_file("BANK", p, as_of=AS_OF)
    assert not res.fatal and res.failures.empty
    assert list(res.df["row_id"]) == ["BANK-0000000", "BANK-0000001"]
    assert list(res.df["amount"]) == [45000.0, 1200.5]
    assert list(res.df["direction"]) == ["DEBIT", "CREDIT"]
    assert list(res.df["txn_date"]) == [pd.Timestamp("2026-08-12"), pd.Timestamp("2026-08-13")]
    assert res.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()


def test_alias_headers_and_statement_preamble(tmp_path):
    """Bank exports start with account details, and every bank names its columns differently."""
    p = write_csv(tmp_path / "stmt.csv", [
        ["Account Statement"], ["Account No: 1234567890", "", "", ""], [""],
        ["Transaction Date", "Particulars", "Debit", "Credit", "Ref No"],
        ["01/09/2026", "UPI-ZOHO-777777", "500", "", "R9"],
    ])
    res = ingest_file("BANK", p, as_of=AS_OF)
    assert res.header_row == 3
    assert res.column_mapping == {"date": "Transaction Date", "description": "Particulars",
                                   "debit": "Debit", "credit": "Credit", "reference": "Ref No"}
    assert res.rows_read == 1 and res.rows_accepted == 1
    assert res.df.loc[0, "row_id"] == "BANK-0000000"          # numbering starts below the header


def test_semicolon_delimited_file_is_detected(tmp_path):
    p = write_csv(tmp_path / "ledger.csv", [
        ["Entry_Date", "Vendor_Name", "Voucher_No", "Debit", "Credit"],
        ["2026-08-12", "Acme, Inc", "V1", "100.00", ""]], delimiter=";")
    res = ingest_file("LEDGER", p, as_of=AS_OF)
    assert res.rows_accepted == 1 and res.df.loc[0, "vendor_raw"] == "Acme, Inc"


def test_excel_with_preamble_real_dates_and_numeric_cells(tmp_path):
    p = write_xlsx(tmp_path / "bank.xlsx", {"Statement": [
        ["HDFC Bank - Statement of Account"], ["Period: Aug 2026"], [],
        BANK_HEADER,
        [datetime(2026, 8, 12), datetime(2026, 8, 12), "NEFT-TCS-563349", "REF1", 45000.0, None],
        [datetime(2026, 8, 13), datetime(2026, 8, 13), "IMPS-INFOSYS-1", 12345, None, 250.75],
    ]})
    res = ingest_file("BANK", p, as_of=AS_OF)
    assert not res.fatal and res.rows_accepted == 2
    assert res.sheet_name == "Statement" and res.header_row == 3
    assert list(res.df["reference"]) == ["REF1", "12345"]       # Excel's 12345.0 is not "12345.0"
    assert list(res.df["amount"]) == [45000.0, 250.75]


def test_excel_cover_sheet_in_front_of_data_is_skipped(tmp_path):
    p = write_xlsx(tmp_path / "ledger.xlsx", {
        "Cover": [["Quarterly ledger export"], ["Prepared by finance"]],
        "Data": [["Entry_Date", "Vendor_Name", "Voucher_No", "Debit", "Credit"],
                 [datetime(2026, 8, 1), "Acme", "V1", 10.0, None]]})
    res = ingest_file("LEDGER", p, as_of=AS_OF)
    assert res.sheet_name == "Data" and res.rows_accepted == 1


def test_single_amount_with_type_column_for_erp(tmp_path):
    p = write_csv(tmp_path / "erp.csv", [
        ["invoice_date", "party_name", "invoice_amount", "invoice_type", "linked_voucher_no"],
        ["12-Aug-26", "Acme", "1000", "Purchase", "V1"],
        ["13-Aug-26", "Beta", "2000", "Sales", "V2"]])
    res = ingest_file("ERP", p, as_of=AS_OF)
    assert list(res.df["direction"]) == ["DEBIT", "CREDIT"]
    assert list(res.df["txn_date"]) == [pd.Timestamp("2026-08-12"), pd.Timestamp("2026-08-13")]


# ---------------------------------------------------------------- validation

def test_bad_rows_are_quarantined_with_reasons_and_stable_row_ids(tmp_path):
    p = write_csv(tmp_path / "bank.csv", [
        BANK_HEADER,
        ["12/08/2026", "", "NEFT-OK-1", "R1", "100.00", ""],         # 0 good
        ["31/02/2026", "", "NEFT-BADDATE-2", "R2", "100.00", ""],     # 1 impossible date
        ["13/08/2026", "", "NEFT-BADAMT-3", "R3", "N/A", ""],         # 2 non-numeric amount
        ["", "", "TOTAL", "", "999.00", ""],                           # 3 summary line
        ["", "", "", "", "", ""],                                      # 4 blank
        ["14/08/2026", "", "NEFT-ZERO-4", "R4", "0", ""],              # 5 zero value
        ["15/08/2026", "", "NEFT-BOTH-5", "R5", "10", "10"],           # 6 debit AND credit
        ["16/08/2026", "", "NEFT-NOAMT-6", "R6", "", ""],              # 7 no amount
        ["17/08/2026", "", "NEFT-OK-7", "R7", "", "55.00"],            # 8 good
    ])
    res = ingest_file("BANK", p, as_of=AS_OF)
    assert list(res.df["row_id"]) == ["BANK-0000000", "BANK-0000008"]
    assert res.rows_read == 9 and res.rows_excluded == 7
    by_row = res.failures.groupby("row_id")["code"].apply(set).to_dict()
    assert by_row == {
        "BANK-0000001": {"UNPARSEABLE_DATE"},
        "BANK-0000002": {"UNPARSEABLE_AMOUNT"},
        "BANK-0000003": {"SUMMARY_ROW"},
        "BANK-0000004": {"BLANK_ROW"},
        "BANK-0000005": {"ZERO_AMOUNT"},
        "BANK-0000006": {"AMBIGUOUS_AMOUNT"},
        "BANK-0000007": {"MISSING_AMOUNT"},
    }
    bad = res.failures.set_index("row_id")
    assert bad.loc["BANK-0000002", "raw_value"] == "N/A"
    assert bad.loc["BANK-0000001", "severity"] == "ERROR"
    assert bad.loc["BANK-0000004", "severity"] == "INFO"


def test_row_with_two_problems_records_both_and_is_excluded_once(tmp_path):
    p = write_csv(tmp_path / "bank.csv", [BANK_HEADER, ["TBD", "", "NEFT-X-1", "R1", "abc", ""],
                                          ["12/08/2026", "", "NEFT-OK-2", "R2", "5", ""]])
    res = ingest_file("BANK", p, as_of=AS_OF)
    assert set(codes(res, row_id="BANK-0000000")) == {"UNPARSEABLE_DATE", "UNPARSEABLE_AMOUNT"}
    assert res.rows_accepted == 1 and res.rows_excluded == 1


def test_a_vendor_called_total_is_not_mistaken_for_a_summary_line(tmp_path):
    p = write_csv(tmp_path / "ledger.csv", [
        ["Entry_Date", "Vendor_Name", "Voucher_No", "Debit", "Credit"],
        ["2026-08-12", "Total Logistics Pvt Ltd", "V1", "100", ""]])
    res = ingest_file("LEDGER", p, as_of=AS_OF)
    assert res.rows_accepted == 1 and res.failures.empty


def test_future_dates_and_missing_vendor_warn_but_keep_the_row(tmp_path):
    p = write_csv(tmp_path / "ledger.csv", [
        ["Entry_Date", "Vendor_Name", "Voucher_No", "Debit", "Credit"],
        ["2026-12-25", "", "V1", "100", ""]])
    res = ingest_file("LEDGER", p, as_of=AS_OF)
    assert res.rows_accepted == 1
    assert set(res.failures["code"]) == {"FUTURE_DATE", "MISSING_VENDOR"}
    assert (res.failures["severity"] == "WARNING").all() and not res.failures["row_excluded"].any()


def test_missing_reference_only_warns_for_sources_that_should_have_one(tmp_path):
    erp = write_csv(tmp_path / "erp.csv", [
        ["invoice_date", "party_name", "invoice_amount", "invoice_type", "linked_voucher_no"],
        ["12-Aug-26", "Acme", "1000", "Purchase", ""]])
    assert "MISSING_REFERENCE" in codes(ingest_file("ERP", erp, as_of=AS_OF))
    bank = write_csv(tmp_path / "bank.csv", [BANK_HEADER, ["12/08/2026", "", "NEFT-A-1", "", "5", ""]])
    assert "MISSING_REFERENCE" not in codes(ingest_file("BANK", bank, as_of=AS_OF))


# ---------------------------------------------------------------- file-level failures

def test_unsupported_extension_is_a_fatal_file_failure(tmp_path):
    p = tmp_path / "bank.pdf"
    p.write_bytes(b"%PDF-1.4")
    res = ingest_file("BANK", p, as_of=AS_OF)
    assert res.fatal and codes(res) == ["UNSUPPORTED_FORMAT"] and res.df.empty


def test_file_without_a_recognisable_header_is_fatal(tmp_path):
    p = write_csv(tmp_path / "bank.csv", [["foo", "bar"], ["1", "2"]])
    res = ingest_file("BANK", p, as_of=AS_OF)
    assert res.fatal and codes(res) == ["HEADER_NOT_FOUND"]


def test_header_only_file_reports_empty(tmp_path):
    res = ingest_file("BANK", write_csv(tmp_path / "bank.csv", [BANK_HEADER]), as_of=AS_OF)
    assert res.fatal and "EMPTY_FILE" in codes(res)


def test_corrupt_excel_is_reported_not_raised(tmp_path):
    p = tmp_path / "bank.xlsx"
    p.write_bytes(b"this is not a zip archive")
    res = ingest_file("BANK", p, as_of=AS_OF)
    assert res.fatal and codes(res) == ["UNREADABLE_FILE"]


def test_missing_description_column_is_a_warning_not_a_failure(tmp_path):
    p = write_csv(tmp_path / "bank.csv", [["Txn_Date", "Withdrawal_Amt", "Deposit_Amt"], ["12/08/2026", "10", ""]])
    res = ingest_file("BANK", p, as_of=AS_OF)
    assert not res.fatal and res.rows_accepted == 1
    assert "MISSING_COLUMN" in codes(res)


# ---------------------------------------------------------------- date inference

def _dates(values, preferred=()):
    out, state, notes = parse_dates(pd.Series(values, dtype=object), preferred)
    return list(out), list(state), notes


def test_day_first_is_chosen_when_the_column_proves_it():
    out, state, notes = _dates(["25/07/2026", "03/04/2026"], ("%d/%m/%Y",))
    assert out == [pd.Timestamp("2026-07-25"), pd.Timestamp("2026-04-03")] and notes == []


def test_month_first_is_chosen_when_the_column_proves_it():
    out, _, notes = _dates(["07/25/2026", "03/04/2026"], ("%d/%m/%Y",))
    assert out == [pd.Timestamp("2026-07-25"), pd.Timestamp("2026-03-04")] and notes == []


def test_a_genuinely_ambiguous_column_warns_instead_of_guessing_silently():
    out, _, notes = _dates(["03/04/2026", "05/06/2026"], ("%d/%m/%Y",))
    assert out[0] == pd.Timestamp("2026-04-03")                   # day-first assumed...
    assert notes == ["AMBIGUOUS_DAY_MONTH"]                       # ...and said so


def test_mixed_formats_excel_serials_and_junk_in_one_column():
    out, state, _ = _dates(["2026-08-12", "12-Aug-26", datetime(2026, 8, 14), EXCEL_SERIAL_2026_08_29, "N/A", "", None])
    assert out[0] == pd.Timestamp("2026-08-12") and out[1] == pd.Timestamp("2026-08-12")
    assert out[2] == pd.Timestamp("2026-08-14")
    assert out[3] == pd.Timestamp("2026-08-29")                   # Excel serial date
    assert state == ["ok", "ok", "ok", "ok", "invalid", "blank", "blank"]


def test_iso_datetime_with_time_component_is_normalised_to_midnight():
    out, _, _ = _dates(["2026-08-12T14:23:01", "2026-08-13T00:00:00"], ("ISO8601",))
    assert out == [pd.Timestamp("2026-08-12"), pd.Timestamp("2026-08-13")]


# ---------------------------------------------------------------- amount parsing

def test_amount_formats():
    vals, state = parse_amounts(pd.Series(["\u20b91,23,456.78", "(500)", "Rs. 99", 42, 3.5, "-", "nil", "", None,
                                           "N/A", "12,34,5.6.7", "abc"], dtype=object))
    assert vals.iloc[:5].tolist() == [123456.78, -500.0, 99.0, 42.0, 3.5]
    assert list(state) == ["ok"] * 5 + ["blank"] * 4 + ["invalid"] * 3


# ---------------------------------------------------------------- discovery

def test_find_source_file_prefers_excel_over_csv(tmp_path):
    (tmp_path / "bank.csv").write_text("x")
    assert find_source_file(tmp_path, "BANK").suffix == ".csv"
    (tmp_path / "bank.xlsx").write_bytes(b"x")
    assert find_source_file(tmp_path, "BANK").suffix == ".xlsx"
    assert find_source_file(tmp_path, "ERP") is None
