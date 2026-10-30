"""
Multi-source ingestion: CSV / TSV / Excel  ->  validated rows + validation failures.

Real exports are messy, and v1 handled that by assuming fixed column names and
silently coercing anything it could not parse to NaT or 0. That is the wrong
failure mode for financial data: a mangled amount quietly becoming 0.00 is
worse than a loud rejection. This layer does the opposite:

* finds the header row (bank statements often start with account details),
* maps columns by alias ("Txn Date", "Transaction Date", "Date" -> ``date``),
* infers each column's date format from the data, and flags day/month ambiguity,
* validates every row, and *quarantines* what it cannot trust, recording why
  (``validation_failures``) rather than dropping or guessing,
* fingerprints the file (SHA-256) so a run can prove what it consumed.

Row ids are derived from the row's position in the original file, so every
failure, match and exception traces back to an exact line of the source.
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .preprocessing import finalize_frame

log = logging.getLogger(__name__)

# code -> (severity, row_excluded, meaning).  ERROR rows are quarantined; INFO rows are
# non-transactional lines that are skipped; WARNING rows are kept unless noted.
FAILURE_CODES = {
    "UNSUPPORTED_FORMAT": ("ERROR", True, "File extension is not CSV/TSV/Excel"),
    "UNREADABLE_FILE": ("ERROR", True, "File could not be opened or parsed"),
    "EMPTY_FILE": ("ERROR", True, "No data rows found below the header"),
    "HEADER_NOT_FOUND": ("ERROR", True, "No header row containing a date and an amount column was found"),
    "MISSING_COLUMN": ("WARNING", False, "A recommended column is absent"),
    "DUPLICATE_COLUMN": ("WARNING", False, "Two columns map to the same field; the first was used"),
    "AMBIGUOUS_DAY_MONTH": ("WARNING", False, "Dates are valid as both DD/MM and MM/DD; DD/MM assumed"),
    "DUPLICATE_FILE": ("WARNING", False, "An identical file was already reconciled in an earlier run"),
    "BLANK_ROW": ("INFO", True, "Empty row skipped"),
    "SUMMARY_ROW": ("INFO", True, "Total / balance line skipped (not a transaction)"),
    "MISSING_DATE": ("ERROR", True, "Transaction date is blank"),
    "UNPARSEABLE_DATE": ("ERROR", True, "Transaction date could not be interpreted"),
    "MISSING_AMOUNT": ("ERROR", True, "No amount present"),
    "UNPARSEABLE_AMOUNT": ("ERROR", True, "Amount is not a valid number"),
    "AMBIGUOUS_AMOUNT": ("ERROR", True, "Both debit and credit are populated"),
    "ZERO_AMOUNT": ("WARNING", True, "Zero-value row excluded"),
    "FUTURE_DATE": ("WARNING", False, "Date is after the as-of date"),
    "MISSING_VENDOR": ("WARNING", False, "Description / vendor is blank"),
    "MISSING_REFERENCE": ("WARNING", False, "Reference is blank where one is expected"),
}
FAILURE_COLUMNS = ["source_system", "file_name", "source_row", "row_id", "field", "code", "severity",
                   "row_excluded", "raw_value", "message"]


class IngestionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Source schemas (aliases are matched after stripping case, spaces and punctuation)
# ---------------------------------------------------------------------------

def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _aliases(**fields) -> dict:
    return {k: tuple(_norm(a) for a in v) for k, v in fields.items()}


@dataclass(frozen=True)
class SourceSchema:
    source: str
    aliases: dict
    date_formats: tuple = ()
    reference_expected: bool = False
    fixed_direction: str | None = None


SCHEMAS = {
    "BANK": SourceSchema("BANK", _aliases(
        date=["Txn Date", "Transaction Date", "Tran Date", "Date", "Posting Date"],
        description=["Narration", "Description", "Particulars", "Transaction Remarks", "Remarks",
                     "Transaction Description", "Details"],
        reference=["Chq/Ref No", "Chq Ref No", "Ref No", "Reference", "Reference No", "UTR", "UTR No",
                   "Cheque No", "Ref/Chq No", "Chq No"],
        debit=["Withdrawal Amt", "Withdrawal", "Withdrawals", "Debit", "Debit Amount", "Dr", "Dr Amount", "Paid Out"],
        credit=["Deposit Amt", "Deposit", "Deposits", "Credit", "Credit Amount", "Cr", "Cr Amount", "Paid In"],
        amount=["Amount", "Txn Amount", "Transaction Amount"],
        type=["Dr/Cr", "Type", "Txn Type", "Cr/Dr"],
    ), date_formats=("%d/%m/%Y",)),
    "LEDGER": SourceSchema("LEDGER", _aliases(
        date=["Entry Date", "Voucher Date", "Date", "Posting Date", "Txn Date", "Transaction Date"],
        description=["Vendor Name", "Vendor", "Party", "Party Name", "Account Name", "Ledger Name",
                     "Particulars", "Narration", "Name", "Description"],
        reference=["Voucher No", "Voucher Number", "Voucher", "Ref No", "Reference", "Doc No", "Document No"],
        debit=["Debit", "Dr", "Debit Amount"],
        credit=["Credit", "Cr", "Credit Amount"],
        amount=["Amount"],
        type=["Dr/Cr", "Type"],
    ), date_formats=("%Y-%m-%d",)),
    "PAYMENT_GATEWAY": SourceSchema("PAYMENT_GATEWAY", _aliases(
        date=["Settlement Date", "Settled On", "Date", "Txn Date", "Transaction Date", "Created At"],
        description=["Merchant Name", "Merchant", "Customer", "Customer Name", "Description"],
        reference=["UTR Number", "UTR", "Settlement UTR", "Bank Reference", "Reference", "UTR No"],
        amount=["Amount", "Settled Amount", "Net Amount", "Txn Amount", "Transaction Amount"],
        status=["Status", "Txn Status", "Payment Status"],
    ), date_formats=("ISO8601",), reference_expected=True, fixed_direction="CREDIT"),
    "ERP": SourceSchema("ERP", _aliases(
        date=["Invoice Date", "Doc Date", "Document Date", "Date", "Posting Date"],
        description=["Party Name", "Vendor", "Vendor Name", "Supplier", "Customer", "Customer Name", "Party"],
        reference=["Linked Voucher No", "Linked Voucher", "Voucher No", "GL Voucher No", "Reference"],
        amount=["Invoice Amount", "Amount", "Total", "Grand Total", "Invoice Value"],
        type=["Invoice Type", "Type", "Doc Type", "Document Type"],
    ), date_formats=("%d-%b-%y",), reference_expected=True),
}

_DEBIT_WORDS = {"dr", "debit", "d", "purchase", "purchases", "payment", "withdrawal", "out", "expense"}
_CREDIT_WORDS = {"cr", "credit", "c", "sales", "sale", "receipt", "deposit", "in", "income"}
_BLANK_MARKERS = {"-", "--", "\u2014", "\u2013", "nil", "null", "none", "nan", "nat"}
_SUMMARY_RE = re.compile(
    r"^\s*(grand\s+)?(total|sub\s*total|closing\s+balance|opening\s+balance|balance\s*(b/?f|c/?f)|"
    r"brought\s+forward|carried\s+forward)\b", re.I)
_CURRENCY_RE = re.compile(r"(?i)(\u20b9|rs\.?|inr|\$|usd|\u20ac|eur|\u00a3|gbp)")
_CONFLICT = {"%d/%m/%Y": "%m/%d/%Y", "%m/%d/%Y": "%d/%m/%Y"}


# ---------------------------------------------------------------------------
# Cell-level helpers
# ---------------------------------------------------------------------------

def _is_blank(v) -> bool:
    if v is None or v is pd.NaT:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str):
        return v.strip() == ""
    return False


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _short(v) -> str:
    return "" if _is_blank(v) else str(v)[:200]


def _clean_reference(v) -> str:
    if _is_blank(v):
        return ""
    if _is_number(v) and float(v).is_integer():
        return str(int(v))      # Excel stores 12345 as 12345.0
    return str(v).strip()


# ---------------------------------------------------------------------------
# Date and amount parsing
# ---------------------------------------------------------------------------

def _parse_with(values: pd.Index, fmt: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
    except (ValueError, TypeError):
        parsed = pd.DatetimeIndex([pd.NaT] * len(values))
    if getattr(parsed, "tz", None) is not None:
        parsed = parsed.tz_localize(None)
    return pd.Series(parsed.as_unit("ns"), index=values)


def _parse_strings(unique_values: pd.Index, preferred: tuple) -> tuple[pd.Series, bool]:
    """Pick the format that parses the most distinct values, then rescue stragglers
    with the remaining formats. Day-first and month-first are never mixed within a
    column: silently reading 03/04 as March in one row and April in the next is
    exactly the kind of error reconciliation exists to catch."""
    candidates = list(dict.fromkeys(list(preferred) + list(config.DATE_FORMATS)))
    parsed = {fmt: _parse_with(unique_values, fmt) for fmt in candidates}
    counts = {fmt: int(p.notna().sum()) for fmt, p in parsed.items()}
    best = max(candidates, key=lambda f: (counts[f], -candidates.index(f)))
    result = parsed[best].copy()

    conflict = _CONFLICT.get(best)
    ambiguous = False
    if conflict and counts[best] > 0 and counts[conflict] == counts[best]:
        both = parsed[best].notna() & parsed[conflict].notna()
        ambiguous = bool((parsed[best][both] != parsed[conflict][both]).any())

    missing = result.isna()
    for fmt in candidates:
        if not missing.any():
            break
        if fmt == best or fmt == conflict:
            continue
        fill = missing & parsed[fmt].notna()
        result[fill] = parsed[fmt][fill]
        missing = result.isna()
    return result, ambiguous


def parse_dates(series: pd.Series, preferred: tuple = ()) -> tuple[pd.Series, pd.Series, list]:
    """-> (timestamps at midnight, state in {'ok','blank','invalid'}, notes)."""
    notes: list[str] = []
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    state = pd.Series("invalid", index=series.index, dtype=object)

    blank = series.map(_is_blank).astype(bool)
    state[blank] = "blank"
    is_dt = series.map(lambda v: isinstance(v, (datetime, date, np.datetime64))).astype(bool) & ~blank
    is_num = series.map(_is_number).astype(bool) & ~blank & ~is_dt     # Excel serial numbers
    is_str = ~blank & ~is_dt & ~is_num

    if is_dt.any():
        out[is_dt] = pd.DatetimeIndex(pd.to_datetime(series[is_dt].tolist(), errors="coerce")).as_unit("ns")
    if is_num.any():
        serial = series[is_num].astype(float)
        plausible = serial[serial.between(20000, 80000)]
        if len(plausible):
            conv = pd.to_datetime(plausible, unit="D", origin="1899-12-30", errors="coerce")
            out[conv.index] = conv.dt.as_unit("ns")
    if is_str.any():
        strs = series[is_str].astype(str).str.strip()
        parsed_unique, ambiguous = _parse_strings(pd.Index(strs.unique()), preferred)
        out[is_str] = strs.map(parsed_unique).to_numpy()
        if ambiguous:
            notes.append("AMBIGUOUS_DAY_MONTH")

    out = out.dt.normalize()
    ok = out.notna() & out.dt.year.between(1990, 2100)
    out[~ok] = pd.NaT
    state[ok] = "ok"
    return out, state, notes


def parse_amounts(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """-> (float values, state in {'ok','blank','invalid'}). A lone dash or 'nil' is a
    blank (statements use it for 'no amount'); 'N/A' or letters are invalid."""
    values = pd.Series(np.nan, index=series.index, dtype=float)
    state = pd.Series("invalid", index=series.index, dtype=object)
    blank = series.map(_is_blank).astype(bool)
    is_num = series.map(_is_number).astype(bool) & ~blank
    is_str = ~blank & ~is_num

    marker_mask = pd.Series(False, index=series.index)
    if is_num.any():
        values[is_num] = series[is_num].astype(float)
    if is_str.any():
        s = series[is_str].astype(str).str.strip()
        marker_mask[is_str] = s.str.lower().isin(_BLANK_MARKERS).to_numpy()
        cleaned = s.str.replace(_CURRENCY_RE, "", regex=True).str.replace(r"[,\s]", "", regex=True)
        negative = cleaned.str.match(r"^\(.*\)$")
        cleaned = cleaned.str.replace(r"^\((.*)\)$", r"\1", regex=True)
        num = pd.to_numeric(cleaned, errors="coerce")
        num = num.where(~negative, -num)
        values[is_str] = num.to_numpy()
    blank = blank | marker_mask
    values[~np.isfinite(values.to_numpy(dtype=float))] = np.nan
    state[values.notna()] = "ok"
    state[blank] = "blank"
    values[blank] = np.nan
    return values, state


# ---------------------------------------------------------------------------
# Reading files into a raw grid
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_delimiter(text: str) -> str:
    lines = [ln for ln in text.splitlines()[:60] if ln.strip()]
    best, best_score = ",", -1.0
    for d in (",", ";", "\t", "|"):
        counts = [len(next(csv.reader([ln], delimiter=d))) for ln in lines]
        if not counts:
            continue
        modal = max(set(counts), key=counts.count)
        if modal < 2:
            continue
        score = counts.count(modal) / len(counts) * min(modal, 20)
        if score > best_score:
            best, best_score = d, score
    return best


def _read_delimited(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise IngestionError("UNREADABLE_FILE", "could not decode file as UTF-8 or Windows-1252")
    rows = list(csv.reader(io.StringIO(text), delimiter=_detect_delimiter(text)))
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows, dtype=object)


def _iter_grids(path: Path, sheet=None):
    """Yield (sheet_name, raw grid with no header applied). Excel workbooks are
    tried sheet by sheet so a cover sheet in front of the data is harmless."""
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv", ".txt"):
        yield None, _read_delimited(path)
        return
    engine = "xlrd" if ext == ".xls" else "openpyxl"
    try:
        workbook = pd.ExcelFile(path, engine=engine)
    except ImportError:
        raise IngestionError("UNREADABLE_FILE", "reading .xls files requires the optional 'xlrd' package")
    except Exception as exc:  # corrupt / password-protected / not really an Excel file
        raise IngestionError("UNREADABLE_FILE", f"could not open workbook: {exc}")
    names = [sheet] if sheet is not None else workbook.sheet_names
    for name in names:
        try:
            yield name, workbook.parse(name, header=None, dtype=object)
        except Exception as exc:
            raise IngestionError("UNREADABLE_FILE", f"could not read sheet {name!r}: {exc}")


def _score_header(cells, schema: SourceSchema):
    mapping: dict[str, tuple[int, str]] = {}
    duplicates: list[str] = []
    for j, cell in enumerate(cells):
        key = _norm(cell) if not _is_blank(cell) else ""
        if not key:
            continue
        for canon, aliases in schema.aliases.items():
            if key in aliases:
                if canon in mapping:
                    duplicates.append(str(cell).strip())
                else:
                    mapping[canon] = (j, str(cell).strip())
                break
    valid = "date" in mapping and any(k in mapping for k in ("amount", "debit", "credit"))
    return (len(mapping) if valid else 0), mapping, duplicates


def _locate_table(path: Path, schema: SourceSchema, sheet=None):
    last_error = None
    for name, grid in _iter_grids(path, sheet):
        if grid.empty:
            continue
        best_score, best_row, best_map, best_dups = 0, None, {}, []
        for i in range(min(config.HEADER_SCAN_ROWS, len(grid))):
            score, mapping, dups = _score_header(grid.iloc[i].tolist(), schema)
            if score > best_score:
                best_score, best_row, best_map, best_dups = score, i, mapping, dups
        if best_row is not None:
            return grid, best_row, best_map, best_dups, name
        last_error = name
    raise IngestionError(
        "HEADER_NOT_FOUND",
        "no header row with a date column and an amount (or debit/credit) column was found in the first "
        f"{config.HEADER_SCAN_ROWS} rows" + (f" of sheet {last_error!r}" if last_error else ""))


# ---------------------------------------------------------------------------
# Row validation
# ---------------------------------------------------------------------------

def _failure_rows(source, file_name, mask: pd.Series, code, field_name, raw: pd.Series | None, message) -> pd.DataFrame:
    idx = mask.index[mask.to_numpy(dtype=bool)]
    if len(idx) == 0:
        return pd.DataFrame(columns=FAILURE_COLUMNS)
    severity, excluded, _ = FAILURE_CODES[code]
    return pd.DataFrame({
        "source_system": source, "file_name": file_name, "source_row": idx.astype(int),
        "row_id": [f"{source}-{int(i):07d}" for i in idx], "field": field_name, "code": code,
        "severity": severity, "row_excluded": excluded,
        "raw_value": raw.loc[idx].map(_short).to_numpy() if raw is not None else "",
        "message": message,
    })


def _type_direction(series: pd.Series) -> pd.Series:
    t = series.map(lambda v: "" if _is_blank(v) else str(v).strip().lower())
    out = pd.Series(None, index=series.index, dtype=object)
    out[t.isin(_DEBIT_WORDS)] = "DEBIT"
    out[t.isin(_CREDIT_WORDS)] = "CREDIT"
    return out


def _process_rows(source: str, schema: SourceSchema, table: pd.DataFrame, file_name: str, as_of: pd.Timestamp):
    n = len(table)
    idx = table.index

    def col(name):
        return table[name] if name in table else pd.Series([None] * n, index=idx, dtype=object)

    fails: list[pd.DataFrame] = []
    add = lambda mask, code, field_name, raw, msg: fails.append(  # noqa: E731
        _failure_rows(source, file_name, mask, code, field_name, raw, msg))

    blank_row = table.map(_is_blank).all(axis=1) if table.shape[1] else pd.Series(True, index=idx)
    desc_raw = col("description")
    desc = desc_raw.map(lambda v: "" if _is_blank(v) else str(v).strip())

    dates, dstate, notes = parse_dates(col("date"), schema.date_formats)

    # --- amounts: debit/credit columns, or a single (optionally signed / typed) amount ---
    dual = "debit" in table or "credit" in table
    if dual:
        d_val, d_state = parse_amounts(col("debit"))
        c_val, c_state = parse_amounts(col("credit"))
        invalid = (d_state == "invalid") | (c_state == "invalid")
        missing = (d_state == "blank") & (c_state == "blank")
        d_pos, c_pos = d_val.fillna(0).abs() > 0, c_val.fillna(0).abs() > 0
        ambiguous = d_pos & c_pos
        amount = pd.Series(np.where(d_pos, d_val.abs(), c_val.abs()), index=idx).astype(float)
        direction = pd.Series(np.where(d_pos, "DEBIT", "CREDIT"), index=idx, dtype=object)
        zero = ~invalid & ~missing & ~ambiguous & ~d_pos & ~c_pos
        raw_amount = col("debit").where(~col("debit").map(_is_blank), col("credit"))
        amount_field = "debit/credit"
    else:
        a_val, a_state = parse_amounts(col("amount"))
        invalid, missing = a_state == "invalid", a_state == "blank"
        ambiguous = pd.Series(False, index=idx)
        amount = a_val.abs()
        zero = a_val == 0
        typed = _type_direction(col("type")) if "type" in table else pd.Series(None, index=idx, dtype=object)
        by_sign = pd.Series(np.where(a_val < 0, "DEBIT", schema.fixed_direction or "CREDIT"), index=idx, dtype=object)
        direction = typed.where(typed.notna(), by_sign)
        raw_amount = col("amount")
        amount_field = "amount"

    # --- classify rows ---
    summary = desc.map(lambda s: bool(_SUMMARY_RE.match(s))) & ((dstate != "ok") | missing) & ~blank_row
    skip = blank_row | summary
    live = ~skip
    err_date_blank = live & (dstate == "blank")
    err_date_bad = live & (dstate == "invalid")
    err_amt_bad = live & invalid
    err_amt_missing = live & missing & ~invalid
    err_amt_ambig = live & ambiguous
    error = err_date_blank | err_date_bad | err_amt_bad | err_amt_missing | err_amt_ambig
    zero_row = live & ~error & zero
    keep = live & ~error & ~zero

    add(blank_row, "BLANK_ROW", "", None, "row is empty")
    add(summary & ~blank_row, "SUMMARY_ROW", "description", desc_raw, "total / balance line, not a transaction")
    add(err_date_blank, "MISSING_DATE", "date", col("date"), "transaction date is blank")
    add(err_date_bad, "UNPARSEABLE_DATE", "date", col("date"), "date could not be interpreted")
    add(err_amt_bad, "UNPARSEABLE_AMOUNT", amount_field, raw_amount, "amount is not a valid number")
    add(err_amt_missing, "MISSING_AMOUNT", amount_field, raw_amount, "no amount present")
    add(err_amt_ambig, "AMBIGUOUS_AMOUNT", amount_field, raw_amount, "both debit and credit populated")
    add(zero_row, "ZERO_AMOUNT", amount_field, raw_amount, "zero-value row excluded")
    future = keep & (dates > as_of + pd.Timedelta(days=config.MAX_FUTURE_DAYS))
    add(future, "FUTURE_DATE", "date", col("date"), f"date is after the as-of date {as_of.date()}")
    if "description" in table:
        add(keep & (desc == ""), "MISSING_VENDOR", "description", desc_raw, "description / vendor is blank")
    reference = col("reference").map(_clean_reference)
    if schema.reference_expected and "reference" in table:
        add(keep & (reference == ""), "MISSING_REFERENCE", "reference", col("reference"), "reference is blank")

    accepted = pd.DataFrame({
        "source_row": idx[keep.to_numpy()].astype(int),
        "txn_date": dates[keep], "vendor_raw": desc[keep], "reference": reference[keep],
        "amount": amount[keep].round(2), "direction": direction[keep],
    }).reset_index(drop=True)
    if "status" in table:
        accepted["status"] = col("status")[keep].map(lambda v: "" if _is_blank(v) else str(v).strip()).to_numpy()

    nonempty = [f for f in fails if len(f)]
    failures = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=FAILURE_COLUMNS)
    return accepted, failures, notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    source: str
    file_name: str
    file_path: str = ""
    sha256: str = ""
    sheet_name: str | None = None
    header_row: int | None = None
    column_mapping: dict = field(default_factory=dict)
    rows_read: int = 0
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=FAILURE_COLUMNS))

    @property
    def rows_accepted(self) -> int:
        return len(self.df)

    @property
    def rows_excluded(self) -> int:
        return max(self.rows_read - self.rows_accepted, 0)

    @property
    def fatal(self) -> bool:
        """True if the file as a whole could not be used (header / format / read failure)."""
        if self.failures.empty:
            return False
        file_level = self.failures["source_row"].isna() & (self.failures["severity"] == "ERROR")
        return bool(file_level.any())

    def summary(self) -> dict:
        by_code = self.failures["code"].value_counts().to_dict() if not self.failures.empty else {}
        return {"source": self.source, "file": self.file_name, "rows_read": self.rows_read,
                "rows_accepted": self.rows_accepted, "rows_excluded": self.rows_excluded,
                "failures_by_code": by_code}


def _file_failure(source, file_name, code, message, field_name="") -> pd.DataFrame:
    severity, excluded, _ = FAILURE_CODES[code]
    return pd.DataFrame([{"source_system": source, "file_name": file_name, "source_row": pd.NA, "row_id": None,
                          "field": field_name, "code": code, "severity": severity, "row_excluded": excluded,
                          "raw_value": "", "message": message}], columns=FAILURE_COLUMNS)


def ingest_file(source: str, path, *, as_of=None, sheet=None) -> IngestionResult:
    """Read, map, validate and normalise one source file. Never raises for bad *data*:
    problems come back as ``result.failures`` (check ``result.fatal``)."""
    path = Path(path)
    schema = SCHEMAS[source]
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(date.today())
    result = IngestionResult(source=source, file_name=path.name, file_path=str(path))
    file_fails: list[pd.DataFrame] = []

    def fail(code, message, field_name=""):
        file_fails.append(_file_failure(source, path.name, code, message, field_name))

    def finish():
        parts = [f for f in file_fails + [result.failures] if len(f)]
        result.failures = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=FAILURE_COLUMNS)
        if result.df.empty:
            result.df = finalize_frame(pd.DataFrame({
                "source_row": pd.Series(dtype=int), "txn_date": pd.Series(dtype="datetime64[ns]"),
                "vendor_raw": pd.Series(dtype=str), "reference": pd.Series(dtype=str),
                "amount": pd.Series(dtype=float), "direction": pd.Series(dtype=str)}), source)
        return result

    if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        fail("UNSUPPORTED_FORMAT", f"unsupported file type {path.suffix!r}; expected one of "
                                   f"{', '.join(config.SUPPORTED_EXTENSIONS)}")
        return finish()
    try:
        result.sha256 = _sha256(path)
        grid, header_row, mapping, dup_cols, sheet_name = _locate_table(path, schema, sheet)
    except OSError as exc:
        fail("UNREADABLE_FILE", f"could not read file: {exc}")
        return finish()
    except IngestionError as exc:
        fail(exc.code, str(exc))
        return finish()

    result.sheet_name, result.header_row = sheet_name, header_row
    result.column_mapping = {canon: header for canon, (_, header) in mapping.items()}
    for header in dup_cols:
        fail("DUPLICATE_COLUMN", f"column {header!r} also maps to an already-mapped field", header)
    for needed in ("description",) + (("reference",) if schema.reference_expected else ()):
        if needed not in mapping:
            fail("MISSING_COLUMN", f"no '{needed}' column found; rows will have an empty {needed}", needed)

    data = grid.iloc[header_row + 1:].reset_index(drop=True)
    if data.empty or data.map(_is_blank).all(axis=None):
        fail("EMPTY_FILE", "no data rows below the header")
        return finish()
    blank_cells = data.map(_is_blank).all(axis=1).to_numpy()
    last = int(np.flatnonzero(~blank_cells)[-1])          # trailing formatting-only rows are not data
    data = data.iloc[: last + 1]
    result.rows_read = len(data)

    table = pd.DataFrame({canon: data.iloc[:, j] for canon, (j, _) in mapping.items()})
    accepted, failures, notes = _process_rows(source, schema, table, path.name, as_of)
    for note in notes:
        fail(note, "every date is valid as both DD/MM and MM/DD; DD/MM was assumed", "date")
    result.failures = failures
    result.df = finalize_frame(accepted, source) if len(accepted) else pd.DataFrame()
    log.info("ingested %s: %d rows read, %d accepted, %d excluded", path.name, result.rows_read,
             result.rows_accepted, result.rows_excluded)
    return finish()


def find_source_file(directory, source: str):
    """Locate ``<stem>.<ext>`` for a source, preferring Excel over CSV if both exist."""
    directory = Path(directory)
    stem = config.SOURCE_FILE_STEMS[source]
    for ext in (".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt"):
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_sources(directory) -> dict:
    found = {}
    for source in config.SOURCES:
        path = find_source_file(directory, source)
        if path is not None:
            found[source] = path
    return found
