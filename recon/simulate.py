"""
Synthetic multi-source data with held-out ground truth.

Real bank and ledger exports are confidential, so a portfolio project has to
simulate -- and the usual trap is data so clean that every detector looks
perfect. This generator builds "true" business transactions, projects each into
two source systems with a *labelled* defect (or none), and writes the labels to
files the pipeline never reads. ``recon.evaluation`` can then report measured
precision and recall instead of an impression.

What gets simulated
-------------------
* Four sources in their own authentic shapes: Bank (noisy, truncated
  narrations, debit/credit columns, DD/MM/YYYY), Ledger (clean vendor names,
  ISO dates), Payment Gateway (ISO timestamps, settlement UTR) and ERP
  (DD-Mon-YY, linked voucher numbers). Any source can be written as Excel with
  a statement preamble above the header.
* Discrepancies: missing, duplicate, amount / date / reference mismatches, each
  with an "auto-resolvable" and a "genuine exception" variant.
* Four anomaly patterns: round-number spikes on weekends, structuring just
  under a reporting threshold, reference reuse, and large payments to brand-new
  vendors.
* Junk rows a real file contains: unparseable dates and amounts, TOTAL /
  CLOSING BALANCE lines, blank rows -- to exercise the ingestion validator.
* Weekday-heavy dates. v1 drew dates uniformly, so 29% of transactions landed
  on weekends and "weekend" meant nothing; B2B payments are not like that.

Caveat, stated plainly: results measured on simulated defects show that the
pipeline finds *these patterns*; they are not a claim about real-world accuracy.
"""
from __future__ import annotations

import argparse
import json
import random
import string
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from openpyxl import Workbook

from . import config

TXN_CODES = ["NEFT", "IMPS", "RTGS", "UPI", "CHQ"]
ACCOUNT_HEADS = ["Sundry Debtors", "Sundry Creditors", "Bank Charges", "Sales Revenue", "Purchases",
                 "Professional Fees", "Rent", "Salaries"]
DATE_FORMATS = {"BANK": "%d/%m/%Y", "LEDGER": "%Y-%m-%d", "PAYMENT_GATEWAY": "%Y-%m-%dT%H:%M:%S", "ERP": "%d-%b-%y"}
SHEET_PREAMBLE = {
    "BANK": [["Acme Bank Ltd"], ["Statement of Account"], ["Account No: 50100123456789"], []],
    "LEDGER": [["General Ledger Export"], ["Company: Demo Traders Pvt Ltd"], []],
    "PAYMENT_GATEWAY": [["Settlement Report"], ["Merchant ID: M-100234"], []],
    "ERP": [["ERP Invoice Register"], []],
}
JUNK_COLUMNS = {   # which column holds what, per source (for building junk rows)
    "BANK": {"date": "Txn_Date", "desc": "Narration", "ref": "Chq_Ref_No", "amount": "Withdrawal_Amt"},
    "LEDGER": {"date": "Entry_Date", "desc": "Vendor_Name", "ref": "Voucher_No", "amount": "Debit"},
    "PAYMENT_GATEWAY": {"date": "settlement_date", "desc": "merchant_name", "ref": "utr_number", "amount": "amount"},
    "ERP": {"date": "invoice_date", "desc": "party_name", "ref": "linked_voucher_no", "amount": "invoice_amount"},
}
JUNK_KINDS = ["JUNK_BAD_DATE", "JUNK_BAD_AMOUNT", "JUNK_SUMMARY", "JUNK_BLANK"]
JUNK_EXPECTED_SEVERITY = {"JUNK_BAD_DATE": "ERROR", "JUNK_BAD_AMOUNT": "ERROR", "JUNK_SUMMARY": "INFO", "JUNK_BLANK": "INFO"}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def build_vendor_pool(py_rng: random.Random, total_transactions: int, seed: int, avg_txns_per_vendor: int = 8):
    """Regular vendors (many transactions -> a spending history to learn from) and occasional ones.
    Sized relative to volume: a fixed pool means ~185 transactions per vendor at 50,000, which is not a
    real vendor base and creates ambiguity collisions that say nothing about the matcher."""
    total_vendors = max(270, total_transactions // avg_txns_per_vendor)
    n_regular = int(total_vendors * 2 / 3)
    fake = Faker("en_IN")
    fake.seed_instance(seed)
    flavor = ["TCS", "Infosys", "Reliance Retail", "Zoho Corporation", "Flipkart Internet", "Wipro", "HDFC Bank", "Swiggy"]
    names = set(flavor)
    while len(names) < total_vendors:
        names.add(fake.company())
    names = sorted(names)
    py_rng.shuffle(names)
    return names[:n_regular], names[n_regular:total_vendors], fake


def clean_token(vendor_name: str, max_len: int | None = None) -> str:
    token = "".join(ch for ch in vendor_name.upper() if ch.isalnum())
    return token[:max_len] if max_len else token


def make_reference(rng, prefix: str, n_digits: int = 10) -> str:
    return prefix + "".join(str(d) for d in rng.integers(0, 10, size=n_digits))


def random_amount(rng, round_bias: float = 0.15) -> float:
    magnitude = rng.choice([1_000, 5_000, 20_000, 75_000, 300_000], p=[0.30, 0.30, 0.25, 0.12, 0.03])
    amount = float(rng.uniform(0.3, 1.8) * magnitude)
    if rng.random() < round_bias:
        amount = max(round(amount / 1000) * 1000, 1000)     # rounding a sub-500 amount used to yield 0.00
    return round(amount, 2)


def sample_dates(rng, start: date, end: date, n: int) -> list:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    weights = np.array([config.WEEKDAY_WEIGHTS[d.weekday()] for d in days], dtype=float)
    picks = rng.choice(len(days), size=n, p=weights / weights.sum())
    return [days[i] for i in picks]


class TrueTxn:
    __slots__ = ("txn_id", "pair", "date", "vendor", "amount", "ref_a", "ref_b", "direction")

    def __init__(self, txn_id, pair, dt, vendor, amount, ref_a, ref_b, direction):
        self.txn_id, self.pair, self.date, self.vendor = txn_id, pair, dt, vendor
        self.amount, self.ref_a, self.ref_b, self.direction = amount, ref_a, ref_b, direction


def _new_txn(rng, pair_cfg, txn_id, dt, vendor, amount):
    if pair_cfg["use_reference_check"]:       # the same reference travels through both systems
        shared = make_reference(rng, "UTR" if pair_cfg["source_a"] == "PAYMENT_GATEWAY" else "REF")
        ref_a = ref_b = shared
    else:                                      # Bank UTRs and Ledger vouchers are unrelated schemes
        ref_a, ref_b = make_reference(rng, "REF"), make_reference(rng, "V")
    direction = "CREDIT" if pair_cfg["source_a"] == "PAYMENT_GATEWAY" else ("DEBIT" if rng.random() < 0.55 else "CREDIT")
    return TrueTxn(txn_id, pair_cfg["name"], dt, vendor, amount, ref_a, ref_b, direction)


def generate_true_transactions(rng, pair_cfg, count, vendor_pool, start, end, id_offset):
    regulars, occasional = vendor_pool
    dates = sample_dates(rng, start, end, count)
    txns = []
    for i in range(count):
        vendor = str(rng.choice(regulars)) if rng.random() < 0.85 else str(rng.choice(occasional))
        txns.append(_new_txn(rng, pair_cfg, id_offset + i, dates[i], vendor, random_amount(rng)))
    return txns


def _gt(t, pair_cfg, dtype, expected, note):
    return {"true_txn_id": t.txn_id, "pair": pair_cfg["name"], "date": t.date, "vendor": t.vendor,
            "amount": t.amount, "discrepancy_type": dtype, "expected_outcome": expected, "note": note}


def _rows(t):
    base = {"_true_txn_id": t.txn_id, "date": t.date, "vendor": t.vendor, "amount": t.amount, "direction": t.direction}
    return {**base, "reference": t.ref_a}, {**base, "reference": t.ref_b}


def inject(rng, pair_cfg, true_txns):
    """Project each true transaction into source A and B, applying a labelled defect. Every record
    carries a hidden _true_txn_id/_discrepancy that only the evaluation ever reads."""
    records_a, records_b, truth = [], [], []
    use_ref = pair_cfg["use_reference_check"]
    types = list(config.DISCREPANCY_RATES)
    weights = np.array(list(config.DISCREPANCY_RATES.values()))
    kinds = rng.choice(types, size=len(true_txns), p=weights / weights.sum())

    for t, kind in zip(true_txns, kinds):
        row_a, row_b = _rows(t)
        label, expected, note = str(kind), "MATCHED", "clean_match"

        if kind == "MISSING":
            side = "A" if rng.random() < 0.5 else "B"
            (records_a if side == "A" else records_b).append({**(row_a if side == "A" else row_b), "_discrepancy": label})
            truth.append(_gt(t, pair_cfg, label, "MISSING", f"{side}_only"))
            continue
        if kind == "DUPLICATE":
            dup_a = rng.random() < 0.5            # one side posts the row twice, the other once
            for _ in range(2 if dup_a else 1):
                records_a.append({**row_a, "_discrepancy": label})      # separate dicts: no aliasing
            for _ in range(1 if dup_a else 2):
                records_b.append({**row_b, "_discrepancy": label})
            truth.append(_gt(t, pair_cfg, label, "DUPLICATE", "duplicated_one_side"))
            continue
        if kind == "AMOUNT_MISMATCH":
            if rng.random() < 0.4:                 # tolerance-level: should auto-resolve
                delta = rng.uniform(0.5, config.AMOUNT_TOLERANCE_ABS * 0.9)
                label, expected = "AMOUNT_MISMATCH_SMALL", "MATCHED"
            else:                                  # genuine: must NOT auto-resolve
                delta = max(300.0, t.amount * rng.uniform(0.03, 0.20))
                label, expected = "AMOUNT_MISMATCH_LARGE", "AMOUNT_MISMATCH"
            row_b["amount"] = round(row_b["amount"] + rng.choice([-1, 1]) * delta, 2)
            note = "amount_perturbed_on_b"
        elif kind == "DATE_MISMATCH":
            if rng.random() < 0.5:
                offset = int(rng.integers(config.DATE_TOLERANCE_DAYS + 1, config.FUZZY_DATE_WINDOW_DAYS + 1))
                label, expected = "DATE_MISMATCH_MODERATE", "MATCHED"
            else:
                offset = int(rng.integers(config.FUZZY_DATE_WINDOW_DAYS + 3, 25))
                label, expected = "DATE_MISMATCH_LARGE", "DATE_MISMATCH"
            row_b["date"] = row_b["date"] + timedelta(days=int(rng.choice([-1, 1])) * offset)
            note = "date_shifted_on_b"
        elif kind == "REFERENCE_MISMATCH":
            if use_ref:
                row_b["reference"] = make_reference(rng, "OLDV")
                note = "reference_garbled_on_b"
            else:                                  # references are not comparable for this pair: a plain match
                label, note = "MATCH", "reference_check_not_applicable_for_pair"
        elif kind == "SUSPICIOUS_ROUND_SPIKE":
            spike = float(rng.uniform(5, 10))
            amount = float(round(t.amount * spike / config.ROUND_AMOUNT_DIVISOR) * config.ROUND_AMOUNT_DIVISOR) \
                or float(config.ROUND_AMOUNT_DIVISOR * 5)
            row_a["amount"] = row_b["amount"] = amount
            shift = (5 - t.date.weekday()) % 7 + (1 if rng.random() < 0.3 else 0)       # a Saturday (or Sunday)
            row_a["date"] = row_b["date"] = t.date + timedelta(days=shift)
            note = "round_weekend_vendor_spike"

        records_a.append({**row_a, "_discrepancy": label})
        records_b.append({**row_b, "_discrepancy": label})
        truth.append(_gt(t, pair_cfg, label, expected, note))
    return records_a, records_b, truth


def inject_extra_anomalies(rng, pair_cfg, records_a, records_b, truth, vendor_pool, fake, start, end, id_offset):
    """Anomaly patterns that span several transactions or need brand-new entities. Added to the
    primary Bank<->Ledger pair; each reconciles cleanly (they are suspicious, not mismatched)."""
    n_pair = len([g for g in truth])
    count = int(round(config.EXTRA_ANOMALY_RATE * n_pair))
    count = max(count, 2) if n_pair >= 300 else 0
    if count == 0:
        return id_offset
    regulars, _ = vendor_pool
    weekdays = [d for d in (start + timedelta(days=i) for i in range((end - start).days + 1)) if d.weekday() < 5]
    next_id = id_offset

    def add(t, label, note):
        a, b = _rows(t)
        records_a.append({**a, "_discrepancy": label})
        records_b.append({**b, "_discrepancy": label})
        truth.append(_gt(t, pair_cfg, label, "MATCHED", note))

    # 1) structuring: 2-3 payments to one vendor within 3 days, each just under the reporting threshold
    for _ in range(count):
        vendor = str(rng.choice(regulars))
        base = weekdays[int(rng.integers(0, max(1, len(weekdays) - 4)))]
        for k in range(int(rng.integers(2, 4))):
            amount = round(float(rng.uniform(0.92, 0.999) * config.STRUCTURING_THRESHOLD), 2)
            add(_new_txn(rng, pair_cfg, next_id, base + timedelta(days=k), vendor, amount), "SUSPICIOUS_STRUCTURING",
                "payments_just_under_threshold")
            next_id += 1
    # 2) large payment to a brand-new vendor
    taken = set(regulars) | set(vendor_pool[1])
    for _ in range(count):
        name = fake.company()
        while name in taken:
            name = fake.company()
        taken.add(name)
        amount = round(float(rng.uniform(1_500_000, 4_000_000)), 2)
        add(_new_txn(rng, pair_cfg, next_id, weekdays[int(rng.integers(0, len(weekdays)))], name, amount),
            "SUSPICIOUS_NEW_VENDOR", "large_payment_to_new_vendor")
        next_id += 1
    # 3) reference reuse: a bank reference appears twice on different amounts
    clean = [i for i, r in enumerate(records_a) if r["_discrepancy"] == "MATCH"]
    picks = rng.choice(clean, size=min(2 * count, len(clean) // 2 * 2), replace=False)
    for first, second in zip(picks[::2], picks[1::2]):
        records_a[int(second)]["reference"] = records_a[int(first)]["reference"]
        for idx in (int(first), int(second)):
            records_a[idx]["_discrepancy"] = "SUSPICIOUS_REF_REUSE"
        involved = {records_a[int(first)]["_true_txn_id"], records_a[int(second)]["_true_txn_id"]}
        for r in records_b:
            if r["_true_txn_id"] in involved:
                r["_discrepancy"] = "SUSPICIOUS_REF_REUSE"
        for g in truth:
            if g["true_txn_id"] in involved:
                g["discrepancy_type"], g["note"] = "SUSPICIOUS_REF_REUSE", "bank_reference_reused"
    return next_id


# ---------------------------------------------------------------------------
# Source-specific formatting
# ---------------------------------------------------------------------------

def _random_chars(rng, n, length):
    chars = np.array(list(string.ascii_uppercase + string.digits))
    return ["".join(row) for row in rng.choice(chars, size=(n, length))]


def to_bank_df(records, rng):
    codes = rng.choice(TXN_CODES, size=len(records))
    value_shift = rng.integers(0, 2, size=len(records))
    rows = []
    for r, code, shift in zip(records, codes, value_shift):
        debit = r["direction"] == "DEBIT"
        rows.append({
            "Txn_Date": r["date"].strftime("%d/%m/%Y"),
            "Value_Date": (r["date"] + timedelta(days=int(shift))).strftime("%d/%m/%Y"),
            "Narration": f"{code}-{clean_token(r['vendor'], 12)}-{r['reference'][-6:]}",
            "Chq_Ref_No": r["reference"],
            "Withdrawal_Amt": r["amount"] if debit else "", "Deposit_Amt": "" if debit else r["amount"]})
    return pd.DataFrame(rows)


def to_ledger_df(records, rng):
    heads = rng.choice(ACCOUNT_HEADS, size=len(records))
    return pd.DataFrame([{
        "Entry_Date": r["date"].strftime("%Y-%m-%d"), "Vendor_Name": r["vendor"], "Voucher_No": r["reference"],
        "Debit": r["amount"] if r["direction"] == "DEBIT" else "", "Credit": "" if r["direction"] == "DEBIT" else r["amount"],
        "Account_Head": h} for r, h in zip(records, heads)])


def to_gateway_df(records, rng):
    hours, minutes = rng.integers(9, 21, size=len(records)), rng.integers(0, 60, size=len(records))
    txn_ids = _random_chars(rng, len(records), 10)
    statuses = rng.choice(["SUCCESS"] * 9 + ["REFUNDED"], size=len(records))
    return pd.DataFrame([{
        "settlement_date": datetime.combine(r["date"], datetime.min.time()).replace(hour=int(h), minute=int(m)).strftime("%Y-%m-%dT%H:%M:%S"),
        "merchant_name": r["vendor"], "gateway_txn_id": "PG" + tid, "amount": r["amount"], "status": s,
        "utr_number": r["reference"]} for r, h, m, tid, s in zip(records, hours, minutes, txn_ids, statuses)])


def to_erp_df(records, rng):
    numbers = rng.integers(1000, 9999, size=len(records))
    return pd.DataFrame([{
        "invoice_date": r["date"].strftime("%d-%b-%y"), "party_name": r["vendor"],
        "invoice_no": f"INV-{r['date']:%y%m}-{n}", "invoice_amount": r["amount"],
        "invoice_type": "Purchase" if r["direction"] == "DEBIT" else "Sales",
        "linked_voucher_no": r["reference"]} for r, n in zip(records, numbers)])


FORMATTERS = {"BANK": to_bank_df, "LEDGER": to_ledger_df, "PAYMENT_GATEWAY": to_gateway_df, "ERP": to_erp_df}


# ---------------------------------------------------------------------------
# Junk rows
# ---------------------------------------------------------------------------

def _junk_row(source: str, kind: str, columns: list, rng, start: date) -> dict:
    cols = JUNK_COLUMNS[source]
    row = {c: "" for c in columns}
    when = (start + timedelta(days=int(rng.integers(0, 60)))).strftime(DATE_FORMATS[source])
    row[cols["ref"]] = f"JNK{int(rng.integers(10**7, 10**8))}"
    if kind == "JUNK_BAD_DATE":
        row[cols["date"]] = str(rng.choice(["31/02/2026", "N/A", "TBD", "2026-13-45", "00/00/0000"]))
        row[cols["desc"]], row[cols["amount"]] = "NEFT-JUNKVENDOR-000001", round(float(rng.uniform(100, 5000)), 2)
    elif kind == "JUNK_BAD_AMOUNT":
        row[cols["date"]], row[cols["desc"]] = when, "NEFT-JUNKVENDOR-000002"
        row[cols["amount"]] = str(rng.choice(["N/A", "abc", "12,34,5.6.7", "#REF!"]))
    elif kind == "JUNK_SUMMARY":
        row[cols["desc"]] = str(rng.choice(["TOTAL", "CLOSING BALANCE", "OPENING BALANCE", "Grand Total"]))
        row[cols["amount"]] = round(float(rng.uniform(1e5, 9e6)), 2)
        row[cols["ref"]] = ""
    else:                                          # JUNK_BLANK
        row = {c: "" for c in columns}
    return row


def inject_junk(df: pd.DataFrame, labels: list, source: str, rng, start: date, rate: float):
    """Insert junk rows at random interior positions (never first or last: a trailing blank row is
    trimmed as formatting, not reported, and the evaluation needs every junk row to be observable)."""
    n_junk = int(round(rate * len(df))) if len(df) >= 200 else 0
    if n_junk == 0:
        return df, labels
    positions = sorted(rng.choice(np.arange(1, len(df)), size=n_junk, replace=False).tolist())
    kinds = [JUNK_KINDS[i % len(JUNK_KINDS)] for i in range(n_junk)]
    records = df.to_dict("records")
    columns = list(df.columns)
    out_rows, out_labels, cursor = [], [], 0
    for pos, kind in zip(positions, kinds):
        out_rows += records[cursor:pos]
        out_labels += labels[cursor:pos]
        out_rows.append(_junk_row(source, kind, columns, rng, start))
        out_labels.append((-1, kind))
        cursor = pos
    out_rows += records[cursor:]
    out_labels += labels[cursor:]
    return pd.DataFrame(out_rows, columns=columns), out_labels


# ---------------------------------------------------------------------------
# Writing files
# ---------------------------------------------------------------------------

def _excel_cell(value, source: str, column: str):
    if value == "" or value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if column == JUNK_COLUMNS[source]["date"] or column in ("Value_Date",):
        try:
            return datetime.strptime(str(value), DATE_FORMATS[source])        # real date cells, like Excel exports
        except ValueError:
            return value                                                       # junk stays as the text it was
    return value


def write_source(df: pd.DataFrame, path: Path, source: str, as_excel: bool) -> None:
    if not as_excel:
        df.to_csv(path, index=False)
        return
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Statement")
    for line in SHEET_PREAMBLE[source]:
        ws.append(line)
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append([_excel_cell(v, source, c) for v, c in zip(row, df.columns)])
    wb.save(path)


def main(total_transactions: int, seed: int = config.RANDOM_SEED, out_dir=None, end_date: date | None = None,
         excel_sources=(), junk_rate: float = config.JUNK_ROW_RATE, days: int = 90) -> dict:
    rng, py_rng = np.random.default_rng(seed), random.Random(seed)
    out_dir = Path(out_dir or config.RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    end_date = end_date or (date.today() - timedelta(days=1))
    start_date = end_date - timedelta(days=days)
    excel_sources = {s.upper() for s in excel_sources}

    regulars, occasional, fake = build_vendor_pool(py_rng, total_transactions, seed)
    per_source = {s: [] for s in config.SOURCES}
    truth, next_id = [], 0
    for pair in config.RECON_PAIRS:
        n = int(total_transactions * pair["share"])
        txns = generate_true_transactions(rng, pair, n, (regulars, occasional), start_date, end_date, next_id)
        next_id += n
        recs_a, recs_b, gt = inject(rng, pair, txns)
        if pair["name"] == "BANK_VS_LEDGER":
            next_id = inject_extra_anomalies(rng, pair, recs_a, recs_b, gt, (regulars, occasional), fake,
                                             start_date, end_date, next_id)
        per_source[pair["source_a"]] += recs_a
        per_source[pair["source_b"]] += recs_b
        truth += gt

    written = {}
    for source, records in per_source.items():
        if not records:
            continue
        py_rng.shuffle(records)
        df = FORMATTERS[source](records, rng)
        labels = [(r["_true_txn_id"], r["_discrepancy"]) for r in records]
        df, labels = inject_junk(df, labels, source, rng, start_date, junk_rate)
        as_excel = source in excel_sources
        stem = config.SOURCE_FILE_STEMS[source]
        for old in out_dir.glob(f"{stem}.*"):               # never leave a stale twin that discovery might prefer
            old.unlink()
        path = out_dir / f"{stem}.{'xlsx' if as_excel else 'csv'}"
        write_source(df, path, source, as_excel)
        pd.DataFrame({"row_index": range(len(labels)), "true_txn_id": [l[0] for l in labels],
                      "discrepancy": [l[1] for l in labels]}).to_csv(out_dir / f"_debug_{stem}_mapping.csv", index=False)
        written[source] = {"file": path.name, "rows": len(df), "format": "xlsx" if as_excel else "csv",
                           "junk_rows": sum(1 for l in labels if l[0] == -1)}

    pd.DataFrame(truth).to_csv(out_dir / "_ground_truth.csv", index=False)
    manifest = {"total_transactions": total_transactions, "seed": seed, "end_date": end_date.isoformat(),
                "days": days, "excel_sources": sorted(excel_sources), "junk_rate": junk_rate, "files": written,
                "true_transactions": len(truth)}
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def cli():
    parser = argparse.ArgumentParser(description="Generate synthetic Bank / Ledger / Payment Gateway / ERP files")
    parser.add_argument("--num-transactions", type=int, default=5000, help="base transactions across all pairs")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--excel", default="", help="comma-separated sources to write as .xlsx, e.g. BANK,ERP")
    parser.add_argument("--end-date", default=None, help="last transaction date (YYYY-MM-DD); default yesterday")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    m = main(args.num_transactions, args.seed, args.out_dir,
             date.fromisoformat(args.end_date) if args.end_date else None,
             [s for s in args.excel.split(",") if s])
    print(f"Generated {m['true_transactions']:,} true transactions ({m['total_transactions']:,} requested)")
    for source, info in m["files"].items():
        print(f"  {info['file']:24s} {info['rows']:>8,} rows  [{info['format']}]  incl. {info['junk_rows']} junk rows")
    print("  _ground_truth.csv + _debug_*_mapping.csv  (held out; the pipeline never reads them)")
