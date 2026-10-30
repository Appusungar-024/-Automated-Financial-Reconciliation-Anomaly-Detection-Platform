"""
Normalisation layer.

Ingestion (``recon.ingestion``) turns any source file into clean, validated
rows. This module then makes those rows *comparable* across sources:

    row_id | source_system | source_row | txn_date | vendor_raw | vendor_clean
    reference | amount | direction | dup_hash | is_duplicate

``row_id`` is derived from the row's position in the original file, so every
match, exception and audit entry can be traced back to an exact line of the
source document.
"""
import re

import numpy as np
import pandas as pd

COMMON_SUFFIXES_RE = re.compile(
    r"\b(PVT\.?\s*LTD\.?|PRIVATE\s+LIMITED|LIMITED|LTD\.?|LLP|INC\.?|LLC|"
    r"CORP(?:ORATION)?|PLC|CO\.?|GROUP|AND\s+SONS|&\s*SONS)\b"
)
BANK_TXN_CODE_RE = re.compile(r"^(NEFT|IMPS|RTGS|UPI|CHQ)-")
# Bank narrations end in a *numeric* reference ("-294419"). Restricting this to
# digits matters: an earlier version matched any alphanumeric tail and so also
# stripped the second half of hyphenated surnames ("Guha-Iyengar" -> "GUHA"),
# silently desynchronising vendor names between Bank and Ledger.
TRAILING_REF_RE = re.compile(r"-\d{4,}$")
NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")

INTERNAL_COLUMNS = ["row_id", "source_system", "source_row", "txn_date", "vendor_raw", "vendor_clean",
                    "reference", "amount", "direction", "dup_hash", "is_duplicate"]


def normalize_vendor(raw) -> str:
    """'NEFT-TCS-563349', 'T.C.S.' and 'TCS Ltd' collapse toward one comparable
    token. Order matters: strip prefixes/suffixes as whole words while spaces
    still separate them, then compress whitespace and punctuation last."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = str(raw).upper().strip()
    s = BANK_TXN_CODE_RE.sub("", s)
    s = TRAILING_REF_RE.sub("", s)
    s = COMMON_SUFFIXES_RE.sub(" ", s)
    return NON_ALNUM_RE.sub("", s)


def row_fingerprint(df: pd.DataFrame) -> pd.Series:
    """date | amount | vendor | reference -- identical fingerprints within one
    source are accidental double postings."""
    return (df["txn_date"].dt.strftime("%Y-%m-%d") + "|" + df["amount"].round(2).astype(str) + "|" +
            df["vendor_clean"] + "|" + df["reference"].fillna("").astype(str))


def flag_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """First occurrence is the legitimate one; repeats are flagged. Flagged rows
    still take part in matching, so a duplicate can be explained rather than
    mistaken for a missing transaction."""
    df = df.copy()
    df["is_duplicate"] = df.duplicated(subset="dup_hash", keep="first")
    return df


def finalize_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Ingested rows (source_row, txn_date, vendor_raw, reference, amount,
    direction) -> the common internal schema."""
    df = df.copy().reset_index(drop=True)
    df["source_system"] = source
    df["txn_date"] = pd.to_datetime(df["txn_date"]).dt.normalize().astype("datetime64[ns]")
    df["reference"] = df["reference"].fillna("").astype(str)
    df["vendor_raw"] = df["vendor_raw"].fillna("").astype(str)
    df["amount"] = df["amount"].astype(float).round(2)

    # vendor names repeat heavily; normalise each distinct name once
    uniques = df["vendor_raw"].unique()
    lookup = {u: normalize_vendor(u) for u in uniques}
    df["vendor_clean"] = df["vendor_raw"].map(lookup).astype(str)   # astype: an empty map() returns float64

    df["row_id"] = [f"{source}-{int(i):07d}" for i in df["source_row"]]
    df["dup_hash"] = row_fingerprint(df)
    df = flag_duplicates(df)
    extra = [c for c in df.columns if c not in INTERNAL_COLUMNS]
    return df[INTERNAL_COLUMNS + extra]
