"""
Central configuration.

Everything a reviewer might ask "why that number?" about lives here, in one
place, so thresholds are easy to defend and easy to change. A snapshot of the
tunables is stored with every reconciliation run (see ``config_snapshot``) so
any past result can be reproduced and explained.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = DATA_DIR / "reports"
for _d in (RAW_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database -- PostgreSQL if DATABASE_URL is set, otherwise a local SQLite file
# so the project runs with zero setup. Same schema and code path either way.
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'reconciliation.db').as_posix()}")

# ---------------------------------------------------------------------------
# Reconciliation pairs. Order matters: reference-verified pairs run first so
# their high-confidence matches claim rows before the looser Bank<->Ledger
# stages can grab them through a coincidental amount collision.
#
# use_reference_check: Bank UTRs and Ledger voucher numbers come from unrelated
# numbering schemes, so comparing them would manufacture noise. Gateway
# settlement UTRs and ERP linked-voucher numbers ARE carried into Bank/Ledger,
# so agreement (or disagreement) there is a real signal.
# ---------------------------------------------------------------------------
RECON_PAIRS = [
    {"name": "GATEWAY_VS_BANK", "source_a": "PAYMENT_GATEWAY", "source_b": "BANK", "share": 0.25, "use_reference_check": True},
    {"name": "ERP_VS_LEDGER", "source_a": "ERP", "source_b": "LEDGER", "share": 0.15, "use_reference_check": True},
    {"name": "BANK_VS_LEDGER", "source_a": "BANK", "source_b": "LEDGER", "share": 0.60, "use_reference_check": False},
]
SOURCES = ("BANK", "LEDGER", "PAYMENT_GATEWAY", "ERP")
SOURCE_FILE_STEMS = {"BANK": "bank", "LEDGER": "ledger", "PAYMENT_GATEWAY": "payment_gateway", "ERP": "erp"}
SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls")

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
# Tried per column; the format that parses the most values wins. d/m is listed
# before m/d, so a genuinely ambiguous column (every value <= 12) resolves to
# day-first and raises a warning rather than guessing silently.
DATE_FORMATS = (
    "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%y", "%d-%b-%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%Y/%m/%d", "%d/%m/%y", "%d %b %Y", "%d-%B-%Y", "ISO8601", "%m/%d/%Y",
)
MAX_FUTURE_DAYS = 1          # dates later than as_of + this are flagged FUTURE_DATE
HEADER_SCAN_ROWS = 40        # rows scanned for the header (statements often have a preamble)

# ---------------------------------------------------------------------------
# Matching cascade
# ---------------------------------------------------------------------------
DATE_TOLERANCE_DAYS = 3          # injected "moderate" date drift starts above this
AMOUNT_TOLERANCE_ABS = 5.0       # stage 3: absolute tolerance (bank charges/rounding)
AMOUNT_TOLERANCE_PCT = 0.005     # stage 3: or 0.5%, whichever is larger
FUZZY_DATE_WINDOW_DAYS = 5       # stage 4
FUZZY_AMOUNT_TOLERANCE_PCT = 0.01
FUZZY_VENDOR_THRESHOLD = 82      # rapidfuzz score 0-100

# Stage 5 (exception classification)
CLASSIFY_DATE_WINDOW_DAYS = 45
CLASSIFY_AMOUNT_BAND_PCT = 0.30
CLASSIFY_MIN_AMOUNT_BAND = 500.0
CLASSIFY_VENDOR_GATE = 70        # vendor similarity needed to be a mismatch candidate at all
CLASSIFY_MIN_COMPOSITE = 40
CLASSIFY_MAX_CANDIDATES = 150

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
CURRENCY_SYMBOL = "\u20b9"
ROUND_AMOUNT_DIVISOR = 10_000
ZSCORE_OUTLIER_THRESHOLD = 3.0
STRUCTURING_THRESHOLD = 50_000
STRUCTURING_WINDOW_DAYS = 3
NEW_VENDOR_MAX_ROWS = 2          # a vendor seen on <= this many rows counts as "new"
NEW_VENDOR_AMOUNT_QUANTILE = 0.995
NEW_VENDOR_MIN_DATASET_ROWS = 200   # a percentile cut-off is meaningless on a tiny upload

# Points per rule (capped at 100). Judgement calls, documented so they can be argued with.
RULE_POINTS = {
    "round_amount": 15,
    "weekend": 15,
    "vendor_outlier": 35,
    "reference_reuse": 60,       # same reference, same source, different amounts: tampering pattern
    "structuring": 60,
    "new_vendor_large": 55,       # rare, high-severity: a stand-alone review trigger (ghost-vendor pattern)
}

# Isolation Forest (unsupervised: it never sees labels)
ML_N_ESTIMATORS = 200
ML_CONTAMINATION = "auto"
ML_RANDOM_STATE = 42
ML_TAIL_START = 0.97             # only the top 3% of anomaly scores earn a non-zero model score

# Final risk = noisy-OR of rule score and model score; flag at/above this
RISK_FLAG_THRESHOLD = 50

# ---------------------------------------------------------------------------
# Review prioritisation. Exceptions get a 0-100 priority built from four parts.
# ---------------------------------------------------------------------------
EXCEPTION_SEVERITY = {           # policy judgement: how bad is this kind of break?
    "AMOUNT_MISMATCH": 80,       # direct misstatement
    "DUPLICATE": 75,             # possible duplicate payment / cash leakage
    "MISSING": 70,               # unrecorded transaction
    "PROBABLE_MATCH": 45,        # likely the same transaction, needs confirmation
    "DATE_MISMATCH": 40,         # usually a timing / cut-off difference
}
PRIORITY_WEIGHTS = {"amount": 0.40, "severity": 0.25, "risk": 0.20, "age": 0.15}
PRIORITY_BANDS = (("CRITICAL", 70), ("HIGH", 55), ("MEDIUM", 40), ("LOW", 0))
AGING_CAP_DAYS = 90
AGING_BUCKETS = ((0, 7, "0-7 days"), (8, 14, "8-14 days"), (15, 30, "15-30 days"),
                 (31, 60, "31-60 days"), (61, 10**6, "60+ days"))

# ---------------------------------------------------------------------------
# Review workflow
# ---------------------------------------------------------------------------
OPEN_STATUSES = ("OPEN", "IN_REVIEW", "ESCALATED")
CLOSED_STATUSES = ("RESOLVED", "DISMISSED")
ALL_STATUSES = OPEN_STATUSES + CLOSED_STATUSES
STATUS_TRANSITIONS = {
    "OPEN": {"IN_REVIEW", "ESCALATED", "RESOLVED", "DISMISSED"},
    "IN_REVIEW": {"OPEN", "ESCALATED", "RESOLVED", "DISMISSED"},
    "ESCALATED": {"IN_REVIEW", "RESOLVED", "DISMISSED"},
    "RESOLVED": {"OPEN"},        # re-open
    "DISMISSED": {"OPEN"},
}
RESOLUTION_CODES = {
    "MATCHED_MANUALLY": "Matched manually to its counterpart",
    "TIMING_DIFFERENCE": "Timing / cut-off difference, clears next period",
    "BANK_ERROR": "Error at the bank or gateway, correction requested",
    "LEDGER_CORRECTION": "Ledger / ERP entry corrected or booked",
    "DUPLICATE_CONFIRMED": "Duplicate confirmed, reversal raised",
    "FALSE_POSITIVE": "Not an exception on review",
    "WRITTEN_OFF": "Immaterial, written off per policy",
    "OTHER": "Other (see notes)",
}
CODES_REQUIRING_NOTE = ("OTHER", "WRITTEN_OFF", "FALSE_POSITIVE")

# ---------------------------------------------------------------------------
# Simulation (see recon/simulate.py). Rates sum to 1.0 per reconciliation pair.
# ---------------------------------------------------------------------------
DISCREPANCY_RATES = {
    "MATCH": 0.775,
    "MISSING": 0.06,
    "DUPLICATE": 0.04,
    "AMOUNT_MISMATCH": 0.04,
    "DATE_MISMATCH": 0.04,
    "REFERENCE_MISMATCH": 0.03,
    "SUSPICIOUS_ROUND_SPIKE": 0.015,
}
EXTRA_ANOMALY_RATE = 0.0015       # extra structuring / reference-reuse / new-vendor items, per pattern
JUNK_ROW_RATE = 0.004             # malformed / summary / blank rows injected per source file
WEEKDAY_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 0.15, 0.05)   # Mon..Sun: B2B payments are weekday-heavy
RANDOM_SEED = 42


def config_snapshot() -> dict:
    """Tunables recorded with every run so results are reproducible and explainable."""
    keys = [
        "DATE_TOLERANCE_DAYS", "AMOUNT_TOLERANCE_ABS", "AMOUNT_TOLERANCE_PCT", "FUZZY_DATE_WINDOW_DAYS",
        "FUZZY_AMOUNT_TOLERANCE_PCT", "FUZZY_VENDOR_THRESHOLD", "CLASSIFY_VENDOR_GATE", "CLASSIFY_MIN_COMPOSITE",
        "ROUND_AMOUNT_DIVISOR", "ZSCORE_OUTLIER_THRESHOLD", "STRUCTURING_THRESHOLD", "RULE_POINTS",
        "ML_N_ESTIMATORS", "ML_CONTAMINATION", "ML_RANDOM_STATE", "ML_TAIL_START", "RISK_FLAG_THRESHOLD",
        "PRIORITY_WEIGHTS", "EXCEPTION_SEVERITY", "AGING_CAP_DAYS",
    ]
    g = globals()
    snap = {k: g[k] for k in keys}
    snap["RECON_PAIRS"] = [{k: p[k] for k in ("name", "source_a", "source_b", "use_reference_check")} for p in RECON_PAIRS]
    return snap
