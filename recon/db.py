"""
Database layer: engine, schema, integrity rules and analytics views.

Works against PostgreSQL (set DATABASE_URL) or a local SQLite file (the default,
zero-setup fallback) through the same SQLAlchemy Core definitions. ``db/schema.sql``
carries the same schema as hand-written PostgreSQL DDL for reviewers who want to
read the SQL rather than the Python.

Integrity decisions worth knowing
---------------------------------
* ``raw_transactions`` is keyed by **(run_id, row_id)**. v1 keyed it by row_id alone,
  so reconciling a second file in the same database collided on the first.
* Matches and exceptions reference raw rows with composite foreign keys, so nothing
  can point at a transaction that was never loaded (SQLite enforcement is switched on
  explicitly -- it is off by default).
* Exception status has a CHECK constraint, so a bad value cannot be written even by
  code that bypasses the workflow functions.
* ``audit_log`` is **append-only, enforced by database triggers**: UPDATE and DELETE
  (and TRUNCATE on PostgreSQL) raise an error. An audit trail that application code
  could quietly rewrite is not an audit trail.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import (Boolean, CheckConstraint, Column, Date, DateTime, DDL, ForeignKey, ForeignKeyConstraint,
                        Index, Integer, MetaData, Numeric, String, Table, Text, UniqueConstraint, create_engine, event,
                        text)

from . import config

log = logging.getLogger(__name__)

metadata = MetaData()
_engine = None

MONEY = Numeric(18, 2, asdecimal=False)      # floats on read (no Decimal on SQLite, no warnings)
SCORE = Numeric(6, 1, asdecimal=False)
RATE = Numeric(7, 2, asdecimal=False)


def utcnow() -> datetime:
    """Naive UTC timestamp (what every DateTime column here stores)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
reconciliation_runs = Table(
    "reconciliation_runs", metadata,
    Column("run_id", String(40), primary_key=True),
    Column("started_at", DateTime, nullable=False),
    Column("finished_at", DateTime),
    Column("status", String(12), nullable=False, default="RUNNING"),
    Column("as_of_date", Date),
    Column("total_transactions", Integer),
    Column("matched_count", Integer),
    Column("exception_count", Integer),
    Column("match_rate", RATE),
    Column("total_value", MONEY),
    Column("matched_value", MONEY),
    Column("runtime_seconds", RATE),
    Column("notes", Text),
    Column("config_snapshot", Text),
    Column("error_message", Text),
    CheckConstraint("status IN ('RUNNING', 'COMPLETED', 'FAILED')", name="ck_runs_status"),
)

source_files = Table(
    "source_files", metadata,
    Column("file_id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), ForeignKey("reconciliation_runs.run_id"), nullable=False),
    Column("source_system", String(32), nullable=False),
    Column("file_name", String(255), nullable=False),
    Column("sha256", String(64)),
    Column("sheet_name", String(255)),
    Column("header_row", Integer),
    Column("column_mapping", Text),
    Column("rows_read", Integer),
    Column("rows_accepted", Integer),
    Column("rows_excluded", Integer),
    Column("ingested_at", DateTime),
    Index("ix_source_files_run", "run_id"),
    Index("ix_source_files_sha", "sha256"),
)

validation_failures = Table(
    "validation_failures", metadata,
    Column("failure_id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), ForeignKey("reconciliation_runs.run_id"), nullable=False),
    Column("source_system", String(32)),
    Column("file_name", String(255)),
    Column("source_row", Integer),
    Column("row_id", String(40)),
    Column("field", String(64)),
    Column("code", String(40), nullable=False),
    Column("severity", String(10), nullable=False),
    Column("row_excluded", Boolean, nullable=False),
    Column("raw_value", Text),
    Column("message", Text),
    Column("created_at", DateTime),
    CheckConstraint("severity IN ('ERROR', 'WARNING', 'INFO')", name="ck_failures_severity"),
    Index("ix_failures_run", "run_id", "code"),
)

raw_transactions = Table(
    "raw_transactions", metadata,
    Column("run_id", String(40), ForeignKey("reconciliation_runs.run_id"), primary_key=True),
    Column("row_id", String(40), primary_key=True),
    Column("source_system", String(32), nullable=False),
    Column("source_row", Integer),
    Column("txn_date", Date, nullable=False),
    Column("vendor_raw", Text),
    Column("vendor_clean", String(255)),
    Column("reference", String(128)),
    Column("amount", MONEY, nullable=False),
    Column("direction", String(8)),
    Column("is_duplicate", Boolean, default=False),
    Column("rule_score", SCORE),
    Column("ml_score", SCORE),
    Column("risk_score", SCORE),
    Column("is_suspicious", Boolean, default=False),
    Column("suspicion_reasons", Text),
    Column("ingested_at", DateTime),
    Index("ix_raw_run_source", "run_id", "source_system"),
    Index("ix_raw_run_risk", "run_id", "risk_score"),
)

matched_records = Table(
    "matched_records", metadata,
    Column("match_id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False),
    Column("pair_name", String(64)),
    Column("row_id_a", String(40), nullable=False),
    Column("row_id_b", String(40), nullable=False),
    Column("source_a", String(32)),
    Column("source_b", String(32)),
    Column("date_a", Date), Column("date_b", Date),
    Column("amount_a", MONEY), Column("amount_b", MONEY),
    Column("vendor_a", Text), Column("vendor_b", Text),
    Column("reference_a", String(128)), Column("reference_b", String(128)),
    Column("match_type", String(32), nullable=False),
    Column("match_confidence", SCORE),
    Column("reference_mismatch", Boolean, default=False),
    # decision rationale: why was this pair matched?
    Column("amount_diff", MONEY),
    Column("date_diff_days", Integer),
    Column("vendor_similarity", SCORE),
    Column("rule_applied", Text),
    Column("decided_by", String(128), default="system"),
    Column("matched_at", DateTime),
    ForeignKeyConstraint(["run_id", "row_id_a"], ["raw_transactions.run_id", "raw_transactions.row_id"]),
    ForeignKeyConstraint(["run_id", "row_id_b"], ["raw_transactions.run_id", "raw_transactions.row_id"]),
    Index("ix_matches_run_a", "run_id", "row_id_a"),
    Index("ix_matches_run_b", "run_id", "row_id_b"),
    Index("ix_matches_run_pair", "run_id", "pair_name", "match_type"),
)

exceptions = Table(
    "exceptions", metadata,
    Column("exception_id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False),
    Column("pair_name", String(128)),
    Column("row_id", String(40), nullable=False),
    Column("source_system", String(32)),
    Column("txn_date", Date),
    Column("vendor_raw", Text),
    Column("reference", String(128)),
    Column("amount", MONEY),
    Column("direction", String(8)),
    Column("exception_type", String(32), nullable=False),
    Column("suggested_match_row_id", String(40)),
    Column("suggested_match_confidence", SCORE),
    Column("suggested_match_reason", Text),
    Column("risk_score", SCORE),
    Column("age_days", Integer),
    Column("priority_score", SCORE),
    Column("priority_band", String(10)),
    # review workflow
    Column("status", String(12), nullable=False, default="OPEN"),
    Column("assigned_to", String(128)),
    Column("resolution_code", String(32)),
    Column("resolved_by", String(128)),
    Column("resolved_at", DateTime),
    Column("resolution_notes", Text),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
    ForeignKeyConstraint(["run_id", "row_id"], ["raw_transactions.run_id", "raw_transactions.row_id"]),
    UniqueConstraint("run_id", "row_id", name="uq_exceptions_run_row"),
    CheckConstraint("status IN ('OPEN', 'IN_REVIEW', 'ESCALATED', 'RESOLVED', 'DISMISSED')", name="ck_exceptions_status"),
    Index("ix_exc_run_status", "run_id", "status"),
    Index("ix_exc_run_priority", "run_id", "priority_score"),
)

audit_log = Table(
    "audit_log", metadata,
    Column("audit_id", Integer, primary_key=True, autoincrement=True),
    Column("event_time", DateTime, nullable=False),
    Column("run_id", String(40)),
    Column("event_type", String(40), nullable=False),
    Column("actor", String(128), nullable=False),
    Column("entity_type", String(32)),
    Column("entity_id", String(64)),
    Column("old_value", Text),
    Column("new_value", Text),
    Column("details", Text),
    Index("ix_audit_run", "run_id"),
    Index("ix_audit_entity", "entity_type", "entity_id"),
    Index("ix_audit_time", "event_time"),
)

# ---------------------------------------------------------------------------
# Append-only enforcement for audit_log
# ---------------------------------------------------------------------------
_PG_FUNCTION = DDL("""
CREATE OR REPLACE FUNCTION audit_log_block_change() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only: UPDATE, DELETE and TRUNCATE are not permitted';
END;
$$ LANGUAGE plpgsql
""")
_PG_ROW_TRIGGER = DDL("""
CREATE TRIGGER audit_log_no_update_delete BEFORE UPDATE OR DELETE ON audit_log
FOR EACH ROW EXECUTE FUNCTION audit_log_block_change()
""")
_PG_TRUNCATE_TRIGGER = DDL("""
CREATE TRIGGER audit_log_no_truncate BEFORE TRUNCATE ON audit_log
FOR EACH STATEMENT EXECUTE FUNCTION audit_log_block_change()
""")
_SQLITE_NO_UPDATE = DDL("""
CREATE TRIGGER audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE is not permitted'); END
""")
_SQLITE_NO_DELETE = DDL("""
CREATE TRIGGER audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only: DELETE is not permitted'); END
""")
for _ddl in (_PG_FUNCTION, _PG_ROW_TRIGGER, _PG_TRUNCATE_TRIGGER):
    event.listen(audit_log, "after_create", _ddl.execute_if(dialect="postgresql"))
for _ddl in (_SQLITE_NO_UPDATE, _SQLITE_NO_DELETE):
    event.listen(audit_log, "after_create", _ddl.execute_if(dialect="sqlite"))


# ---------------------------------------------------------------------------
# Engine management
# ---------------------------------------------------------------------------

def configure(url: str):
    """(Re)point the module at a database. Tests use this to get an isolated SQLite file."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    _engine = create_engine(url, future=True, **kwargs)
    if _engine.dialect.name == "sqlite":
        @event.listens_for(_engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):      # SQLite ignores FKs unless asked
            dbapi_connection.execute("PRAGMA foreign_keys = ON")
    return _engine


def get_engine():
    return _engine if _engine is not None else configure(config.DATABASE_URL)


def dialect() -> str:
    return get_engine().dialect.name


def backend_label() -> str:
    return "PostgreSQL" if dialect() == "postgresql" else ("SQLite (local file)" if dialect() == "sqlite" else dialect())


# ---------------------------------------------------------------------------
# Schema + views
# ---------------------------------------------------------------------------
ANALYTICS_SQL = config.BASE_DIR / "db" / "analytics.sql"


def _view_statements(only_drops: bool = False) -> list[str]:
    raw = ANALYTICS_SQL.read_text(encoding="utf-8")
    sql = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("--"))
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    return [s for s in statements if s.upper().startswith("DROP VIEW")] if only_drops else statements


def init_db() -> None:
    """Create tables (idempotent) and (re)create the analytics views."""
    engine = get_engine()
    metadata.create_all(engine)
    with engine.begin() as conn:
        for stmt in _view_statements():
            conn.execute(text(stmt))


_ready_for = None


def ensure_ready() -> None:
    """init_db() once per engine. Streamlit re-executes the whole script on every click; re-creating
    eight views each time would be wasteful, and on PostgreSQL DROP VIEW takes locks."""
    global _ready_for
    engine = get_engine()
    if _ready_for is not engine:
        init_db()
        _ready_for = engine


def drop_all() -> None:
    """Development / test helper: remove every view and table."""
    engine = get_engine()
    with engine.begin() as conn:
        for stmt in reversed(_view_statements(only_drops=True)):
            conn.execute(text(stmt))
    metadata.drop_all(engine)
    if dialect() == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("DROP FUNCTION IF EXISTS audit_log_block_change() CASCADE"))
