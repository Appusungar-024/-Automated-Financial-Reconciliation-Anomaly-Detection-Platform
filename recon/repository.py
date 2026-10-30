"""
Repository: every database read and write the application performs.

Two properties matter more than any individual function:

* **Audit entries are written in the same transaction as the change they describe.**
  If a status update fails, its audit row never exists; if the audit insert fails, the
  update is rolled back. There is no state where the data changed but the record of
  who changed it is missing.
* **Workflow rules are enforced here, not in the UI.** The dashboard cannot be used to
  skip a status transition or close an exception without a reason, because the rules
  live below it (and the database CHECK constraints sit below those).
"""
from __future__ import annotations

import json
import logging
import uuid

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sqlalchemy import func, insert, select, text, update

from . import config, db
from .db import (audit_log, exceptions, matched_records, raw_transactions, reconciliation_runs, source_files,
                 utcnow, validation_failures)

log = logging.getLogger(__name__)


class WorkflowError(ValueError):
    """A review action that breaks the workflow rules (bad transition, missing reason, ...)."""


def _json(obj) -> str:
    return json.dumps(obj, default=str, sort_keys=True)


def _frame(conn, stmt, params=None, dates=()) -> pd.DataFrame:
    df = pd.read_sql(stmt, conn, params=params)
    for col in dates:
        if col in df:
            df[col] = pd.to_datetime(df[col])
    return df


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def log_event(conn, event_type: str, actor: str = "system", entity_type=None, entity_id=None,
              old_value=None, new_value=None, details=None, run_id=None) -> None:
    conn.execute(insert(audit_log).values(
        event_time=utcnow(), run_id=run_id, event_type=event_type, actor=actor or "system",
        entity_type=entity_type, entity_id=None if entity_id is None else str(entity_id),
        old_value=None if old_value is None else str(old_value),
        new_value=None if new_value is None else str(new_value),
        details=None if details is None else _json(details)))


def audit_trail(run_id=None, *, event_types=None, actors=None, entity_type=None, entity_id=None,
                search=None, limit=2000) -> pd.DataFrame:
    stmt = select(audit_log)
    if run_id:
        stmt = stmt.where(audit_log.c.run_id == run_id)
    if event_types:
        stmt = stmt.where(audit_log.c.event_type.in_(list(event_types)))
    if actors:
        stmt = stmt.where(audit_log.c.actor.in_(list(actors)))
    if entity_type:
        stmt = stmt.where(audit_log.c.entity_type == entity_type)
    if entity_id is not None:
        stmt = stmt.where(audit_log.c.entity_id == str(entity_id))
    if search:
        like = f"%{search.lower()}%"
        stmt = stmt.where(func.lower(func.coalesce(audit_log.c.details, "")).like(like) |
                          func.lower(func.coalesce(audit_log.c.entity_id, "")).like(like))
    stmt = stmt.order_by(audit_log.c.audit_id.desc())
    if limit:
        stmt = stmt.limit(limit)
    with db.get_engine().connect() as conn:
        return _frame(conn, stmt, dates=("event_time",))


def exception_history(exception_id: int) -> pd.DataFrame:
    with db.get_engine().connect() as conn:
        stmt = (select(audit_log).where(audit_log.c.entity_type == "exception",
                                        audit_log.c.entity_id == str(exception_id)).order_by(audit_log.c.audit_id))
        return _frame(conn, stmt, dates=("event_time",))


def reviewer_activity() -> pd.DataFrame:
    with db.get_engine().connect() as conn:
        return _frame(conn, text("SELECT * FROM v_reviewer_activity ORDER BY last_event DESC"))


# ---------------------------------------------------------------------------
# Run lifecycle and bulk persistence
# ---------------------------------------------------------------------------

def create_run(run_id: str, as_of, notes: str = "", actor: str = "system", snapshot: dict | None = None) -> None:
    with db.get_engine().begin() as conn:
        conn.execute(insert(reconciliation_runs).values(
            run_id=run_id, started_at=utcnow(), status="RUNNING", as_of_date=pd.Timestamp(as_of).date() if as_of is not None else None,
            notes=notes, config_snapshot=_json(snapshot if snapshot is not None else config.config_snapshot())))
        log_event(conn, "RUN_STARTED", actor, "run", run_id, details={"notes": notes}, run_id=run_id)


def fail_run(run_id: str, error: str, actor: str = "system", files: list[dict] | None = None,
             failures: pd.DataFrame | None = None) -> None:
    """Mark a run FAILED. When ingestion is what failed, the files and their validation failures are
    stored too, so the user can see exactly why a file was rejected."""
    now = utcnow()
    with db.get_engine().begin() as conn:
        for f in files or []:
            conn.execute(insert(source_files).values(run_id=run_id, ingested_at=now, **f))
        if failures is not None and not failures.empty:
            f_out = failures.copy()
            f_out["run_id"], f_out["created_at"] = run_id, now
            f_out["row_excluded"] = f_out["row_excluded"].astype(bool)
            _insert_df(conn, f_out, validation_failures)
        conn.execute(update(reconciliation_runs).where(reconciliation_runs.c.run_id == run_id)
                     .values(status="FAILED", finished_at=now, error_message=error[:2000]))
        log_event(conn, "RUN_FAILED", actor, "run", run_id, details={"error": error[:2000]}, run_id=run_id)


def _insert_df(conn, df: pd.DataFrame, table) -> int:
    """Bulk insert a frame into `table`, keeping only the columns the table defines."""
    if df.empty:
        return 0
    df = df[[c for c in table.c.keys() if c in df.columns]].copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date          # a Date column must receive dates, not timestamps
    chunk = max(1, min(2000, 30000 // max(1, df.shape[1])))      # stay under SQLite's bound-parameter limit
    df.to_sql(table.name, conn, if_exists="append", index=False, method="multi", chunksize=chunk)
    return len(df)


def save_run(run_id: str, *, files: list[dict], failures: pd.DataFrame, raw: pd.DataFrame, matches: pd.DataFrame,
             exceptions_df: pd.DataFrame, stage_stats: pd.DataFrame, anomaly_info: dict, metrics: dict,
             actor: str = "system") -> None:
    """Persist everything a run produced -- atomically. Either the whole run lands, or none of it does."""
    now = utcnow()
    with db.get_engine().begin() as conn:
        for f in files:
            conn.execute(insert(source_files).values(run_id=run_id, ingested_at=now, **f))
            log_event(conn, "FILE_INGESTED", actor, "file", f["file_name"], run_id=run_id, details={
                k: f.get(k) for k in ("source_system", "sha256", "sheet_name", "header_row", "rows_read",
                                      "rows_accepted", "rows_excluded")})

        if not failures.empty:
            f_out = failures.copy()
            f_out["run_id"], f_out["created_at"] = run_id, now
            f_out["row_excluded"] = f_out["row_excluded"].astype(bool)
            _insert_df(conn, f_out, validation_failures)
            by_code = failures.groupby(["severity", "code"]).size().rename("n").reset_index()
            log_event(conn, "VALIDATION_SUMMARY", actor, "run", run_id, run_id=run_id,
                      details={"total": int(len(failures)), "by_code": by_code.to_dict("records")})

        raw_out = raw.copy()
        raw_out["run_id"], raw_out["ingested_at"] = run_id, now
        _insert_df(conn, raw_out, raw_transactions)

        if not matches.empty:
            m_out = matches.copy()
            m_out["run_id"], m_out["matched_at"] = run_id, now
            _insert_df(conn, m_out, matched_records)
        if not exceptions_df.empty:
            e_out = exceptions_df.copy()
            e_out["run_id"], e_out["status"] = run_id, "OPEN"
            e_out["created_at"] = e_out["updated_at"] = now
            _insert_df(conn, e_out, exceptions)

        if not stage_stats.empty:
            for pair, grp in stage_stats.groupby("pair_name", sort=False):
                log_event(conn, "MATCH_STAGE_SUMMARY", actor, "run", run_id, run_id=run_id, details={
                    "pair": pair, "stages": grp[["stage", "matched_pairs", "remaining_a", "remaining_b"]].to_dict("records")})
        log_event(conn, "ANOMALY_MODEL_FIT", actor, "run", run_id, run_id=run_id, details=anomaly_info)

        conn.execute(update(reconciliation_runs).where(reconciliation_runs.c.run_id == run_id)
                     .values(status="COMPLETED", finished_at=now, **metrics))
        log_event(conn, "RUN_COMPLETED", actor, "run", run_id, run_id=run_id, details=metrics)


def previous_runs_with_hash(sha256: str, exclude_run: str | None = None) -> list[str]:
    stmt = select(source_files.c.run_id).where(source_files.c.sha256 == sha256).distinct()
    if exclude_run:
        stmt = stmt.where(source_files.c.run_id != exclude_run)
    with db.get_engine().connect() as conn:
        return [r[0] for r in conn.execute(stmt)]


# ---------------------------------------------------------------------------
# Reads: runs, KPIs and analytics views
# ---------------------------------------------------------------------------

def list_runs(limit: int = 100) -> pd.DataFrame:
    stmt = select(reconciliation_runs).order_by(reconciliation_runs.c.started_at.desc()).limit(limit)
    with db.get_engine().connect() as conn:
        return _frame(conn, stmt, dates=("started_at", "finished_at", "as_of_date"))


def get_run(run_id: str) -> dict | None:
    with db.get_engine().connect() as conn:
        row = conn.execute(select(reconciliation_runs).where(reconciliation_runs.c.run_id == run_id)).mappings().first()
        return dict(row) if row else None


def _view(name: str, run_id: str | None = None, order: str = "") -> pd.DataFrame:
    sql = f"SELECT * FROM {name}" + (" WHERE run_id = :r" if run_id else "") + (f" ORDER BY {order}" if order else "")
    with db.get_engine().connect() as conn:
        return _frame(conn, text(sql), params={"r": run_id} if run_id else None)


def band_summary(run_id: str) -> pd.DataFrame:
    """Unresolved exceptions by priority band: how many and how much."""
    sql = ("SELECT priority_band, COUNT(*) AS exceptions, COALESCE(SUM(amount), 0) AS amount FROM exceptions "
           "WHERE run_id = :r AND status IN ('OPEN', 'IN_REVIEW', 'ESCALATED') GROUP BY priority_band")
    with db.get_engine().connect() as conn:
        return _frame(conn, text(sql), params={"r": run_id})


def exception_summary(run_id): return _view("v_exception_summary", run_id)
def exception_aging(run_id): return _view("v_exception_aging", run_id, "bucket_order")
def source_reconciliation(run_id): return _view("v_source_reconciliation", run_id, "source_system")
def recon_by_pair(run_id): return _view("v_recon_by_pair", run_id, "pair_name, match_type")
def data_quality_summary(run_id): return _view("v_data_quality", run_id, "failures DESC")
def run_trend(): return _view("v_run_trend", None, "started_at")


def kpis(run_id: str) -> dict:
    """Headline numbers for one run. Reconciliation rates are *live* (they move as reviewers
    manually match items); the at-run-time figures are kept on the run record itself."""
    run = get_run(run_id)
    if run is None:
        return {}
    src = source_reconciliation(run_id)
    open_list = tuple(config.OPEN_STATUSES)
    with db.get_engine().connect() as conn:
        p = {"r": run_id}
        by_status = _frame(conn, text("SELECT status, COUNT(*) AS n, COALESCE(SUM(amount), 0) AS amount "
                                      "FROM exceptions WHERE run_id = :r GROUP BY status"), p)
        open_stats = conn.execute(text(
            "SELECT COUNT(*), COALESCE(SUM(amount), 0), AVG(age_days) FROM exceptions "
            "WHERE run_id = :r AND status IN ('OPEN', 'IN_REVIEW', 'ESCALATED')"), p).one()
        urgent = conn.execute(text(
            "SELECT COUNT(*) FROM exceptions WHERE run_id = :r AND status IN ('OPEN', 'IN_REVIEW', 'ESCALATED') "
            "AND priority_band IN ('CRITICAL', 'HIGH')"), p).scalar()
        flagged = conn.execute(text("SELECT COUNT(*) FROM raw_transactions WHERE run_id = :r AND is_suspicious = :t"),
                               {"r": run_id, "t": True}).scalar()
        quarantined = conn.execute(text("SELECT COUNT(*) FROM validation_failures WHERE run_id = :r "
                                        "AND row_id IS NOT NULL AND row_excluded = :t AND severity <> 'INFO'"),
                                   {"r": run_id, "t": True}).scalar()
        audit_events = conn.execute(text("SELECT COUNT(*) FROM audit_log WHERE run_id = :r"), p).scalar()
    total_rows, matched_rows = int(src["total_rows"].sum()), int(src["matched_rows"].sum())
    total_value, matched_value = float(src["total_value"].sum()), float(src["matched_value"].sum())
    status_n = dict(zip(by_status["status"], by_status["n"]))
    n_exc = int(by_status["n"].sum()) if not by_status.empty else 0
    closed = sum(int(status_n.get(s, 0)) for s in config.CLOSED_STATUSES)
    return {
        "run": run, "total_rows": total_rows, "matched_rows": matched_rows,
        "match_rate": matched_rows / total_rows * 100 if total_rows else 0.0,
        "total_value": total_value, "matched_value": matched_value,
        "value_match_rate": matched_value / total_value * 100 if total_value else 0.0,
        "exceptions_total": n_exc, "exceptions_open": int(open_stats[0]), "exceptions_closed": closed,
        "unresolved_amount": float(open_stats[1] or 0), "avg_age_open": float(open_stats[2] or 0),
        "urgent_open": int(urgent or 0), "resolution_rate": closed / n_exc * 100 if n_exc else 100.0,
        "by_status": status_n, "suspicious": int(flagged or 0), "quarantined_rows": int(quarantined or 0),
        "audit_events": int(audit_events or 0),
    }


def control_checks(run_id: str) -> list[dict]:
    """Reconciliation control totals. Each check states what must be true, what is true, and whether
    they agree -- the sort of tick-and-tie an auditor performs by hand. They stay valid after reviewers
    act (a manual match adds a match row for exceptions that still exist), so the identities account
    for manually matched items explicitly."""
    k = kpis(run_id)
    if not k:
        return []
    run = k["run"]
    p = {"r": run_id}
    with db.get_engine().connect() as conn:
        def scalar(sql, extra=None):
            return int(conn.execute(text(sql), {**p, **(extra or {})}).scalar() or 0)
        raw_n = scalar("SELECT COUNT(*) FROM raw_transactions WHERE run_id = :r")
        manual_exc = scalar("SELECT COUNT(*) FROM exceptions WHERE run_id = :r AND resolution_code = 'MATCHED_MANUALLY'")
        claims = scalar("SELECT COUNT(*) FROM (SELECT row_id_a AS row_id FROM matched_records WHERE run_id = :r "
                        "UNION ALL SELECT row_id_b FROM matched_records WHERE run_id = :r) c")
        distinct_claims = scalar("SELECT COUNT(*) FROM (SELECT row_id_a AS row_id FROM matched_records WHERE run_id = :r "
                                 "UNION SELECT row_id_b FROM matched_records WHERE run_id = :r) c")
        open_but_matched = scalar(
            "SELECT COUNT(*) FROM exceptions e WHERE e.run_id = :r AND e.status IN ('OPEN', 'IN_REVIEW', 'ESCALATED') "
            "AND (e.row_id IN (SELECT row_id_a FROM matched_records WHERE run_id = :r) "
            "OR e.row_id IN (SELECT row_id_b FROM matched_records WHERE run_id = :r))")
    checks = [
        ("Rows stored equal rows counted when the run finished", run["total_transactions"], raw_n),
        ("Unmatched rows equal exceptions raised (net of manual matches)",
         k["total_rows"] - k["matched_rows"], k["exceptions_total"] - manual_exc),
        ("No row is claimed by more than one match", claims, distinct_claims),
        ("No row is both matched and still an open exception", 0, open_but_matched),
        ("Open plus closed exceptions equal all exceptions", k["exceptions_total"],
         k["exceptions_open"] + k["exceptions_closed"]),
    ]
    return [{"check": name, "expected": int(exp), "actual": int(act), "passed": int(exp) == int(act)}
            for name, exp, act in checks]


# ---------------------------------------------------------------------------
# Reads: exceptions, matches, transactions
# ---------------------------------------------------------------------------

def _queue_conditions(run_id, statuses=None, types=None, sources=None, bands=None, assigned_to=None,
                      unassigned=False, min_amount=None, search=None):
    c = [exceptions.c.run_id == run_id]
    if statuses:
        c.append(exceptions.c.status.in_(list(statuses)))
    if types:
        c.append(exceptions.c.exception_type.in_(list(types)))
    if sources:
        c.append(exceptions.c.source_system.in_(list(sources)))
    if bands:
        c.append(exceptions.c.priority_band.in_(list(bands)))
    if assigned_to:
        c.append(exceptions.c.assigned_to == assigned_to)
    if unassigned:
        c.append(exceptions.c.assigned_to.is_(None))
    if min_amount:
        c.append(exceptions.c.amount >= float(min_amount))
    if search:
        like = f"%{search.lower()}%"
        c.append(func.lower(func.coalesce(exceptions.c.vendor_raw, "")).like(like) |
                 func.lower(func.coalesce(exceptions.c.reference, "")).like(like) |
                 func.lower(exceptions.c.row_id).like(like))
    return c


def review_queue(run_id: str, *, limit: int | None = 500, **filters) -> pd.DataFrame:
    stmt = (select(exceptions).where(*_queue_conditions(run_id, **filters))
            .order_by(exceptions.c.priority_score.desc(), exceptions.c.amount.desc(), exceptions.c.exception_id))
    if limit:
        stmt = stmt.limit(limit)
    with db.get_engine().connect() as conn:
        return _frame(conn, stmt, dates=("txn_date", "created_at", "updated_at", "resolved_at"))


def queue_totals(run_id: str, **filters) -> dict:
    stmt = select(func.count(), func.coalesce(func.sum(exceptions.c.amount), 0)).where(*_queue_conditions(run_id, **filters))
    with db.get_engine().connect() as conn:
        n, total = conn.execute(stmt).one()
    return {"count": int(n), "amount": float(total)}


def exception_ids(run_id: str, **filters) -> list[int]:
    stmt = select(exceptions.c.exception_id).where(*_queue_conditions(run_id, **filters)).order_by(exceptions.c.exception_id)
    with db.get_engine().connect() as conn:
        return [r[0] for r in conn.execute(stmt)]


def get_exception(exception_id: int) -> dict | None:
    with db.get_engine().connect() as conn:
        row = conn.execute(select(exceptions).where(exceptions.c.exception_id == exception_id)).mappings().first()
        return dict(row) if row else None


def get_transaction(run_id: str, row_id: str) -> dict | None:
    with db.get_engine().connect() as conn:
        row = conn.execute(select(raw_transactions).where(raw_transactions.c.run_id == run_id,
                                                          raw_transactions.c.row_id == row_id)).mappings().first()
        return dict(row) if row else None


def exception_for_row(run_id: str, row_id: str) -> dict | None:
    with db.get_engine().connect() as conn:
        row = conn.execute(select(exceptions).where(exceptions.c.run_id == run_id,
                                                    exceptions.c.row_id == row_id)).mappings().first()
        return dict(row) if row else None


def matches(run_id: str, *, pair_names=None, match_types=None, judgmental_only=False, limit: int | None = 1000) -> pd.DataFrame:
    stmt = select(matched_records).where(matched_records.c.run_id == run_id)
    if pair_names:
        stmt = stmt.where(matched_records.c.pair_name.in_(list(pair_names)))
    if judgmental_only:      # the decisions an auditor actually wants to look at
        match_types = ["AMOUNT_TOLERANCE", "FUZZY_VENDOR", "MANUAL"]
    if match_types:
        stmt = stmt.where(matched_records.c.match_type.in_(list(match_types)))
    stmt = stmt.order_by(matched_records.c.match_confidence.asc(), matched_records.c.match_id)
    if limit:
        stmt = stmt.limit(limit)
    with db.get_engine().connect() as conn:
        return _frame(conn, stmt, dates=("date_a", "date_b", "matched_at"))


def top_risky(run_id: str, *, min_risk: float = 0.0, flagged_only: bool = False, limit: int = 500) -> pd.DataFrame:
    stmt = select(raw_transactions).where(raw_transactions.c.run_id == run_id, raw_transactions.c.risk_score >= min_risk)
    if flagged_only:
        stmt = stmt.where(raw_transactions.c.is_suspicious.is_(True))
    stmt = stmt.order_by(raw_transactions.c.risk_score.desc(), raw_transactions.c.amount.desc()).limit(limit)
    with db.get_engine().connect() as conn:
        return _frame(conn, stmt, dates=("txn_date",))


def validation_failures_df(run_id: str, *, severity=None, source=None, code=None, limit: int | None = 2000) -> pd.DataFrame:
    stmt = select(validation_failures).where(validation_failures.c.run_id == run_id)
    if severity:
        stmt = stmt.where(validation_failures.c.severity.in_(list(severity)))
    if source:
        stmt = stmt.where(validation_failures.c.source_system.in_(list(source)))
    if code:
        stmt = stmt.where(validation_failures.c.code.in_(list(code)))
    stmt = stmt.order_by(validation_failures.c.failure_id)
    if limit:
        stmt = stmt.limit(limit)
    with db.get_engine().connect() as conn:
        return _frame(conn, stmt)


def source_files_df(run_id: str) -> pd.DataFrame:
    with db.get_engine().connect() as conn:
        return _frame(conn, select(source_files).where(source_files.c.run_id == run_id).order_by(source_files.c.file_id))


# ---------------------------------------------------------------------------
# Review workflow
# ---------------------------------------------------------------------------

def _load_exception(conn, exception_id: int, lock: bool = True) -> dict:
    stmt = select(exceptions).where(exceptions.c.exception_id == exception_id)
    if lock:
        stmt = stmt.with_for_update()           # PostgreSQL row lock; SQLite serialises writers anyway
    row = conn.execute(stmt).mappings().first()
    if row is None:
        raise WorkflowError(f"exception {exception_id} does not exist")
    return dict(row)


def _apply(conn, row: dict, actor: str, *, status=None, assigned_to=None, resolution_code=None, note=None,
           bulk_id: str | None = None) -> dict:
    """Validate, then write, one exception change and its audit events. Validation happens first and
    raises before any statement is issued, so a rejected action leaves no trace."""
    old_status = row["status"]
    note = (note or "").strip()
    changes: dict = {}
    events: list[tuple] = []
    new_status = status if status is not None else old_status

    if new_status not in config.ALL_STATUSES:
        raise WorkflowError(f"unknown status {new_status!r}")

    if new_status != old_status:
        if new_status not in config.STATUS_TRANSITIONS[old_status]:
            raise WorkflowError(f"an exception cannot move from {old_status} to {new_status}")
        if new_status in config.CLOSED_STATUSES:
            if resolution_code not in config.RESOLUTION_CODES:
                raise WorkflowError("closing an exception needs a valid resolution code")
            if (new_status == "DISMISSED" or resolution_code in config.CODES_REQUIRING_NOTE) and not note:
                raise WorkflowError(f"a note is required to {new_status.lower()} with code {resolution_code}")
            changes.update(resolution_code=resolution_code, resolved_by=actor, resolved_at=utcnow(),
                           resolution_notes=note or None)
        elif old_status in config.CLOSED_STATUSES:                 # re-opening
            if not note:
                raise WorkflowError("a note explaining why the exception is re-opened is required")
            changes.update(resolution_code=None, resolved_by=None, resolved_at=None)
        changes["status"] = new_status
        events.append(("EXCEPTION_STATUS_CHANGED", old_status, new_status,
                       {"resolution_code": resolution_code, "note": note or None}))
        # picking up an unassigned item assigns it to whoever took it
        if new_status == "IN_REVIEW" and assigned_to is None and not row["assigned_to"]:
            assigned_to = actor
    elif note:
        events.append(("EXCEPTION_NOTE_ADDED", None, None, {"note": note}))
        if old_status in config.OPEN_STATUSES:
            changes["resolution_notes"] = note

    if assigned_to is not None and (assigned_to or None) != row["assigned_to"]:
        changes["assigned_to"] = assigned_to or None
        events.append(("EXCEPTION_ASSIGNED", row["assigned_to"], assigned_to or None, {}))

    if not changes and not events:
        return row
    changes["updated_at"] = utcnow()
    conn.execute(update(exceptions).where(exceptions.c.exception_id == row["exception_id"]).values(**changes))
    for event_type, old, new, details in events:
        if bulk_id:
            details = {**details, "bulk_id": bulk_id}
        log_event(conn, event_type, actor, "exception", row["exception_id"], old, new, details, run_id=row["run_id"])
    return {**row, **changes}


def update_exception(exception_id: int, *, actor: str, status=None, assigned_to=None, resolution_code=None,
                     note=None) -> dict:
    """Change an exception's status / assignee / note. Raises WorkflowError if a rule is broken."""
    with db.get_engine().begin() as conn:
        row = _load_exception(conn, exception_id)
        return _apply(conn, row, actor, status=status, assigned_to=assigned_to,
                      resolution_code=resolution_code, note=note)


def bulk_update(exception_ids_: list[int], *, actor: str, status: str, resolution_code=None, note=None) -> dict:
    """Apply one action to many exceptions in a single transaction. Items that break a rule are
    skipped and reported rather than aborting the batch; every change is audited individually and
    tied together by a bulk_id."""
    bulk_id = uuid.uuid4().hex[:10]
    updated: list[int] = []
    skipped: list[tuple[int, str]] = []
    run_ids: set = set()
    with db.get_engine().begin() as conn:
        for start in range(0, len(exception_ids_), 500):
            chunk = exception_ids_[start:start + 500]
            rows = {r["exception_id"]: dict(r) for r in conn.execute(
                select(exceptions).where(exceptions.c.exception_id.in_(chunk)).with_for_update()).mappings()}
            for eid in chunk:
                if eid not in rows:
                    skipped.append((eid, "does not exist"))
                    continue
                try:
                    _apply(conn, rows[eid], actor, status=status, resolution_code=resolution_code, note=note,
                           bulk_id=bulk_id)
                    updated.append(eid)
                    run_ids.add(rows[eid]["run_id"])
                except WorkflowError as exc:
                    skipped.append((eid, str(exc)))
        log_event(conn, "BULK_ACTION", actor, "run", next(iter(run_ids), None), run_id=next(iter(run_ids), None),
                  details={"bulk_id": bulk_id, "status": status, "resolution_code": resolution_code, "note": note,
                           "updated": len(updated), "skipped": len(skipped)})
    return {"bulk_id": bulk_id, "updated": updated, "skipped": skipped}


def manual_match(exception_id_a: int, exception_id_b: int, *, actor: str, note: str = "") -> dict:
    """A reviewer pairs two open exceptions from different sources. Creates a MANUAL row in
    matched_records (so reconciliation rates update) and closes both exceptions, atomically."""
    if exception_id_a == exception_id_b:
        raise WorkflowError("an exception cannot be matched to itself")
    with db.get_engine().begin() as conn:
        a, b = _load_exception(conn, exception_id_a), _load_exception(conn, exception_id_b)
        if a["run_id"] != b["run_id"]:
            raise WorkflowError("exceptions from different runs cannot be matched")
        for e in (a, b):
            if e["status"] not in config.OPEN_STATUSES:
                raise WorkflowError(f"exception {e['exception_id']} is already {e['status']}")
        if a["source_system"] == b["source_system"]:
            raise WorkflowError("a match pairs two different sources")
        run_id = a["run_id"]
        pair = next((p for p in config.RECON_PAIRS if {p["source_a"], p["source_b"]} == {a["source_system"], b["source_system"]}), None)
        if pair and a["source_system"] != pair["source_a"]:
            a, b = b, a                                           # orient to the pair's A/B convention
        raw_a, raw_b = get_transaction_conn(conn, run_id, a["row_id"]), get_transaction_conn(conn, run_id, b["row_id"])
        va, vb = raw_a.get("vendor_clean") or "", raw_b.get("vendor_clean") or ""
        similarity = float(max(fuzz.token_sort_ratio(va, vb), fuzz.partial_ratio(va, vb))) if va and vb else None
        result = conn.execute(insert(matched_records).values(
            run_id=run_id, pair_name=pair["name"] if pair else "MANUAL", row_id_a=a["row_id"], row_id_b=b["row_id"],
            source_a=a["source_system"], source_b=b["source_system"], date_a=a["txn_date"], date_b=b["txn_date"],
            amount_a=a["amount"], amount_b=b["amount"], vendor_a=a["vendor_raw"], vendor_b=b["vendor_raw"],
            reference_a=a["reference"], reference_b=b["reference"], match_type="MANUAL", match_confidence=100.0,
            reference_mismatch=False, amount_diff=round(float(b["amount"]) - float(a["amount"]), 2),
            date_diff_days=(b["txn_date"] - a["txn_date"]).days, vendor_similarity=similarity,
            rule_applied="Matched manually by a reviewer", decided_by=actor, matched_at=utcnow()))
        match_id = result.inserted_primary_key[0]
        for me, other in ((a, b), (b, a)):
            conn.execute(update(exceptions).where(exceptions.c.exception_id == me["exception_id"]).values(
                status="RESOLVED", resolution_code="MATCHED_MANUALLY", resolved_by=actor, resolved_at=utcnow(),
                resolution_notes=(note or "").strip() or None, updated_at=utcnow()))
            log_event(conn, "MANUAL_MATCH", actor, "exception", me["exception_id"], me["status"], "RESOLVED",
                      {"match_id": match_id, "counterpart_exception_id": other["exception_id"],
                       "row_id": me["row_id"], "counterpart_row_id": other["row_id"],
                       "amount_diff": round(float(b["amount"]) - float(a["amount"]), 2), "note": (note or "").strip() or None},
                      run_id=run_id)
        return {"match_id": match_id, "exception_ids": [a["exception_id"], b["exception_id"]]}


def get_transaction_conn(conn, run_id: str, row_id: str) -> dict:
    row = conn.execute(select(raw_transactions).where(raw_transactions.c.run_id == run_id,
                                                      raw_transactions.c.row_id == row_id)).mappings().first()
    return dict(row) if row else {}
