"""
End-to-end orchestration:

    ingest + validate  ->  normalise  ->  match cascade (pair worklist)  ->  classify exceptions
        ->  anomaly scoring  ->  prioritise  ->  persist atomically with an audit trail

The one subtle design point is the pair *worklist*. BANK appears in two pairs
(Gateway<->Bank, Bank<->Ledger) and LEDGER in two (ERP<->Ledger, Bank<->Ledger).
Reconciling each pair against the full files would call a gateway-settled bank
deposit a "Bank vs Ledger miss" simply because it was never meant to be in the
ledger directly. Instead only what is still unexplained after one pair carries
into the next, so a row gets a fair shot in every pair it belongs to, and only
what is unmatched everywhere becomes an exception.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from . import anomaly, classification, config, db, ingestion, matching, prioritization, repository

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


@dataclass
class RunResult:
    run_id: str
    as_of: pd.Timestamp
    total: int = 0
    matched_rows: int = 0
    exception_count: int = 0
    match_rate: float = 0.0
    total_value: float = 0.0
    matched_value: float = 0.0
    runtime_s: float = 0.0
    persisted: bool = False
    pairs_run: list = field(default_factory=list)
    ingestion: dict = field(default_factory=dict)
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    matches: pd.DataFrame = field(default_factory=pd.DataFrame)
    exceptions: pd.DataFrame = field(default_factory=pd.DataFrame)
    stage_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    anomaly_info: dict = field(default_factory=dict)

    def headline(self) -> str:
        return (f"{self.total:,} transactions | {self.matched_rows:,} auto-matched ({self.match_rate:.1f}%) | "
                f"{self.exception_count:,} exceptions | {self.runtime_s:.1f}s")


def _file_record(res: ingestion.IngestionResult) -> dict:
    return {"source_system": res.source, "file_name": res.file_name, "sha256": res.sha256,
            "sheet_name": res.sheet_name, "header_row": res.header_row,
            "column_mapping": json.dumps(res.column_mapping, sort_keys=True), "rows_read": res.rows_read,
            "rows_accepted": res.rows_accepted, "rows_excluded": res.rows_excluded}


def run(files: dict | None = None, *, directory=None, persist: bool = True, notes: str = "", actor: str = "system",
        as_of=None, validation_as_of=None, run_id: str | None = None, on_progress=None) -> RunResult:
    """Reconcile the given source files. ``files`` maps source name -> path (CSV or Excel); if omitted,
    the sources found in ``directory`` (default data/raw) are used.

    as_of            reference date for ageing exceptions (default: latest transaction date in the data)
    validation_as_of reference date for the FUTURE_DATE check (default: today)
    """
    t0 = time.time()
    say = on_progress or (lambda msg: None)
    files = dict(files) if files else ingestion.discover_sources(directory or config.RAW_DIR)
    if not files:
        raise PipelineError(f"no source files found in {directory or config.RAW_DIR}")
    run_id = run_id or f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    validation_as_of = pd.Timestamp(validation_as_of) if validation_as_of is not None else pd.Timestamp(date.today())

    pairs = [p for p in config.RECON_PAIRS if p["source_a"] in files and p["source_b"] in files]
    if not pairs:
        raise PipelineError("need both sides of at least one reconciliation pair; got "
                            f"{sorted(files)} (pairs: " + ", ".join(f"{p['source_a']}+{p['source_b']}" for p in config.RECON_PAIRS) + ")")
    used = sorted({s for p in pairs for s in (p["source_a"], p["source_b"])}, key=config.SOURCES.index)
    unused = sorted(set(files) - set(used))

    results: dict = {}
    file_records: list | None = None
    failures: pd.DataFrame | None = None
    if persist:
        db.init_db()                                    # idempotent: creates tables and views if absent
        repository.create_run(run_id, as_of, notes, actor)
    try:
        # ---- 1. ingest + validate ------------------------------------------------------------
        say("Reading and validating source files")
        results = {s: ingestion.ingest_file(s, files[s], as_of=validation_as_of) for s in used}
        if persist:
            for s, res in results.items():
                prior = repository.previous_runs_with_hash(res.sha256, exclude_run=run_id) if res.sha256 else []
                if prior:
                    res.failures = pd.concat([res.failures, ingestion._file_failure(
                        s, res.file_name, "DUPLICATE_FILE",
                        f"identical file (SHA-256 {res.sha256[:12]}...) was already reconciled in run {prior[0]}")], ignore_index=True)
        failures = pd.concat([r.failures for r in results.values() if not r.failures.empty], ignore_index=True) \
            if any(not r.failures.empty for r in results.values()) else pd.DataFrame(columns=ingestion.FAILURE_COLUMNS)
        file_records = [_file_record(r) for r in results.values()]
        fatal = [r for r in results.values() if r.fatal]
        if fatal:
            def why(r):
                f = r.failures[r.failures["source_row"].isna() & (r.failures["severity"] == "ERROR")]
                return "; ".join(f"{c}: {m}" for c, m in zip(f["code"], f["message"]))
            raise PipelineError("source file(s) could not be used -- " +
                                " | ".join(f"{r.source} ({r.file_name}) {why(r)}" for r in fatal))

        frames = {s: r.df for s, r in results.items()}
        raw_all = pd.concat(frames.values(), ignore_index=True)
        if raw_all.empty:
            raise PipelineError("no valid transactions remained after validation")
        as_of_ts = pd.Timestamp(as_of) if as_of is not None else raw_all["txn_date"].max()

        # ---- 2. pair worklist ---------------------------------------------------------------------
        say("Running the matching cascade")
        remaining = dict(frames)
        matched_ids = {s: set() for s in frames}
        match_parts, stage_rows = [], []
        for pair in pairs:
            a_name, b_name = pair["source_a"], pair["source_b"]
            res = matching.reconcile(remaining[a_name], remaining[b_name], pair)
            for st in res.stage_stats:
                stage_rows.append({"pair_name": pair["name"], "stage": st.stage, "matched_pairs": st.matched_pairs,
                                   "remaining_a": st.remaining_a, "remaining_b": st.remaining_b})
            match_parts.append(res.matches)
            matched_ids[a_name] |= set(res.matches["row_id_a"])
            matched_ids[b_name] |= set(res.matches["row_id_b"])
            left_a = set(res.exceptions.loc[res.exceptions["source_role"] == "a", "row_id"])
            left_b = set(res.exceptions.loc[res.exceptions["source_role"] == "b", "row_id"])
            remaining[a_name] = remaining[a_name][remaining[a_name]["row_id"].isin(left_a)]
            remaining[b_name] = remaining[b_name][remaining[b_name]["row_id"].isin(left_b)]
        non_empty = [m for m in match_parts if not m.empty]
        matches = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame(columns=matching.MATCH_COLUMNS)

        # ---- 3. classify what is left ---------------------------------------------------------
        say("Classifying exceptions")
        classified = classification.classify_exceptions(frames, remaining, matched_ids, pairs)

        # ---- 4. anomaly scoring (every transaction, matched or not) ---------------------------
        say("Scoring transactions for anomalies")
        scored, anomaly_info = anomaly.detect(raw_all)

        # ---- 5. prioritise ------------------------------------------------------------------------
        exceptions = prioritization.score_exceptions(classified, scored.set_index("row_id")["risk_score"], as_of_ts)

        # ---- 6. control totals ----------------------------------------------------------------------
        matched_row_ids = set(matches["row_id_a"]) | set(matches["row_id_b"])
        total, matched_rows = len(scored), len(matched_row_ids)
        if matched_rows + len(exceptions) != total or set(exceptions["row_id"]) & matched_row_ids:
            raise PipelineError(f"control total failed: {matched_rows} matched + {len(exceptions)} exceptions != {total} rows")
        metrics = {
            "total_transactions": total, "matched_count": matched_rows, "exception_count": len(exceptions),
            "match_rate": round(matched_rows / total * 100, 2),
            "total_value": round(float(scored["amount"].sum()), 2),
            "matched_value": round(float(scored.loc[scored["row_id"].isin(matched_row_ids), "amount"].sum()), 2),
            "runtime_seconds": round(time.time() - t0, 2),
        }

        result = RunResult(run_id=run_id, as_of=as_of_ts, total=total, matched_rows=matched_rows,
                           exception_count=len(exceptions), match_rate=metrics["match_rate"],
                           total_value=metrics["total_value"], matched_value=metrics["matched_value"],
                           runtime_s=metrics["runtime_seconds"], pairs_run=[p["name"] for p in pairs], ingestion=results,
                           failures=failures, raw=scored, matches=matches, exceptions=exceptions,
                           stage_stats=pd.DataFrame(stage_rows), anomaly_info=anomaly_info)

        # ---- 7. persist atomically ------------------------------------------------------------
        if persist:
            say("Saving results and audit trail")
            if unused:
                log.warning("sources %s are not part of any complete pair and were skipped", unused)
            repository.save_run(run_id, files=file_records, failures=failures, raw=scored, matches=matches,
                                exceptions_df=exceptions, stage_stats=result.stage_stats,
                                anomaly_info=anomaly_info, metrics=metrics, actor=actor)
            result.persisted = True
        log.info("run %s: %s", run_id, result.headline())
        return result

    except Exception as exc:
        if persist:
            try:
                repository.fail_run(run_id, f"{type(exc).__name__}: {exc}", actor,
                                    files=file_records, failures=failures)
            except Exception:                                    # never mask the original error
                log.exception("could not record the failed run %s", run_id)
        raise
