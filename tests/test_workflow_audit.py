import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from recon import config, db, pipeline, repository
from recon.repository import WorkflowError
from tests.conftest import BANK_HEADER, VALIDATION_AS_OF, write_rows


def exc_by_row(run_id, row_id):
    return repository.exception_for_row(run_id, row_id)


def audit_types(run_id=None, **kw):
    return list(repository.audit_trail(run_id, **kw)["event_type"])


# ----------------------------------------------------------------- the run itself

def test_run_is_persisted_with_control_totals_and_provenance(mini_run):
    r = mini_run
    assert (r.total, r.matched_rows, r.exception_count) == (8, 4, 4) and r.persisted
    run = repository.get_run(r.run_id)
    assert run["status"] == "COMPLETED" and run["matched_count"] == 4 and run["match_rate"] == 50.0
    assert '"RECON_PAIRS"' in run["config_snapshot"]                      # parameters recorded for reproducibility
    files = repository.source_files_df(r.run_id)
    assert sorted(files["source_system"]) == ["BANK", "LEDGER"]
    assert files["sha256"].str.len().eq(64).all() and (files["rows_accepted"] == 4).all()


def test_the_known_story_is_classified_correctly(mini_run):
    q = repository.review_queue(mini_run.run_id).set_index("row_id")
    assert q.loc["BANK-0000001", "exception_type"] == "AMOUNT_MISMATCH"
    assert q.loc["LEDGER-0000001", "exception_type"] == "AMOUNT_MISMATCH"
    assert q.loc["BANK-0000002", "exception_type"] == "MISSING"
    assert q.loc["LEDGER-0000003", "exception_type"] == "MISSING"
    assert q.loc["BANK-0000001", "suggested_match_row_id"] == "LEDGER-0000001"


def test_run_lifecycle_is_audited(mini_run):
    kinds = audit_types(mini_run.run_id)
    for expected in ("RUN_STARTED", "FILE_INGESTED", "MATCH_STAGE_SUMMARY", "ANOMALY_MODEL_FIT", "RUN_COMPLETED"):
        assert expected in kinds
    assert kinds.count("FILE_INGESTED") == 2


def test_two_runs_can_share_one_database(tmp_db, mini_files):
    """v1 keyed raw rows by row_id alone, so a second run collided with the first."""
    first = pipeline.run(mini_files, validation_as_of=VALIDATION_AS_OF)
    second = pipeline.run(mini_files, validation_as_of=VALIDATION_AS_OF)
    assert len(repository.list_runs()) == 2 and first.run_id != second.run_id
    assert repository.kpis(second.run_id)["total_rows"] == 8
    warn = repository.validation_failures_df(second.run_id, code=["DUPLICATE_FILE"])
    assert len(warn) == 2 and first.run_id in warn.iloc[0]["message"]      # same files seen before


def test_unusable_file_fails_the_run_but_keeps_the_evidence(tmp_db, mini_files, tmp_path):
    bad = write_rows(tmp_path / "ledger_bad.csv", [["foo", "bar"], ["1", "2"]])
    with pytest.raises(pipeline.PipelineError, match="could not be used"):
        pipeline.run({"BANK": mini_files["BANK"], "LEDGER": bad}, validation_as_of=VALIDATION_AS_OF)
    run = repository.list_runs().iloc[0]
    assert run["status"] == "FAILED" and "HEADER_NOT_FOUND" in run["error_message"]
    assert "HEADER_NOT_FOUND" in set(repository.validation_failures_df(run["run_id"])["code"])
    assert "RUN_FAILED" in audit_types(run["run_id"])


def test_running_with_no_complete_pair_is_refused(tmp_db, mini_files):
    with pytest.raises(pipeline.PipelineError, match="both sides"):
        pipeline.run({"BANK": mini_files["BANK"]}, validation_as_of=VALIDATION_AS_OF)


# ----------------------------------------------------------------- status workflow

def test_closing_an_exception_needs_a_resolution_code(mini_run):
    e = exc_by_row(mini_run.run_id, "BANK-0000002")
    with pytest.raises(WorkflowError, match="resolution code"):
        repository.update_exception(e["exception_id"], actor="asha", status="RESOLVED")
    done = repository.update_exception(e["exception_id"], actor="asha", status="RESOLVED",
                                       resolution_code="TIMING_DIFFERENCE", note="clears next period")
    assert done["status"] == "RESOLVED" and done["resolved_by"] == "asha" and done["resolved_at"] is not None


def test_transitions_follow_the_state_machine(mini_run):
    e = exc_by_row(mini_run.run_id, "BANK-0000002")["exception_id"]
    repository.update_exception(e, actor="asha", status="ESCALATED")
    with pytest.raises(WorkflowError, match="cannot move from ESCALATED to OPEN"):
        repository.update_exception(e, actor="asha", status="OPEN")
    repository.update_exception(e, actor="asha", status="RESOLVED", resolution_code="BANK_ERROR")
    with pytest.raises(WorkflowError, match="cannot move from RESOLVED to IN_REVIEW"):
        repository.update_exception(e, actor="asha", status="IN_REVIEW")


@pytest.mark.parametrize("status,code", [("DISMISSED", "FALSE_POSITIVE"), ("RESOLVED", "OTHER"), ("RESOLVED", "WRITTEN_OFF")])
def test_some_resolutions_demand_a_written_reason(mini_run, status, code):
    e = exc_by_row(mini_run.run_id, "BANK-0000002")["exception_id"]
    with pytest.raises(WorkflowError, match="note is required"):
        repository.update_exception(e, actor="asha", status=status, resolution_code=code)
    assert repository.update_exception(e, actor="asha", status=status, resolution_code=code, note="reviewed")["status"] == status


def test_unknown_resolution_code_and_status_are_rejected(mini_run):
    e = exc_by_row(mini_run.run_id, "BANK-0000002")["exception_id"]
    with pytest.raises(WorkflowError):
        repository.update_exception(e, actor="asha", status="RESOLVED", resolution_code="BECAUSE_I_SAID_SO")
    with pytest.raises(WorkflowError, match="unknown status"):
        repository.update_exception(e, actor="asha", status="DONE")


def test_reopening_requires_a_reason_and_clears_the_resolution(mini_run):
    e = exc_by_row(mini_run.run_id, "BANK-0000002")["exception_id"]
    repository.update_exception(e, actor="asha", status="RESOLVED", resolution_code="BANK_ERROR")
    with pytest.raises(WorkflowError, match="re-opened"):
        repository.update_exception(e, actor="ravi", status="OPEN")
    reopened = repository.update_exception(e, actor="ravi", status="OPEN", note="bank reversed the correction")
    assert reopened["status"] == "OPEN" and reopened["resolution_code"] is None and reopened["resolved_by"] is None


def test_taking_an_item_assigns_it_to_the_taker(mini_run):
    e = exc_by_row(mini_run.run_id, "BANK-0000002")["exception_id"]
    repository.update_exception(e, actor="asha", status="IN_REVIEW")
    assert repository.get_exception(e)["assigned_to"] == "asha"
    repository.update_exception(e, actor="asha", assigned_to="ravi")
    assert repository.get_exception(e)["assigned_to"] == "ravi"


def test_every_workflow_action_is_attributed_in_the_audit_trail(mini_run):
    e = exc_by_row(mini_run.run_id, "BANK-0000002")["exception_id"]
    repository.update_exception(e, actor="asha", status="IN_REVIEW")
    repository.update_exception(e, actor="asha", note="called the bank")
    repository.update_exception(e, actor="asha", status="RESOLVED", resolution_code="LEDGER_CORRECTION", note="booked V9")
    h = repository.exception_history(e)
    assert list(h["event_type"]) == ["EXCEPTION_STATUS_CHANGED", "EXCEPTION_ASSIGNED", "EXCEPTION_NOTE_ADDED",
                                     "EXCEPTION_STATUS_CHANGED"]
    assert set(h["actor"]) == {"asha"}
    last = h.iloc[-1]
    assert (last["old_value"], last["new_value"]) == ("IN_REVIEW", "RESOLVED") and "booked V9" in last["details"]


def test_a_rejected_action_leaves_no_trace(mini_run):
    e = exc_by_row(mini_run.run_id, "BANK-0000002")["exception_id"]
    before = len(repository.audit_trail(limit=None))
    with pytest.raises(WorkflowError):
        repository.update_exception(e, actor="asha", status="RESOLVED")
    assert len(repository.audit_trail(limit=None)) == before
    assert repository.get_exception(e)["status"] == "OPEN"


def test_unknown_exception_id(tmp_db):
    with pytest.raises(WorkflowError, match="does not exist"):
        repository.update_exception(999, actor="asha", status="IN_REVIEW")


# ----------------------------------------------------------------- manual matching

def test_manual_match_pairs_the_items_closes_both_and_moves_the_reconciliation_rate(mini_run):
    run_id = mini_run.run_id
    a, b = exc_by_row(run_id, "BANK-0000002"), exc_by_row(run_id, "LEDGER-0000003")     # Zenith vs Gamma (a reviewer decides)
    before = repository.kpis(run_id)
    out = repository.manual_match(a["exception_id"], b["exception_id"], actor="asha", note="same vendor, renamed")
    after = repository.kpis(run_id)
    assert after["matched_rows"] == before["matched_rows"] + 2 and after["exceptions_open"] == before["exceptions_open"] - 2
    for eid in out["exception_ids"]:
        row = repository.get_exception(eid)
        assert row["status"] == "RESOLVED" and row["resolution_code"] == "MATCHED_MANUALLY" and row["resolved_by"] == "asha"
    m = repository.matches(run_id, match_types=["MANUAL"])
    assert len(m) == 1 and m.iloc[0]["decided_by"] == "asha" and m.iloc[0]["pair_name"] == "BANK_VS_LEDGER"
    assert m.iloc[0]["row_id_a"].startswith("BANK") and m.iloc[0]["amount_diff"] == 400.0
    assert audit_types(run_id).count("MANUAL_MATCH") == 2                 # one entry in each exception's history


def test_manual_match_rules(mini_run):
    run_id = mini_run.run_id
    a, b = exc_by_row(run_id, "BANK-0000002"), exc_by_row(run_id, "BANK-0000001")
    c = exc_by_row(run_id, "LEDGER-0000003")
    with pytest.raises(WorkflowError, match="itself"):
        repository.manual_match(a["exception_id"], a["exception_id"], actor="asha")
    with pytest.raises(WorkflowError, match="two different sources"):
        repository.manual_match(a["exception_id"], b["exception_id"], actor="asha")
    repository.update_exception(c["exception_id"], actor="asha", status="DISMISSED", resolution_code="FALSE_POSITIVE", note="n")
    with pytest.raises(WorkflowError, match="already DISMISSED"):
        repository.manual_match(a["exception_id"], c["exception_id"], actor="asha")
    assert repository.get_exception(a["exception_id"])["status"] == "OPEN"        # nothing half-applied


# ----------------------------------------------------------------- bulk actions

def test_bulk_update_applies_skips_and_audits_each_item(mini_run):
    run_id = mini_run.run_id
    ids = repository.exception_ids(run_id)
    first = ids[0]
    repository.update_exception(first, actor="asha", status="DISMISSED", resolution_code="FALSE_POSITIVE", note="n")
    res = repository.bulk_update(ids, actor="ravi", status="RESOLVED", resolution_code="TIMING_DIFFERENCE", note="month-end cut-off")
    assert len(res["updated"]) == 3 and [s[0] for s in res["skipped"]] == [first]
    assert all(repository.get_exception(i)["status"] == "RESOLVED" for i in res["updated"])
    trail = repository.audit_trail(run_id, event_types=["EXCEPTION_STATUS_CHANGED"])
    assert trail["details"].str.contains(res["bulk_id"]).sum() == 3
    assert "BULK_ACTION" in audit_types(run_id)


# ----------------------------------------------------------------- integrity enforced by the database

def test_audit_log_cannot_be_rewritten_or_deleted(mini_run):
    with db.get_engine().begin() as conn:
        conn.execute(text("INSERT INTO audit_log (event_time, event_type, actor) VALUES (CURRENT_TIMESTAMP, 'X', 'a')"))
    for statement in ("UPDATE audit_log SET actor = 'someone_else'", "DELETE FROM audit_log"):
        with pytest.raises(DBAPIError, match="append-only"):
            with db.get_engine().begin() as conn:
                conn.execute(text(statement))
    assert len(repository.audit_trail(limit=None)) > 0


def test_status_check_constraint_holds_even_when_the_workflow_is_bypassed(mini_run):
    with pytest.raises(DBAPIError):
        with db.get_engine().begin() as conn:
            conn.execute(text("UPDATE exceptions SET status = 'BOGUS'"))


def test_foreign_keys_stop_dangling_references(mini_run):
    with pytest.raises(DBAPIError):
        with db.get_engine().begin() as conn:
            conn.execute(text(
                "INSERT INTO exceptions (run_id, row_id, exception_type, status) "
                f"VALUES ('{mini_run.run_id}', 'BANK-9999999', 'MISSING', 'OPEN')"))


# ----------------------------------------------------------------- reads and views

def test_queue_filters_ordering_and_totals(mini_run):
    run_id = mini_run.run_id
    q = repository.review_queue(run_id)
    assert q["priority_score"].is_monotonic_decreasing
    assert set(repository.review_queue(run_id, types=["MISSING"])["exception_type"]) == {"MISSING"}
    assert len(repository.review_queue(run_id, sources=["LEDGER"])) == 2
    assert len(repository.review_queue(run_id, search="gamma")) == 1
    assert repository.queue_totals(run_id, types=["AMOUNT_MISMATCH"]) == {"count": 2, "amount": 18000.0}
    assert len(repository.review_queue(run_id, unassigned=True)) == 4


def test_sql_views_agree_with_the_underlying_tables(mini_run):
    run_id = mini_run.run_id
    summary = repository.exception_summary(run_id)
    assert summary["exception_count"].sum() == 4 and summary["unresolved_amount"].sum() == 19400.0
    src = repository.source_reconciliation(run_id).set_index("source_system")
    assert (src["total_rows"].to_dict(), src["matched_rows"].to_dict()) == ({"BANK": 4, "LEDGER": 4}, {"BANK": 2, "LEDGER": 2})
    ranked = repository.review_queue(run_id).iloc[0]
    with db.get_engine().connect() as conn:
        top = conn.execute(text("SELECT row_id FROM v_review_queue WHERE queue_rank = 1")).scalar()
    assert top == ranked["row_id"]
    assert repository.exception_aging(run_id)["exception_count"].sum() == 4
    assert repository.run_trend().iloc[0]["unresolved_count"] == 4


def test_kpis_track_resolution_progress(mini_run):
    run_id = mini_run.run_id
    ids = repository.exception_ids(run_id)
    repository.update_exception(ids[0], actor="asha", status="RESOLVED", resolution_code="BANK_ERROR")
    k = repository.kpis(run_id)
    assert k["exceptions_open"] == 3 and k["exceptions_closed"] == 1 and k["resolution_rate"] == 25.0
    assert k["unresolved_amount"] < 19400.0
