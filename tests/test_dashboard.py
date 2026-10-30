"""Drives the real dashboard script headlessly (Streamlit's AppTest) -- including the failure paths."""
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from recon import config, repository

APP = str(config.BASE_DIR / "dashboard.py")


@pytest.fixture(autouse=True)
def fresh_streamlit_cache():
    st.cache_data.clear()              # dashboard readers are cached; each test has its own database
    yield
    st.cache_data.clear()


def open_app(timeout=90) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def eid(run_id, row_id):
    return repository.exception_for_row(run_id, row_id)["exception_id"]


def select_exception(at, exception_id):
    at.selectbox(key="queue_select").select(exception_id).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def errors(at):
    return [e.value for e in at.error]


# ------------------------------------------------------------------ rendering

def test_every_tab_renders_for_a_populated_run(mini_run):
    at = open_app()
    assert [t.label for t in at.tabs] == ["📊 Overview", "🧭 Review Queue", "🚨 Anomalies", "🔗 Matches",
                                          "🧹 Data Quality", "📜 Audit Trail", "⚙️ Run & Ingest"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Reconciliation rate"] == "50.0%" and metrics["Transactions"] == "8"
    assert metrics["Open exceptions"] == "4" and metrics["Unresolved exposure"] == "₹19,400"
    assert mini_run.run_id in at.sidebar.selectbox(key="run_select").value


def test_empty_database_guides_the_user_instead_of_crashing(tmp_db):
    at = open_app()
    assert any("No completed runs" in i.value for i in at.sidebar.info)
    assert sum("No run selected" in i.value for i in at.info) == 6          # every tab but Run & Ingest
    assert any(b.key == "gen_run" for b in at.button)                       # the way forward is on screen


def test_control_totals_are_shown_and_pass(mini_run):
    at = open_app()
    controls = next(df for df in at.dataframe if "check" in df.value.columns)
    assert controls.value["result"].str.contains("pass").all()


def test_reviewer_name_comes_from_the_sidebar_env_default(mini_run, monkeypatch):
    monkeypatch.setenv("RECON_USER", "meera")
    at = open_app()
    assert at.sidebar.text_input(key="reviewer").value == "meera"


# ------------------------------------------------------------------ review workflow through the UI

def test_resolving_an_exception_records_who_did_it(mini_run):
    run_id = mini_run.run_id
    e = eid(run_id, "BANK-0000002")
    at = open_app()
    at.sidebar.text_input(key="reviewer").set_value("priya").run()
    select_exception(at, e)
    at.selectbox(key=f"status_{e}_0").select("RESOLVED").run()
    at.selectbox(key=f"code_{e}_0").select("TIMING_DIFFERENCE")
    at.text_area(key=f"note_{e}_0").input("clears next cycle")
    at.button(key=f"apply_{e}_0").click().run()
    assert not at.exception and not errors(at)
    row = repository.get_exception(e)
    assert row["status"] == "RESOLVED" and row["resolved_by"] == "priya" and row["resolution_notes"] == "clears next cycle"
    assert any("updated" in s.value for s in at.success)
    last = repository.exception_history(e).iloc[-1]
    assert last["actor"] == "priya" and last["event_type"] == "EXCEPTION_STATUS_CHANGED"


def test_the_ui_surfaces_workflow_rule_violations_and_changes_nothing(mini_run):
    e = eid(mini_run.run_id, "BANK-0000002")
    at = open_app()
    select_exception(at, e)
    at.selectbox(key=f"status_{e}_0").select("DISMISSED").run()
    at.selectbox(key=f"code_{e}_0").select("FALSE_POSITIVE")
    at.button(key=f"apply_{e}_0").click().run()                           # no note: dismissal needs a reason
    assert any("note is required" in m for m in errors(at))
    assert repository.get_exception(e)["status"] == "OPEN"
    assert repository.exception_history(e).empty


def test_taking_an_item_assigns_it_to_the_current_reviewer(mini_run):
    e = eid(mini_run.run_id, "BANK-0000002")
    at = open_app()
    at.sidebar.text_input(key="reviewer").set_value("ravi").run()
    select_exception(at, e)
    at.selectbox(key=f"status_{e}_0").select("IN_REVIEW").run()
    at.button(key=f"apply_{e}_0").click().run()
    assert repository.get_exception(e)["assigned_to"] == "ravi"


def test_manual_match_from_the_suggested_counterpart(mini_run):
    run_id = mini_run.run_id
    bank, ledger = eid(run_id, "BANK-0000001"), eid(run_id, "LEDGER-0000001")      # the Acme amount mismatch
    at = open_app()
    assert {m.label: m.value for m in at.metric}["Reconciliation rate"] == "50.0%"
    select_exception(at, bank)
    at.button(key=f"match_{bank}_0").click().run()
    assert not at.exception and not errors(at)
    assert repository.get_exception(bank)["status"] == "RESOLVED" and repository.get_exception(ledger)["status"] == "RESOLVED"
    assert len(repository.matches(run_id, match_types=["MANUAL"])) == 1
    assert {m.label: m.value for m in at.metric}["Reconciliation rate"] == "75.0%"     # the headline moved


def test_filters_narrow_the_queue(mini_run):
    at = open_app()
    at.multiselect(key="q_type").set_value(["MISSING"]).run()
    table = next(df.value for df in at.dataframe if "exception_type" in df.value.columns)
    assert set(table["exception_type"]) == {"MISSING"} and len(table) == 2
    at.multiselect(key="q_type").set_value([]).run()
    at.text_input(key="q_search").input("gamma").run()
    table = next(df.value for df in at.dataframe if "exception_type" in df.value.columns)
    assert len(table) == 1 and "Gamma" in table.iloc[0]["vendor_raw"]


def test_bulk_action_needs_confirmation_then_applies_and_reports_skips(mini_run):
    run_id = mini_run.run_id
    at = open_app()
    at.selectbox(key="bulk_status").select("RESOLVED").run()
    assert at.button(key="bulk_apply").disabled                           # not until the confirmation box is ticked
    at.selectbox(key="bulk_code").select("TIMING_DIFFERENCE")
    at.checkbox(key="bulk_confirm").check().run()
    at.button(key="bulk_apply").click().run()
    assert not at.exception
    assert repository.kpis(run_id)["exceptions_open"] == 0
    assert any("Updated 4" in s.value for s in at.success)
    assert "BULK_ACTION" in set(repository.audit_trail(run_id)["event_type"])


# ------------------------------------------------------------------ other tabs

def test_audit_tab_lists_events_and_who_did_what(mini_run):
    e = eid(mini_run.run_id, "BANK-0000002")
    repository.update_exception(e, actor="asha", status="IN_REVIEW")
    at = open_app()
    trail = next(df.value for df in at.dataframe if "event_type" in df.value.columns and "audit_id" in df.value.columns)
    assert {"RUN_COMPLETED", "EXCEPTION_STATUS_CHANGED"} <= set(trail["event_type"])
    activity = next(df.value for df in at.dataframe if "events" in df.value.columns)
    assert "asha" in set(activity["actor"])


def test_data_quality_tab_reports_findings(tmp_db, tmp_path):
    from recon import pipeline
    from tests.conftest import BANK_HEADER, LEDGER_HEADER, VALIDATION_AS_OF, write_rows
    bank = write_rows(tmp_path / "bank.csv", [BANK_HEADER, ["12/08/2026", "", "NEFT-A-1", "B1", "10", ""],
                                              ["31/02/2026", "", "NEFT-B-2", "B2", "10", ""]])
    ledger = write_rows(tmp_path / "ledger.csv", [LEDGER_HEADER, ["2026-08-12", "A", "V1", "10", "", ""]])
    pipeline.run({"BANK": bank, "LEDGER": ledger}, validation_as_of=VALIDATION_AS_OF)
    at = open_app()
    findings = next(df.value for df in at.dataframe if "code" in df.value.columns and "raw_value" in df.value.columns)
    assert "UNPARSEABLE_DATE" in set(findings["code"]) and "31/02/2026" in set(findings["raw_value"])


# ------------------------------------------------------------------ creating runs

def test_a_simulated_run_can_be_created_from_the_ui(tmp_db):
    at = open_app(timeout=180)
    at.slider(key="gen_n").set_value(300)
    at.run()
    at.button(key="gen_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    runs = repository.list_runs()
    assert len(runs) == 1 and runs.iloc[0]["status"] == "COMPLETED" and runs.iloc[0]["total_transactions"] > 400
    assert runs.iloc[0]["run_id"] in at.sidebar.selectbox(key="run_select").value      # the new run is selected
    assert any("Simulated run complete" in s.value for s in at.success)
