"""
Financial Reconciliation & Anomaly Detection Platform -- dashboard.

    streamlit run dashboard.py

Reads whatever the database holds (PostgreSQL if DATABASE_URL is set, otherwise the local SQLite
file). Every write goes through ``recon.repository``, which enforces the workflow rules and writes
the audit trail in the same transaction -- the UI cannot bypass either.
"""
from __future__ import annotations

import os
import re
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from recon import config, db, ingestion, pipeline, reports, repository, simulate
from recon.formatting import compact_inr, inr
from recon.repository import WorkflowError

st.set_page_config(page_title="Financial Reconciliation Platform", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
    .block-container { padding-top: 1.6rem; }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    .small-note { color: #64748b; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

BAND_COLORS = {"CRITICAL": "#dc2626", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#94a3b8"}
TYPE_COLORS = {"MISSING": "#ef4444", "DUPLICATE": "#f97316", "AMOUNT_MISMATCH": "#eab308",
               "DATE_MISMATCH": "#8b5cf6", "PROBABLE_MATCH": "#64748b"}
STAGES = ["REFERENCE", "EXACT", "DATE_AMOUNT", "AMOUNT_TOLERANCE", "FUZZY_VENDOR", "MANUAL"]
STAGE_COLORS = dict(zip(STAGES, ["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#10b981"]))
STATUS_ORDER = ["OPEN", "IN_REVIEW", "ESCALATED", "RESOLVED", "DISMISSED"]


# --------------------------------------------------------------------------- plumbing

try:
    db.ensure_ready()
    BACKEND = db.backend_label()
except Exception as exc:                                  # unreachable database / bad DATABASE_URL
    st.error(f"Cannot open the database: {exc}")
    st.stop()


@st.cache_data(ttl=20, show_spinner=False)
def c_runs(): return repository.list_runs()
@st.cache_data(ttl=20, show_spinner=False)
def c_kpis(run_id): return repository.kpis(run_id)
@st.cache_data(ttl=20, show_spinner=False)
def c_exception_summary(run_id): return repository.exception_summary(run_id)
@st.cache_data(ttl=20, show_spinner=False)
def c_aging(run_id): return repository.exception_aging(run_id)
@st.cache_data(ttl=20, show_spinner=False)
def c_source(run_id): return repository.source_reconciliation(run_id)
@st.cache_data(ttl=20, show_spinner=False)
def c_pairs(run_id): return repository.recon_by_pair(run_id)
@st.cache_data(ttl=20, show_spinner=False)
def c_bands(run_id): return repository.band_summary(run_id)
@st.cache_data(ttl=20, show_spinner=False)
def c_trend(): return repository.run_trend()
@st.cache_data(ttl=20, show_spinner=False)
def c_controls(run_id): return repository.control_checks(run_id)
@st.cache_data(ttl=20, show_spinner=False)
def c_quality(run_id): return repository.data_quality_summary(run_id)
@st.cache_data(ttl=20, show_spinner=False)
def c_files(run_id): return repository.source_files_df(run_id)


def refresh():
    st.cache_data.clear()


def chart(fig, **kw):
    fig.update_layout(template="plotly_white", margin=dict(l=0, r=10, t=30, b=0), height=kw.pop("height", 320))
    st.plotly_chart(fig, width="stretch", **kw)


def flash():
    msg = st.session_state.pop("flash", None)
    if msg:
        st.success(msg)


# --------------------------------------------------------------------------- sidebar

st.sidebar.title("🏦 Reconciliation")
st.sidebar.caption(f"Storage: **{BACKEND}**")
st.sidebar.text_input("Reviewer name", value=os.getenv("RECON_USER", "analyst"), key="reviewer",
                      help="Recorded against every action in the audit trail.")
ACTOR = (st.session_state.get("reviewer") or "analyst").strip() or "analyst"

runs = c_runs()
done = runs[runs["status"] == "COMPLETED"] if not runs.empty else runs
label_to_run = {}
for r in done.itertuples():
    label_to_run[f"{r.run_id}  ·  {pd.Timestamp(r.started_at):%d %b %H:%M}  ·  {float(r.match_rate):.1f}% matched"] = r.run_id
if "pending_run" in st.session_state:                      # set by the Run tab after a new run finishes
    target = st.session_state.pop("pending_run")
    st.session_state["run_select"] = next((l for l, rid in label_to_run.items() if rid == target), None) \
        or st.session_state.get("run_select")
if label_to_run:
    RUN_ID = label_to_run[st.sidebar.selectbox("Reconciliation run", list(label_to_run), key="run_select")]
    st.sidebar.caption(f"{len(done)} completed run(s) in the database")
else:
    RUN_ID = None
    st.sidebar.info("No completed runs yet. Use **Run & Ingest** to create one.")
st.sidebar.divider()
st.sidebar.caption("Cascade: Reference → Exact → Date+Amount → Amount tolerance → Fuzzy vendor → "
                   "AI-assisted classification (vendor-gated, mutual-best)")
st.sidebar.caption("Risk: rules + Isolation Forest, blended by noisy-OR")

st.title("Financial Reconciliation & Anomaly Detection")
tab_overview, tab_queue, tab_anom, tab_match, tab_quality, tab_audit, tab_run = st.tabs(
    ["📊 Overview", "🧭 Review Queue", "🚨 Anomalies", "🔗 Matches", "🧹 Data Quality", "📜 Audit Trail", "⚙️ Run & Ingest"])


def need_run():
    st.info("No run selected yet. Open **Run & Ingest** to reconcile a dataset or upload your own files.")


# --------------------------------------------------------------------------- overview

def render_overview(run_id: str):
    k = c_kpis(run_id)
    run = k["run"]
    st.caption(f"Run `{run_id}` · {pd.Timestamp(run['started_at']):%d %b %Y %H:%M} UTC · as-of {run['as_of_date']}"
               + (f" · {run['notes']}" if run["notes"] else ""))
    c = st.columns(5)
    c[0].metric("Reconciliation rate", f"{k['match_rate']:.1f}%", f"{k['value_match_rate']:.1f}% by value",
                delta_color="off", border=True,
                help="Share of source rows matched. Live: moves as reviewers match items by hand.")
    c[1].metric("Transactions", f"{k['total_rows']:,}", f"{k['matched_rows']:,} matched", delta_color="off", border=True)
    c[2].metric("Open exceptions", f"{k['exceptions_open']:,}", f"of {k['exceptions_total']:,} raised",
                delta_color="off", border=True)
    c[3].metric("Unresolved exposure", compact_inr(k["unresolved_amount"]), inr(k["unresolved_amount"]),
                delta_color="off", border=True,
                help="Gross value of items still open. A mismatched pair counts both sides.")
    c[4].metric("Avg age of open items", f"{k['avg_age_open']:.0f} days", f"{k['urgent_open']:,} critical / high",
                delta_color="off", border=True)
    c = st.columns(4)
    c[0].metric("Resolution rate", f"{k['resolution_rate']:.1f}%", f"{k['exceptions_closed']:,} closed", delta_color="off", border=True)
    c[1].metric("Value reconciled", compact_inr(k["matched_value"]), f"of {compact_inr(k['total_value'])}", delta_color="off", border=True)
    c[2].metric("Flagged suspicious", f"{k['suspicious']:,}", "risk ≥ %d" % config.RISK_FLAG_THRESHOLD, delta_color="off", border=True)
    c[3].metric("Rows quarantined", f"{k['quarantined_rows']:,}", "failed validation at ingestion", delta_color="off", border=True)

    st.divider()
    left, right = st.columns(2)
    summary = c_exception_summary(run_id)
    with left:
        st.subheader("Unresolved exceptions by type")
        if summary.empty or summary["unresolved_count"].sum() == 0:
            st.success("Nothing unresolved.")
        else:
            by_type = summary.groupby("exception_type")[["unresolved_count", "unresolved_amount"]].sum().reset_index()
            by_value = st.radio("Measure", ["Value", "Count"], horizontal=True, key="ov_measure", label_visibility="collapsed")
            col = "unresolved_amount" if by_value == "Value" else "unresolved_count"
            by_type = by_type.sort_values(col)
            fig = px.bar(by_type, x=col, y="exception_type", orientation="h", color="exception_type",
                         color_discrete_map=TYPE_COLORS,
                         text=by_type[col].map(compact_inr if by_value == "Value" else "{:,}".format))
            fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="")
            chart(fig)
    with right:
        st.subheader("Ageing of unresolved items")
        aging = c_aging(run_id)
        if aging.empty:
            st.success("Nothing unresolved.")
        else:
            fig = px.bar(aging.sort_values("bucket_order"), x="aging_bucket", y="unresolved_amount",
                         text=aging.sort_values("bucket_order")["exception_count"].map("{:,} items".format),
                         color_discrete_sequence=["#2563eb"])
            fig.update_layout(xaxis_title="", yaxis_title="Unresolved value")
            chart(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Reconciliation rate by source")
        src = c_source(run_id).copy()
        src["By count"] = src["matched_rows"] / src["total_rows"] * 100
        src["By value"] = src["matched_value"] / src["total_value"] * 100
        fig = go.Figure([go.Bar(name=n, x=src["source_system"], y=src[n], text=src[n].round(1).astype(str) + "%")
                         for n in ("By count", "By value")])
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 105], title="% matched"), xaxis_title="")
        chart(fig)
    with right:
        st.subheader("Where matches were found")
        pairs = c_pairs(run_id)
        if not pairs.empty:
            fig = px.bar(pairs, x="pair_name", y="matches", color="match_type", color_discrete_map=STAGE_COLORS,
                         category_orders={"match_type": STAGES})
            fig.update_layout(xaxis_title="", yaxis_title="Matched pairs", legend_title="")
            chart(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Open items by priority band")
        bands = c_bands(run_id)
        if bands.empty:
            st.success("Nothing open.")
        else:
            order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            bands = bands.set_index("priority_band").reindex(order).dropna().reset_index()
            fig = px.bar(bands, x="priority_band", y="exceptions", color="priority_band", color_discrete_map=BAND_COLORS,
                         text=bands["amount"].map(compact_inr))
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Open exceptions")
            chart(fig)
    with right:
        st.subheader("Trend across runs")
        trend = c_trend()
        if len(trend) < 2:
            st.caption("Run more reconciliations to see a trend.")
        else:
            fig = px.line(trend, x="started_at", y="match_rate", markers=True)
            fig.update_layout(xaxis_title="", yaxis_title="Match rate %")
            chart(fig)

    st.subheader("Control totals")
    st.caption("Tick-and-tie checks that must always hold, including after reviewers act.")
    checks = pd.DataFrame(c_controls(run_id))
    checks["result"] = checks["passed"].map({True: "✅ pass", False: "❌ FAIL"})
    st.dataframe(checks[["check", "expected", "actual", "result"]], hide_index=True, width="stretch")


# --------------------------------------------------------------------------- review queue

QUEUE_COLUMNS = ["exception_id", "priority_band", "priority_score", "exception_type", "status", "assigned_to",
                 "source_system", "txn_date", "vendor_raw", "amount", "age_days", "risk_score",
                 "suggested_match_confidence"]
QUEUE_CONFIG = {
    "exception_id": st.column_config.NumberColumn("ID", format="%d"),
    "priority_band": "Priority", "priority_score": st.column_config.NumberColumn("Score", format="%.1f"),
    "exception_type": "Type", "status": "Status", "assigned_to": "Assigned to", "source_system": "Source",
    "txn_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"), "vendor_raw": "Vendor / narration",
    "amount": st.column_config.NumberColumn("Amount", format="%.2f"), "age_days": st.column_config.NumberColumn("Age (d)", format="%d"),
    "risk_score": st.column_config.NumberColumn("Risk", format="%.0f"),
    "suggested_match_confidence": st.column_config.NumberColumn("Match conf.", format="%.0f"),
}


def queue_filters(actor: str) -> dict:
    c = st.columns([1.4, 1.4, 1.2, 1.2, 1.1])
    statuses = c[0].multiselect("Status", STATUS_ORDER, default=list(config.OPEN_STATUSES), key="q_status")
    types = c[1].multiselect("Type", sorted(config.EXCEPTION_SEVERITY), key="q_type")
    sources = c[2].multiselect("Source", list(config.SOURCES), key="q_source")
    bands = c[3].multiselect("Priority", list(BAND_COLORS), key="q_band")
    who = c[4].selectbox("Assignee", ["Anyone", "Assigned to me", "Unassigned"], key="q_assignee")
    c = st.columns([1, 2])
    min_amount = c[0].number_input("Minimum amount", min_value=0.0, value=0.0, step=1000.0, key="q_min")
    search = c[1].text_input("Search vendor, reference or row id", key="q_search")
    f = dict(statuses=statuses or None, types=types or None, sources=sources or None, bands=bands or None,
             min_amount=min_amount or None, search=search or None)
    if who == "Assigned to me":
        f["assigned_to"] = actor
    elif who == "Unassigned":
        f["unassigned"] = True
    return f


def comparison_table(exc: dict, other: dict | None):
    rows = [("Source", exc["source_system"], other and other["source_system"]),
            ("Date", str(exc["txn_date"])[:10], other and str(other["txn_date"])[:10]),
            ("Vendor / narration", exc["vendor_raw"], other and other["vendor_raw"]),
            ("Reference", exc["reference"], other and other["reference"]),
            ("Amount", inr(exc["amount"], 2), other and inr(other["amount"], 2))]
    if other:
        gap = float(other["amount"]) - float(exc["amount"])
        days = (pd.Timestamp(other["txn_date"]) - pd.Timestamp(exc["txn_date"])).days
        rows.append(("Difference", "", f"{days:+d} days · {inr(gap, 2)}"))
    return pd.DataFrame(rows, columns=["", "This item", "Suggested counterpart"])


def render_exception_detail(run_id: str, eid: int, actor: str):
    exc = repository.get_exception(eid)
    txn = repository.get_transaction(run_id, exc["row_id"]) or {}
    ver = st.session_state.get("form_ver", 0)
    suffix = f"{eid}_{ver}"
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(f"#### Exception #{eid} · {exc['exception_type'].replace('_', ' ').title()}")
        st.markdown(f"**{exc['priority_band']}** priority ({exc['priority_score']:.1f}) · status **{exc['status']}** · "
                    f"assigned to **{exc['assigned_to'] or 'nobody'}** · {exc['age_days']} days old")
        other = other_exc = None
        if exc["suggested_match_row_id"]:
            other = repository.get_transaction(run_id, exc["suggested_match_row_id"])
            other_exc = repository.exception_for_row(run_id, exc["suggested_match_row_id"])
            st.info(f"Suggested counterpart `{exc['suggested_match_row_id']}` "
                    f"({exc['suggested_match_confidence']:.0f}% confidence): {exc['suggested_match_reason']}")
        elif exc["suggested_match_reason"]:
            st.info(exc["suggested_match_reason"])
        st.dataframe(comparison_table(exc, other), hide_index=True, width="stretch")
        if txn.get("is_suspicious"):
            st.warning(f"Flagged as suspicious (risk {txn['risk_score']:.0f}): {txn['suspicion_reasons']}")
        elif txn.get("risk_score"):
            st.caption(f"Transaction risk score {txn['risk_score']:.0f}")
        if exc["resolution_code"]:
            st.success(f"Resolved as **{exc['resolution_code']}** by {exc['resolved_by']}"
                       + (f": {exc['resolution_notes']}" if exc["resolution_notes"] else ""))
        with st.expander("History (audit trail)"):
            hist = repository.exception_history(eid)
            if hist.empty:
                st.caption("No actions recorded yet.")
            else:
                st.dataframe(hist[["event_time", "event_type", "actor", "old_value", "new_value", "details"]],
                             hide_index=True, width="stretch")

    with right:
        st.markdown("#### Take action")
        current = exc["status"]
        options = [current] + [s for s in STATUS_ORDER if s in config.STATUS_TRANSITIONS[current]]
        new_status = st.selectbox("Status", options, key=f"status_{suffix}")
        assignee = st.text_input("Assigned to", value=exc["assigned_to"] or "", key=f"assignee_{suffix}")
        closing = new_status in config.CLOSED_STATUSES and new_status != current
        code = None
        if closing:
            code = st.selectbox("Resolution code", list(config.RESOLUTION_CODES), key=f"code_{suffix}",
                                format_func=lambda c: f"{c} — {config.RESOLUTION_CODES[c]}")
        note = st.text_area("Note", key=f"note_{suffix}", placeholder="Required when dismissing, re-opening, "
                            "or using FALSE_POSITIVE / WRITTEN_OFF / OTHER")
        if st.button("Apply", type="primary", key=f"apply_{suffix}", width="stretch"):
            try:
                repository.update_exception(
                    eid, actor=actor, status=new_status if new_status != current else None,
                    assigned_to=assignee if assignee != (exc["assigned_to"] or "") else None,
                    resolution_code=code, note=note)
            except WorkflowError as err:
                st.error(str(err))
            else:
                refresh()
                st.session_state["form_ver"] = ver + 1
                st.session_state["flash"] = f"Exception #{eid} updated."
                st.rerun()
        if other_exc and other_exc["status"] in config.OPEN_STATUSES and exc["status"] in config.OPEN_STATUSES:
            st.divider()
            st.caption(f"The suggested counterpart is open as exception #{other_exc['exception_id']}. "
                       "Matching them records a MANUAL match and closes both.")
            if st.button(f"Match with #{other_exc['exception_id']}", key=f"match_{suffix}", width="stretch"):
                try:
                    repository.manual_match(eid, other_exc["exception_id"], actor=actor, note=note)
                except WorkflowError as err:
                    st.error(str(err))
                else:
                    refresh()
                    st.session_state["form_ver"] = ver + 1
                    st.session_state["flash"] = f"Exceptions #{eid} and #{other_exc['exception_id']} matched manually."
                    st.rerun()


def render_bulk(run_id: str, filters: dict, actor: str, total: int):
    with st.expander(f"Bulk action on all {total:,} exceptions matching the filters"):
        st.caption("Applies one decision to every exception currently in the filtered queue. Items that break a "
                   "workflow rule are skipped and reported; every change is audited individually.")
        c = st.columns(3)
        status = c[0].selectbox("Set status to", ["IN_REVIEW", "ESCALATED", "RESOLVED", "DISMISSED"], key="bulk_status")
        code = c[1].selectbox("Resolution code", ["—"] + list(config.RESOLUTION_CODES), key="bulk_code") \
            if status in config.CLOSED_STATUSES else None
        note = c[2].text_input("Note", key="bulk_note")
        ok = st.checkbox(f"I understand this changes up to {total:,} exceptions", key="bulk_confirm")
        if st.button("Apply to all", disabled=not ok or total == 0, key="bulk_apply"):
            ids = repository.exception_ids(run_id, **filters)
            res = repository.bulk_update(ids, actor=actor, status=status,
                                         resolution_code=None if code in (None, "—") else code, note=note)
            refresh()
            msg = f"Updated {len(res['updated']):,} exceptions (batch {res['bulk_id']})."
            if res["skipped"]:
                reasons = pd.Series([r for _, r in res["skipped"]]).value_counts().head(3).to_dict()
                msg += f" Skipped {len(res['skipped']):,}: {reasons}"
            st.session_state["flash"] = msg
            st.rerun()


def render_queue(run_id: str, actor: str):
    flash()
    filters = queue_filters(actor)
    totals = repository.queue_totals(run_id, **filters)
    st.markdown(f"**{totals['count']:,}** exceptions · gross value **{inr(totals['amount'])}**"
                + ("  ·  showing the 200 highest priority" if totals["count"] > 200 else ""))
    df = repository.review_queue(run_id, limit=200, **filters)
    if df.empty:
        st.info("No exceptions match these filters.")
    else:
        st.dataframe(df[QUEUE_COLUMNS], hide_index=True, width="stretch", height=300, column_config=QUEUE_CONFIG)

    c = st.columns([2, 1, 1])
    labels = {int(r.exception_id): f"#{r.exception_id} · {r.priority_band} {r.priority_score:.0f} · {r.exception_type} · "
                                   f"{str(r.vendor_raw)[:26]} · {inr(r.amount)}" for r in df.itertuples()}
    if labels:
        chosen = c[0].selectbox("Open an exception to review", list(labels), format_func=labels.get, key="queue_select")
    if c[1].button("Prepare Excel export", key="prep_export") and totals["count"]:
        st.session_state["export_xlsx"] = reports.exceptions_workbook(repository.review_queue(run_id, limit=None, **filters))
    if "export_xlsx" in st.session_state:
        c[2].download_button("⬇ Download Excel", st.session_state["export_xlsx"], file_name=f"exceptions_{run_id}.xlsx",
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_export")
    render_bulk(run_id, filters, actor, totals["count"])
    if labels:
        st.divider()
        render_exception_detail(run_id, int(chosen), actor)


# --------------------------------------------------------------------------- anomalies

def render_anomalies(run_id: str):
    st.markdown(
        "Every transaction gets a **risk score (0–100)**: explainable *rules* (reference reuse, structuring just under "
        f"{inr(config.STRUCTURING_THRESHOLD)}, round-number and weekend payments, per-vendor outliers, large payments to "
        "new vendors) blended with an unsupervised **Isolation Forest** that scores how unusual a transaction is across "
        "12 features. Either strong signal is enough; agreement raises the score.")
    with st.expander("How the score is built"):
        st.table(pd.DataFrame({"Rule": list(config.RULE_POINTS), "Points": list(config.RULE_POINTS.values())}))
        st.caption(f"Rule points are capped at 100. The model contributes only for its top {100 * (1 - config.ML_TAIL_START):.0f}% "
                   f"most unusual transactions. Flag threshold: risk ≥ {config.RISK_FLAG_THRESHOLD}.")
    c = st.columns([1, 1, 3])
    min_risk = c[0].slider("Minimum risk", 0, 100, config.RISK_FLAG_THRESHOLD, key="an_min")
    only_flagged = c[1].checkbox("Flagged only", value=False, key="an_flagged")
    df = repository.top_risky(run_id, min_risk=float(min_risk), flagged_only=only_flagged, limit=500)
    if df.empty:
        st.info("No transactions at or above this risk level.")
        return
    c[2].markdown(f"**{len(df):,}** transactions shown (top 500 by risk)")
    fig = px.histogram(df, x="risk_score", nbins=20, color_discrete_sequence=["#dc2626"])
    fig.update_layout(xaxis_title="Risk score", yaxis_title="Transactions")
    chart(fig, height=220)
    st.dataframe(df[["row_id", "source_system", "txn_date", "vendor_raw", "amount", "risk_score", "rule_score",
                     "ml_score", "suspicion_reasons"]], hide_index=True, width="stretch", height=420,
                 column_config={"txn_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                                "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                                "risk_score": st.column_config.NumberColumn("Risk", format="%.0f"),
                                "rule_score": st.column_config.NumberColumn("Rules", format="%.0f"),
                                "ml_score": st.column_config.NumberColumn("Model", format="%.0f"),
                                "suspicion_reasons": "Why"})


# --------------------------------------------------------------------------- matches

def render_matches(run_id: str):
    st.caption("Every match records why it was made. Tolerance, fuzzy and manual matches are judgement calls, "
               "so they are the ones worth challenging.")
    c = st.columns([2, 2, 1.5])
    pairs = c[0].multiselect("Pair", [p["name"] for p in config.RECON_PAIRS], key="m_pair")
    types = c[1].multiselect("Rule", STAGES, key="m_type")
    judgmental = c[2].checkbox("Judgemental decisions only", value=True, key="m_judg")
    df = repository.matches(run_id, pair_names=pairs or None, match_types=types or None,
                            judgmental_only=judgmental and not types, limit=1000)
    st.markdown(f"**{len(df):,}** matches shown (lowest confidence first, max 1,000)")
    if df.empty:
        st.info("No matches for this selection.")
        return
    st.dataframe(df[["match_id", "pair_name", "match_type", "match_confidence", "date_diff_days", "amount_diff",
                     "vendor_similarity", "vendor_a", "vendor_b", "amount_a", "amount_b", "rule_applied", "decided_by"]],
                 hide_index=True, width="stretch", height=480,
                 column_config={"match_confidence": st.column_config.NumberColumn("Conf.", format="%.0f"),
                                "amount_diff": st.column_config.NumberColumn("Amount Δ", format="%.2f"),
                                "date_diff_days": st.column_config.NumberColumn("Date Δ (d)", format="%d"),
                                "vendor_similarity": st.column_config.NumberColumn("Vendor sim.", format="%.0f"),
                                "rule_applied": "Why matched"})


# --------------------------------------------------------------------------- data quality

def render_quality(run_id: str):
    st.markdown("Nothing is silently coerced: every row ingestion refused, skipped or flagged is recorded here with the reason.")
    files = c_files(run_id)
    if not files.empty:
        st.subheader("Source files")
        view = files.assign(sha256=files["sha256"].str[:16] + "…")
        st.dataframe(view[["source_system", "file_name", "sheet_name", "header_row", "rows_read", "rows_accepted",
                           "rows_excluded", "sha256"]], hide_index=True, width="stretch")
        with st.expander("Column mapping detected for each file"):
            import json
            for r in files.itertuples():
                st.markdown(f"**{r.source_system}** — `{r.file_name}`")
                st.json(json.loads(r.column_mapping or "{}"), expanded=False)
    summary = c_quality(run_id)
    if summary.empty:
        st.success("No validation findings on this run.")
        return
    st.subheader("Findings")
    fig = px.bar(summary, x="code", y="failures", color="severity", facet_col="source_system",
                 color_discrete_map={"ERROR": "#dc2626", "WARNING": "#f59e0b", "INFO": "#94a3b8"})
    fig.update_layout(xaxis_title="", yaxis_title="Findings")
    chart(fig, height=280)
    c = st.columns(3)
    sev = c[0].multiselect("Severity", ["ERROR", "WARNING", "INFO"], default=["ERROR", "WARNING"], key="dq_sev")
    src = c[1].multiselect("Source", list(config.SOURCES), key="dq_src")
    code = c[2].multiselect("Code", sorted(summary["code"].unique()), key="dq_code")
    df = repository.validation_failures_df(run_id, severity=sev or None, source=src or None, code=code or None, limit=2000)
    st.markdown(f"**{len(df):,}** findings shown")
    st.dataframe(df[["source_system", "file_name", "source_row", "severity", "code", "field", "raw_value", "message",
                     "row_excluded"]], hide_index=True, width="stretch", height=360)
    st.download_button("⬇ Download findings (CSV)", df.to_csv(index=False), file_name=f"validation_{run_id}.csv",
                       mime="text/csv", key="dl_dq")
    with st.expander("What each code means"):
        st.dataframe(pd.DataFrame([(c, s, "yes" if e else "no", m) for c, (s, e, m) in ingestion.FAILURE_CODES.items()],
                                  columns=["Code", "Severity", "Row excluded", "Meaning"]), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- audit

def render_audit(run_id: str, actor: str):
    st.markdown("The audit log is **append-only, enforced by database triggers**: updates and deletes are rejected, "
                "so this record cannot be quietly rewritten.")
    c = st.columns(3)
    types = c[0].multiselect("Event type", ["RUN_STARTED", "FILE_INGESTED", "VALIDATION_SUMMARY", "MATCH_STAGE_SUMMARY",
                                            "ANOMALY_MODEL_FIT", "RUN_COMPLETED", "RUN_FAILED", "EXCEPTION_STATUS_CHANGED",
                                            "EXCEPTION_ASSIGNED", "EXCEPTION_NOTE_ADDED", "MANUAL_MATCH", "BULK_ACTION"],
                             key="au_type")
    actors = c[1].multiselect("Actor", sorted(repository.audit_trail(run_id, limit=5000)["actor"].unique()), key="au_actor")
    search = c[2].text_input("Search details or entity id", key="au_search")
    df = repository.audit_trail(run_id, event_types=types or None, actors=actors or None, search=search or None, limit=1000)
    st.markdown(f"**{len(df):,}** events (newest first, max 1,000)")
    st.dataframe(df[["audit_id", "event_time", "event_type", "actor", "entity_type", "entity_id", "old_value",
                     "new_value", "details"]], hide_index=True, width="stretch", height=380)
    c = st.columns(2)
    with c[0]:
        st.subheader("Who did what")
        st.dataframe(repository.reviewer_activity(), hide_index=True, width="stretch", height=220)
    with c[1]:
        st.subheader("Audit report")
        st.caption("A single workbook for sign-off: summary with control totals, exceptions, judgemental matches, "
                   "anomalies, data quality and the full audit trail.")
        if st.button("Build audit report", key="build_report"):
            st.session_state["audit_xlsx"] = reports.build_audit_report(run_id, actor=actor)
        if "audit_xlsx" in st.session_state:
            st.download_button("⬇ Download audit report (Excel)", st.session_state["audit_xlsx"],
                               file_name=f"audit_report_{run_id}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_report")


# --------------------------------------------------------------------------- run & ingest

def save_uploads(uploads: dict) -> dict:
    root = Path(tempfile.mkdtemp(prefix="recon_upload_"))
    files = {}
    for source, up in uploads.items():
        if up is None:
            continue
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(up.name).name)
        folder = root / source
        folder.mkdir()
        files[source] = folder / name
        files[source].write_bytes(up.getvalue())
    return files


def preview_ingestion(files: dict):
    for source, path in files.items():
        res = ingestion.ingest_file(source, path, as_of=date.today())
        icon = "❌" if res.fatal else ("⚠️" if (res.failures["severity"] == "ERROR").any() else "✅")
        with st.expander(f"{icon} {source} — {path.name}: {res.rows_accepted:,} of {res.rows_read:,} rows usable", expanded=res.fatal):
            if res.fatal:
                st.error("; ".join(res.failures.loc[res.failures["source_row"].isna(), "message"]))
                continue
            st.caption(f"Header found on row {res.header_row + 1}" + (f" of sheet “{res.sheet_name}”" if res.sheet_name else ""))
            st.dataframe(pd.DataFrame({"field": list(res.column_mapping), "column in your file": list(res.column_mapping.values())}),
                         hide_index=True, width="stretch")
            if not res.failures.empty:
                st.dataframe(res.failures.groupby(["severity", "code"]).size().rename("rows").reset_index(),
                             hide_index=True, width="stretch")
            st.dataframe(res.df[["row_id", "txn_date", "vendor_raw", "reference", "amount", "direction"]].head(5),
                         hide_index=True, width="stretch")


def finish_run(result, message: str):
    refresh()
    st.session_state["pending_run"] = result.run_id
    st.session_state["flash"] = f"{message} {result.headline()}"
    st.rerun()


def render_run(actor: str):
    flash()
    st.subheader("A · Generate a simulated dataset")
    st.caption("Bank, Ledger, Payment Gateway and ERP feeds with injected discrepancies, anomalies and junk rows. "
               "Ground-truth labels are held out, so quality can be measured honestly.")
    c = st.columns([2, 1, 2])
    n = c[0].slider("Base transactions", 300, 50_000, 3_000, step=100, key="gen_n")
    seed = c[1].number_input("Seed", value=config.RANDOM_SEED, step=1, key="gen_seed")
    excel = c[2].checkbox("Write Bank and ERP as Excel (with a statement preamble)", value=True, key="gen_excel")
    if st.button("Generate and reconcile", type="primary", key="gen_run"):
        tmp = Path(tempfile.mkdtemp(prefix="recon_sim_"))
        with st.spinner(f"Simulating ~{n:,} transactions and reconciling…"):
            simulate.main(int(n), int(seed), tmp, end_date=date.today() - timedelta(days=1),
                          excel_sources=("BANK", "ERP") if excel else ())
            result = pipeline.run(directory=tmp, notes=f"dashboard simulation n={n}, seed={seed}", actor=actor)
        finish_run(result, "Simulated run complete.")

    st.divider()
    st.subheader("B · Upload your own files")
    st.caption("CSV, TSV or Excel (.xlsx/.xls). Statement preambles, differing column names and date formats are "
               "handled; rows that cannot be trusted are quarantined and reported, never guessed at. "
               "Supply both sides of at least one pair: Bank + Ledger, Gateway + Bank, or ERP + Ledger.")
    cols = st.columns(2)
    uploads = {s: cols[i % 2].file_uploader(s.replace("_", " ").title(), type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"],
                                            key=f"up_{s}") for i, s in enumerate(config.SOURCES)}
    files = save_uploads(uploads)
    if files:
        runnable = [p["name"] for p in config.RECON_PAIRS if p["source_a"] in files and p["source_b"] in files]
        st.markdown("**Will reconcile:** " + (", ".join(runnable) if runnable else "— nothing yet (need both sides of a pair)"))
        preview_ingestion(files)
        note = st.text_input("Run note", key="up_note")
        if st.button("Reconcile uploaded files", type="primary", disabled=not runnable, key="up_run"):
            try:
                with st.spinner("Reconciling…"):
                    result = pipeline.run(files, notes=note or "uploaded via dashboard", actor=actor)
            except pipeline.PipelineError as err:
                refresh()
                st.error(f"Run failed: {err}")
            else:
                finish_run(result, "Run complete.")

    st.divider()
    st.subheader("Run history")
    history = repository.list_runs()
    if history.empty:
        st.caption("No runs yet.")
    else:
        st.dataframe(history[["run_id", "started_at", "status", "total_transactions", "matched_count", "exception_count",
                              "match_rate", "runtime_seconds", "notes", "error_message"]], hide_index=True, width="stretch")


# --------------------------------------------------------------------------- compose

with tab_overview:
    render_overview(RUN_ID) if RUN_ID else need_run()
with tab_queue:
    render_queue(RUN_ID, ACTOR) if RUN_ID else need_run()
with tab_anom:
    render_anomalies(RUN_ID) if RUN_ID else need_run()
with tab_match:
    render_matches(RUN_ID) if RUN_ID else need_run()
with tab_quality:
    render_quality(RUN_ID) if RUN_ID else need_run()
with tab_audit:
    render_audit(RUN_ID, ACTOR) if RUN_ID else need_run()
with tab_run:
    render_run(ACTOR)
