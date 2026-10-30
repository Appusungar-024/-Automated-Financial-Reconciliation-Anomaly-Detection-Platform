"""Command-line entry points (thin wrappers live in the project root)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import config, db, evaluation, ingestion, pipeline


def _logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def run_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reconcile Bank / Ledger / Payment Gateway / ERP files (CSV or Excel).")
    ap.add_argument("--dir", default=None, help=f"folder holding bank/ledger/payment_gateway/erp files (default {config.RAW_DIR})")
    for src in ("bank", "ledger", "gateway", "erp"):
        ap.add_argument(f"--{src}", default=None, help=f"explicit path to the {src} file (CSV or Excel)")
    ap.add_argument("--no-db", action="store_true", help="do not persist (console report only)")
    ap.add_argument("--notes", default="", help="free-text note stored with the run")
    ap.add_argument("--actor", default="cli", help="who is running this (recorded in the audit log)")
    ap.add_argument("--evaluate", action="store_true", help="also grade against ground-truth labels (simulated data only)")
    args = ap.parse_args(argv)
    _logging()

    explicit = {"BANK": args.bank, "LEDGER": args.ledger, "PAYMENT_GATEWAY": args.gateway, "ERP": args.erp}
    files = {k: Path(v) for k, v in explicit.items() if v} or None
    directory = Path(args.dir) if args.dir else None
    try:
        r = pipeline.run(files, directory=directory, persist=not args.no_db, notes=args.notes, actor=args.actor)
    except pipeline.PipelineError as exc:
        print(f"\nRun failed: {exc}", file=sys.stderr)
        return 2

    print(f"\nRun {r.run_id}   [{db.backend_label() if r.persisted else 'not persisted'}]")
    print("Files")
    for s, res in r.ingestion.items():
        where = f"sheet {res.sheet_name!r}, " if res.sheet_name else ""
        print(f"  {s:16s} {res.file_name:22s} {where}header row {res.header_row + 1}:  read {res.rows_read:>7,}  "
              f"accepted {res.rows_accepted:>7,}  excluded {res.rows_excluded:>5,}")
    if not r.failures.empty:
        by = r.failures.groupby("severity").size().to_dict()
        print(f"Data quality: {len(r.failures):,} findings {by}  (stored in validation_failures)")
    print("Matching cascade (matched pairs per stage)")
    pivot = r.stage_stats.pivot_table(index="pair_name", columns="stage", values="matched_pairs", aggfunc="sum", fill_value=0)
    order = [c for c in ("REFERENCE", "EXACT", "DATE_AMOUNT", "AMOUNT_TOLERANCE", "FUZZY_VENDOR") if c in pivot.columns]
    print(pivot[order].to_string())
    print("\n" + r.headline())
    print(f"Value reconciled: {config.CURRENCY_SYMBOL}{r.matched_value:,.0f} of {config.CURRENCY_SYMBOL}{r.total_value:,.0f} "
          f"({r.matched_value / r.total_value * 100:.1f}%)")
    if r.exception_count:
        print("\nExceptions by type / priority band")
        print(pd.crosstab(r.exceptions["exception_type"], r.exceptions["priority_band"]).to_string())
        top = r.exceptions.sort_values("priority_score", ascending=False).head(5)
        print("\nTop of the review queue")
        for row in top.itertuples():
            print(f"  [{row.priority_band:8s} {row.priority_score:5.1f}] {row.exception_type:15s} {row.source_system:16s} "
                  f"{str(row.vendor_raw)[:30]:30s} {config.CURRENCY_SYMBOL}{row.amount:>13,.2f}")
    flagged = int(r.raw["is_suspicious"].sum())
    print(f"\nAnomaly detection: {flagged:,} transactions flagged (risk >= {config.RISK_FLAG_THRESHOLD})")
    if args.evaluate:
        d = directory or (Path(next(iter(files.values()))).parent if files else config.RAW_DIR)
        print("\n" + evaluation.format_report(evaluation.evaluate(r, d)))
    return 0


def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict("records") if obj.index.name else obj.to_dict("records")
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def evaluate_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the pipeline on simulated data and grade it against the held-out labels.")
    ap.add_argument("--dir", default=str(config.RAW_DIR))
    ap.add_argument("--json", default=None, help="also write the metrics to this JSON file")
    args = ap.parse_args(argv)
    _logging()
    d = Path(args.dir)
    r = pipeline.run(directory=d, persist=False)
    rep = evaluation.evaluate(r, d)
    print(f"Run on {d}:  {r.headline()}\n")
    print(evaluation.format_report(rep))
    if args.json:
        Path(args.json).write_text(json.dumps({"headline": r.headline(), **_jsonable(rep)}, indent=2, default=str))
        print(f"\nmetrics written to {args.json}")
    return 0
