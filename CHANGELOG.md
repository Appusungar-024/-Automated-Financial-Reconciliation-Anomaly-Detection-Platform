# Changelog: v1 (bank_recon) → v2 (financial_recon_platform)

## Structure
Flat scripts → `recon/` package + thin CLI wrappers (`generate_data.py`, `run_pipeline.py`, `validate_ground_truth.py`),
`tests/`, `db/schema.sql`, `db/analytics.sql`.

| v1 file | v2 |
|---|---|
| `config.py` | `recon/config.py` (+ ingestion, anomaly, priority, workflow settings; per-run snapshot) |
| `preprocessing.py` | `recon/preprocessing.py` (normalisation only) + **new** `recon/ingestion.py` |
| `matching_engine.py` | `recon/matching.py` (+ decision rationale on every match) |
| `ai_assist.py` | split → `recon/classification.py` (rewritten) + `recon/anomaly.py` (+ scikit-learn) |
| `db_utils.py`, `db/schema.sql` | `recon/db.py`, `recon/repository.py`, rewritten `db/schema.sql`, **new** `db/analytics.sql` |
| `run_pipeline.py` | `recon/pipeline.py` + `recon/cli.py` |
| `validate_ground_truth.py` | `recon/evaluation.py` (+ ingestion, anomaly and pair-level metrics) |
| `dashboard.py` | rewritten (7 tabs) |
| — | **new:** `prioritization.py`, `reports.py`, `formatting.py`, `simulate.py` (extended generator) |

## New against the spec
- **Excel ingestion** + row-level validation + quarantine + SHA-256 provenance.
- **Isolation Forest** anomaly detection (scikit-learn) blended with rules; reasons on every flag.
- **Review workflow**: lifecycle, assignment, resolution codes, manual match, bulk actions, priority scoring.
- **Audit trail**: append-only log enforced by DB triggers; reviewer attribution; control-total checks; Excel audit report.
- **Tests**: 0 → 166.

## Bugs found and fixed (v1 defects)
- Re-running in one database crashed: `raw_transactions` was keyed by `row_id` alone → now `(run_id, row_id)`.
- Unparseable amounts silently became `0`, bad dates `NaT` → now quarantined with a reason.
- The simulator produced **zero-value transactions** (rounding sub-₹500 amounts to the nearest thousand); the new
  validator exposed it.
- Classification's candidate cap discarded true counterparts; blended scoring let unrelated rows outscore them.
- "Reference reuse" anomaly rule fired across systems where a shared reference is normal.
- SQLite bound-parameter limit could break bulk inserts on some platforms.
- Uniform simulated dates put 29% of transactions on weekends (now weekday-weighted).

## Discovered but NOT fixed
- Amount-tolerance matching precision (see `STATUS.md`, open issue #1). Affects v1 and v2.
