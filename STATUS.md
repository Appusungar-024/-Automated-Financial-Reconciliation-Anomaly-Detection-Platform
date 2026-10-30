# Project status

_Snapshot of where v2 stands. Everything marked "measured" was produced by running the code in this repo
(`make test`, `make eval`); nothing here is estimated._

## Spec checklist

| Requirement | Status | Notes |
|---|---|---|
| Multi-source ingestion: CSV **and Excel** | **Done** | `.csv/.tsv/.xlsx/.xls`; finds the header under statement preambles; alias column mapping; per-column date-format inference with DD/MM vs MM/DD ambiguity detection; multi-sheet workbooks. Tested on Excel files of ~42k rows. |
| Validation failures recorded | **Done** | Nothing is silently coerced. Bad dates/amounts, TOTAL lines, blank rows, zero amounts… are quarantined with a reason code in `validation_failures`. Measured: 398 of 398 injected junk rows caught, 0 genuine rows wrongly quarantined. |
| Intelligent reconciliation (exact → date+amount → tolerance → fuzzy) | **Done, with a known precision problem** | See *Open issue #1* below. |
| Exception detection (missing / duplicate / mismatched) | **Done** | Stage-5 classifier rewritten (vendor-gated, mutual-best). Large-amount-mismatch classification 19% → ~55% at 50k scale (still imperfect). |
| Anomaly detection (scikit-learn) | **Done** | Explainable rules + unsupervised Isolation Forest, blended by noisy-OR; every flag carries a readable reason. |
| Review workflow | **Done** | Status lifecycle with enforced transitions, assignment, resolution codes, required notes, **manual match**, bulk actions. Rules live in the repository layer, not the UI. |
| Analytics dashboard (rates, categories, unresolved amounts) | **Done** | 7 tabs; rates by count and by value; ₹ exposure; ageing; priority bands; control totals. Tested headlessly with Streamlit `AppTest`; **not yet reviewed by eye in a browser.** |
| Testing | **Done** | 166 tests: 165 pass on SQLite (1 PostgreSQL-only test skipped), **all 166 pass on live PostgreSQL 16**. Line+branch coverage 94% (measured). |
| Audit trail | **Done** | Append-only `audit_log` enforced by **database triggers** (UPDATE/DELETE rejected, verified on both PostgreSQL and SQLite); audit rows share a transaction with the change they describe; SHA-256 of every input file; config snapshot stored per run. |
| Audit reports | **Done** | One-click Excel workbook: summary + control-total checks, exceptions, judgemental matches, anomalies, data quality, full audit trail. Formula-injection protected. |
| SQL / PostgreSQL | **Done** | Hand-written `db/schema.sql` (verified against the app's schema on a real PostgreSQL), plus portable analytics views (CTEs, window functions) that the dashboard reads. |

## Measured results (simulated data; see caveat below)

50,000 base transactions → ~99k rows, Bank and ERP as Excel files:

| | v1 | v2 |
|---|---|---|
| End-to-end runtime | ~20 s | ~17 s (including parsing two Excel files and the ML model) |
| Overall row-level accuracy | 94.6% | 95.3% |
| Large amount mismatches correctly labelled | 19% | ~55% |
| Injected junk rows caught | n/a (silently coerced) | 100% |
| Anomaly ranking quality (ROC-AUC) | rules only, ~23% recall | ~0.998 blended |
| **Pair-level precision** | _not measured_ | **~91.7%** |

Anomaly detection (5,000-transaction run): rules alone flag ~52% of injected anomalies; rules + Isolation Forest
flag ~100% at ~0.7% false-positive rate. On the round-number-weekend pattern, which the rules only partly cover,
recall rises from ~39% to ~100%.

**Caveat:** these are measured on _simulated_ defects the generator injected. They show the pipeline finds those
patterns; they say nothing certain about real-world accuracy.

## Open issues

1. **Auto-matching is less precise than the match rate suggests (the most important item).**
   Pair-level precision at 50k scale is ~91.7%: about 1 in 12 auto-matched pairs links two _different_ real
   transactions. By stage: Reference 100%, Exact 100%, Date+Amount ~89%, Fuzzy ~88%, **Amount-tolerance only ~28%**.
   Consequently ~27% of rows that _should_ have been flagged were silently auto-matched, and the share of rows
   correctly paired is closer to **~84%** than the ~91% headline.
   - Cause (diagnosed): Stage 3 accepts "same day, amount within 0.5%" with no vendor check, which is close to
     meaningless when hundreds of transactions share a day; Stage 2 coincides on round amounts (99.6% of its false
     matches are multiples of 1,000).
   - **This also applies to v1**, whose "91.4% matched" counted wrong pairings as matches. Row-level accuracy cannot
     see this; the new pair-level metric (`make eval`) does.
   - Planned fix: require vendor corroboration in Stage 3 always, and in Stage 2 whenever the amount is common in the
     book (evidence scaled to coincidence risk); pair inside collision groups by vendor similarity, not row order.
     Not yet implemented.
2. UI has not been looked at in a real browser (only driven headlessly).
3. `.github/workflows/ci.yml` is written but has never run on GitHub Actions.
4. The shipped sample database was built with the current (imprecise) matcher.
5. Anomaly detection was tuned over two iterations after seeing results on simulated anomalies (a feature that
   defaulted to "no deviation" for brand-new vendors was fixed; one rule weight raised). Disclosed, not hidden.

## Not done / ideas
- Pair-level precision threshold in the regression test (blocked on issue #1).
- Multi-currency, user authentication (reviewer name is a free-text field), scheduled runs.
