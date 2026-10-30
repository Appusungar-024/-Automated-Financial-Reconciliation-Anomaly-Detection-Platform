-- Analytics views. Plain ANSI-style SQL so they run unchanged on PostgreSQL and SQLite.
-- Each view is (re)created on init_db(); the dashboard and audit report read from these
-- rather than re-deriving the numbers in Python, so the SQL is the single source of truth.
-- (Statements are separated by semicolons; keep semicolons out of comments.)

DROP VIEW IF EXISTS v_exception_summary;
CREATE VIEW v_exception_summary AS
-- Exceptions by type and source: how many, how many still open, and the gross value at stake.
SELECT run_id,
       exception_type,
       source_system,
       COUNT(*)                                                                          AS exception_count,
       SUM(CASE WHEN status IN ('OPEN', 'IN_REVIEW', 'ESCALATED') THEN 1 ELSE 0 END)      AS unresolved_count,
       SUM(amount)                                                                        AS total_amount,
       SUM(CASE WHEN status IN ('OPEN', 'IN_REVIEW', 'ESCALATED') THEN amount ELSE 0 END) AS unresolved_amount
FROM exceptions
GROUP BY run_id, exception_type, source_system;

DROP VIEW IF EXISTS v_exception_aging;
CREATE VIEW v_exception_aging AS
-- Ageing of unresolved exceptions, bucketed by days outstanding.
SELECT run_id,
       aging_bucket,
       MIN(bucket_order)  AS bucket_order,
       COUNT(*)           AS exception_count,
       SUM(amount)        AS unresolved_amount
FROM (
    SELECT run_id,
           amount,
           CASE WHEN age_days <= 7  THEN '0-7 days'
                WHEN age_days <= 14 THEN '8-14 days'
                WHEN age_days <= 30 THEN '15-30 days'
                WHEN age_days <= 60 THEN '31-60 days'
                ELSE '60+ days' END AS aging_bucket,
           CASE WHEN age_days <= 7  THEN 1
                WHEN age_days <= 14 THEN 2
                WHEN age_days <= 30 THEN 3
                WHEN age_days <= 60 THEN 4
                ELSE 5 END          AS bucket_order
    FROM exceptions
    WHERE status IN ('OPEN', 'IN_REVIEW', 'ESCALATED')
) aged
GROUP BY run_id, aging_bucket;

DROP VIEW IF EXISTS v_review_queue;
CREATE VIEW v_review_queue AS
-- The analyst's worklist: unresolved exceptions ranked by priority within each run.
SELECT e.*,
       ROW_NUMBER() OVER (PARTITION BY e.run_id
                          ORDER BY e.priority_score DESC, e.amount DESC, e.exception_id) AS queue_rank
FROM exceptions e
WHERE e.status IN ('OPEN', 'IN_REVIEW', 'ESCALATED');

DROP VIEW IF EXISTS v_recon_by_pair;
CREATE VIEW v_recon_by_pair AS
-- Matches per reconciliation pair and cascade stage, with average confidence and value.
SELECT run_id,
       pair_name,
       match_type,
       COUNT(*)                                              AS matches,
       AVG(match_confidence)                                 AS avg_confidence,
       SUM(amount_a)                                         AS matched_value,
       SUM(CASE WHEN reference_mismatch THEN 1 ELSE 0 END)   AS reference_mismatches
FROM matched_records
GROUP BY run_id, pair_name, match_type;

DROP VIEW IF EXISTS v_source_reconciliation;
CREATE VIEW v_source_reconciliation AS
-- Reconciliation rate per source system, by count and by value. Live: a manual match made
-- after the run is reflected immediately because it adds a row to matched_records.
WITH matched AS (
    SELECT run_id, row_id_a AS row_id FROM matched_records
    UNION ALL
    SELECT run_id, row_id_b AS row_id FROM matched_records
)
SELECT r.run_id,
       r.source_system,
       COUNT(*)                                                          AS total_rows,
       COUNT(m.row_id)                                                   AS matched_rows,
       SUM(r.amount)                                                     AS total_value,
       SUM(CASE WHEN m.row_id IS NOT NULL THEN r.amount ELSE 0 END)      AS matched_value
FROM raw_transactions r
LEFT JOIN matched m ON m.run_id = r.run_id AND m.row_id = r.row_id
GROUP BY r.run_id, r.source_system;

DROP VIEW IF EXISTS v_reviewer_activity;
CREATE VIEW v_reviewer_activity AS
-- Who did what, and when, according to the append-only audit log.
SELECT actor,
       event_type,
       COUNT(*)          AS events,
       MIN(event_time)   AS first_event,
       MAX(event_time)   AS last_event
FROM audit_log
GROUP BY actor, event_type;

DROP VIEW IF EXISTS v_data_quality;
CREATE VIEW v_data_quality AS
-- Validation failures by source, code and severity.
SELECT run_id,
       source_system,
       code,
       severity,
       COUNT(*)                                            AS failures,
       SUM(CASE WHEN row_excluded THEN 1 ELSE 0 END)       AS rows_excluded
FROM validation_failures
GROUP BY run_id, source_system, code, severity;

DROP VIEW IF EXISTS v_run_trend;
CREATE VIEW v_run_trend AS
-- One line per completed run: headline rates plus what is still unresolved.
SELECT r.run_id,
       r.started_at,
       r.total_transactions,
       r.matched_count,
       r.match_rate,
       r.exception_count,
       COALESCE(u.unresolved_count, 0)   AS unresolved_count,
       COALESCE(u.unresolved_amount, 0)  AS unresolved_amount
FROM reconciliation_runs r
LEFT JOIN (
    SELECT run_id, COUNT(*) AS unresolved_count, SUM(amount) AS unresolved_amount
    FROM exceptions
    WHERE status IN ('OPEN', 'IN_REVIEW', 'ESCALATED')
    GROUP BY run_id
) u ON u.run_id = r.run_id
WHERE r.status = 'COMPLETED';
