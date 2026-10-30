-- Financial Reconciliation Platform -- PostgreSQL schema (v2)
--
--   psql -U recon_user -d recon_platform -f db/schema.sql      -- tables, constraints, audit triggers
--   psql -U recon_user -d recon_platform -f db/analytics.sql   -- analytics views
--
-- The application creates exactly this schema on its own (recon/db.py via SQLAlchemy), so running
-- these files by hand is optional. They are here because a reviewer should be able to read the SQL.
-- tests/test_schema_file.py applies this file to a real PostgreSQL and diffs it against the
-- application's own schema, so the two cannot drift apart.

CREATE TABLE IF NOT EXISTS reconciliation_runs (
    run_id              VARCHAR(40)   PRIMARY KEY,
    started_at          TIMESTAMP     NOT NULL,
    finished_at         TIMESTAMP,
    status              VARCHAR(12)   NOT NULL DEFAULT 'RUNNING',
    as_of_date          DATE,                              -- reference date used for ageing exceptions
    total_transactions  INTEGER,
    matched_count       INTEGER,
    exception_count     INTEGER,
    match_rate          NUMERIC(7,2),
    total_value         NUMERIC(18,2),
    matched_value       NUMERIC(18,2),
    runtime_seconds     NUMERIC(7,2),
    notes               TEXT,
    config_snapshot     TEXT,                              -- JSON: every threshold in force for this run
    error_message       TEXT,
    CONSTRAINT ck_runs_status CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED'))
);

-- Provenance: exactly which files a run consumed (SHA-256 proves it).
CREATE TABLE IF NOT EXISTS source_files (
    file_id         SERIAL        PRIMARY KEY,
    run_id          VARCHAR(40)   NOT NULL REFERENCES reconciliation_runs (run_id),
    source_system   VARCHAR(32)   NOT NULL,                -- BANK / LEDGER / PAYMENT_GATEWAY / ERP
    file_name       VARCHAR(255)  NOT NULL,
    sha256          VARCHAR(64),
    sheet_name      VARCHAR(255),
    header_row      INTEGER,
    column_mapping  TEXT,                                  -- JSON: canonical field -> header found in the file
    rows_read       INTEGER,
    rows_accepted   INTEGER,
    rows_excluded   INTEGER,
    ingested_at     TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_source_files_run ON source_files (run_id);
CREATE INDEX IF NOT EXISTS ix_source_files_sha ON source_files (sha256);

-- Every row ingestion refused or flagged, and why. Nothing is silently coerced.
CREATE TABLE IF NOT EXISTS validation_failures (
    failure_id     SERIAL        PRIMARY KEY,
    run_id         VARCHAR(40)   NOT NULL REFERENCES reconciliation_runs (run_id),
    source_system  VARCHAR(32),
    file_name      VARCHAR(255),
    source_row     INTEGER,                                -- NULL for file-level failures
    row_id         VARCHAR(40),
    field          VARCHAR(64),
    code           VARCHAR(40)   NOT NULL,                 -- e.g. UNPARSEABLE_DATE, SUMMARY_ROW
    severity       VARCHAR(10)   NOT NULL,
    row_excluded   BOOLEAN       NOT NULL,
    raw_value      TEXT,
    message        TEXT,
    created_at     TIMESTAMP,
    CONSTRAINT ck_failures_severity CHECK (severity IN ('ERROR', 'WARNING', 'INFO'))
);
CREATE INDEX IF NOT EXISTS ix_failures_run ON validation_failures (run_id, code);

-- Keyed by (run_id, row_id): the same file can be reconciled in many runs.
CREATE TABLE IF NOT EXISTS raw_transactions (
    run_id             VARCHAR(40)    NOT NULL REFERENCES reconciliation_runs (run_id),
    row_id             VARCHAR(40)    NOT NULL,            -- e.g. BANK-0000123: position in the source file
    source_system      VARCHAR(32)    NOT NULL,
    source_row         INTEGER,
    txn_date           DATE           NOT NULL,
    vendor_raw         TEXT,
    vendor_clean       VARCHAR(255),
    reference          VARCHAR(128),
    amount             NUMERIC(18,2)  NOT NULL,
    direction          VARCHAR(8),
    is_duplicate       BOOLEAN,
    rule_score         NUMERIC(6,1),
    ml_score           NUMERIC(6,1),
    risk_score         NUMERIC(6,1),
    is_suspicious      BOOLEAN,
    suspicion_reasons  TEXT,
    ingested_at        TIMESTAMP,
    PRIMARY KEY (run_id, row_id)
);
CREATE INDEX IF NOT EXISTS ix_raw_run_source ON raw_transactions (run_id, source_system);
CREATE INDEX IF NOT EXISTS ix_raw_run_risk   ON raw_transactions (run_id, risk_score);

-- One row per matched pair, with the reasoning behind the decision.
CREATE TABLE IF NOT EXISTS matched_records (
    match_id            SERIAL         PRIMARY KEY,
    run_id              VARCHAR(40)    NOT NULL,
    pair_name           VARCHAR(64),
    row_id_a            VARCHAR(40)    NOT NULL,
    row_id_b            VARCHAR(40)    NOT NULL,
    source_a            VARCHAR(32),
    source_b            VARCHAR(32),
    date_a              DATE,
    date_b              DATE,
    amount_a            NUMERIC(18,2),
    amount_b            NUMERIC(18,2),
    vendor_a            TEXT,
    vendor_b            TEXT,
    reference_a         VARCHAR(128),
    reference_b         VARCHAR(128),
    match_type          VARCHAR(32)    NOT NULL,           -- REFERENCE / EXACT / DATE_AMOUNT / AMOUNT_TOLERANCE / FUZZY_VENDOR / MANUAL
    match_confidence    NUMERIC(6,1),
    reference_mismatch  BOOLEAN,
    amount_diff         NUMERIC(18,2),                     -- decision rationale
    date_diff_days      INTEGER,
    vendor_similarity   NUMERIC(6,1),
    rule_applied        TEXT,
    decided_by          VARCHAR(128),                      -- 'system' or the reviewer who matched it by hand
    matched_at          TIMESTAMP,
    FOREIGN KEY (run_id, row_id_a) REFERENCES raw_transactions (run_id, row_id),
    FOREIGN KEY (run_id, row_id_b) REFERENCES raw_transactions (run_id, row_id)
);
CREATE INDEX IF NOT EXISTS ix_matches_run_a    ON matched_records (run_id, row_id_a);
CREATE INDEX IF NOT EXISTS ix_matches_run_b    ON matched_records (run_id, row_id_b);
CREATE INDEX IF NOT EXISTS ix_matches_run_pair ON matched_records (run_id, pair_name, match_type);

CREATE TABLE IF NOT EXISTS exceptions (
    exception_id                SERIAL         PRIMARY KEY,
    run_id                      VARCHAR(40)    NOT NULL,
    pair_name                   VARCHAR(128),
    row_id                      VARCHAR(40)    NOT NULL,
    source_system               VARCHAR(32),
    txn_date                    DATE,
    vendor_raw                  TEXT,
    reference                   VARCHAR(128),
    amount                      NUMERIC(18,2),
    direction                   VARCHAR(8),
    exception_type              VARCHAR(32)    NOT NULL,   -- MISSING / DUPLICATE / AMOUNT_MISMATCH / DATE_MISMATCH / PROBABLE_MATCH
    suggested_match_row_id      VARCHAR(40),
    suggested_match_confidence  NUMERIC(6,1),
    suggested_match_reason      TEXT,
    risk_score                  NUMERIC(6,1),
    age_days                    INTEGER,
    priority_score              NUMERIC(6,1),
    priority_band               VARCHAR(10),               -- CRITICAL / HIGH / MEDIUM / LOW
    -- review workflow
    status                      VARCHAR(12)    NOT NULL DEFAULT 'OPEN',
    assigned_to                 VARCHAR(128),
    resolution_code             VARCHAR(32),
    resolved_by                 VARCHAR(128),
    resolved_at                 TIMESTAMP,
    resolution_notes            TEXT,
    created_at                  TIMESTAMP,
    updated_at                  TIMESTAMP,
    FOREIGN KEY (run_id, row_id) REFERENCES raw_transactions (run_id, row_id),
    CONSTRAINT uq_exceptions_run_row UNIQUE (run_id, row_id),
    CONSTRAINT ck_exceptions_status CHECK (status IN ('OPEN', 'IN_REVIEW', 'ESCALATED', 'RESOLVED', 'DISMISSED'))
);
CREATE INDEX IF NOT EXISTS ix_exc_run_status   ON exceptions (run_id, status);
CREATE INDEX IF NOT EXISTS ix_exc_run_priority ON exceptions (run_id, priority_score);

-- Append-only record of everything that happened: runs, ingestion, matching summaries, anomaly
-- model fits, and every reviewer action (who, what, from, to, why).
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id     SERIAL        PRIMARY KEY,
    event_time   TIMESTAMP     NOT NULL,
    run_id       VARCHAR(40),
    event_type   VARCHAR(40)   NOT NULL,
    actor        VARCHAR(128)  NOT NULL,
    entity_type  VARCHAR(32),
    entity_id    VARCHAR(64),
    old_value    TEXT,
    new_value    TEXT,
    details      TEXT                                      -- JSON
);
CREATE INDEX IF NOT EXISTS ix_audit_run    ON audit_log (run_id);
CREATE INDEX IF NOT EXISTS ix_audit_entity ON audit_log (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS ix_audit_time   ON audit_log (event_time);

-- The audit log is append-only, enforced by the database rather than by application convention.
CREATE OR REPLACE FUNCTION audit_log_block_change() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only: UPDATE, DELETE and TRUNCATE are not permitted';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_log_no_update_delete ON audit_log;
CREATE TRIGGER audit_log_no_update_delete BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_block_change();

DROP TRIGGER IF EXISTS audit_log_no_truncate ON audit_log;
CREATE TRIGGER audit_log_no_truncate BEFORE TRUNCATE ON audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION audit_log_block_change();
