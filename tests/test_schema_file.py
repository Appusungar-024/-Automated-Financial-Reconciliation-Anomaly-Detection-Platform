"""db/schema.sql is hand-written for readers; this keeps it honest against the application's own schema.
Needs a real PostgreSQL:  TEST_DATABASE_URL=postgresql+psycopg2://user:pw@localhost/db pytest"""
import os

import pytest
from sqlalchemy import text

from recon import config, db

pytestmark = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
                                reason="needs TEST_DATABASE_URL pointing at PostgreSQL")

QUERY = """
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name IN :tables
"""


def snapshot(conn):
    tables = tuple(db.metadata.tables)
    rows = conn.execute(text(QUERY.replace(":tables", "(" + ",".join(f"'{t}'" for t in tables) + ")")))
    return {tuple(r) for r in rows}


def test_schema_sql_builds_the_same_tables_as_the_application():
    db.configure(os.environ["TEST_DATABASE_URL"])
    db.drop_all()
    db.init_db()
    with db.get_engine().connect() as conn:
        app_schema = snapshot(conn)
    db.drop_all()
    with db.get_engine().begin() as conn:
        conn.exec_driver_sql((config.BASE_DIR / "db" / "schema.sql").read_text(encoding="utf-8"))
    with db.get_engine().connect() as conn:
        file_schema = snapshot(conn)
    assert app_schema and file_schema == app_schema, (app_schema ^ file_schema)

    # and the hand-written audit trigger really does block tampering
    with db.get_engine().begin() as conn:
        conn.exec_driver_sql("INSERT INTO audit_log (event_time, event_type, actor) VALUES (now(), 'X', 'a')")
    with pytest.raises(Exception, match="append-only"):
        with db.get_engine().begin() as conn:
            conn.exec_driver_sql("DELETE FROM audit_log")
    db.drop_all()
    db.init_db()           # leave the database usable for whatever runs next
    db.get_engine().dispose()
    db._engine = None
