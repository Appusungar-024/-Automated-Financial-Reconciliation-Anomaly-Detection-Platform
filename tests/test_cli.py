import json
from datetime import date

import pytest

from recon import cli, repository, simulate

END = date(2026, 8, 29)


@pytest.fixture(scope="module")
def sim_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("clisim")
    simulate.main(700, 9, d, end_date=END, excel_sources=("BANK",))
    return d


def test_run_cli_prints_a_readable_summary_and_can_evaluate(sim_dir, capsys):
    assert cli.run_main(["--dir", str(sim_dir), "--no-db", "--evaluate"]) == 0
    out = capsys.readouterr().out
    for expected in ("Files", "bank.xlsx", "Matching cascade", "REFERENCE", "Exceptions by type", "Top of the review queue",
                     "Anomaly detection", "MATCHING & CLASSIFICATION", "INGESTION VALIDATION", "ANOMALY DETECTION"):
        assert expected in out, expected


def test_run_cli_persists_and_attributes_the_run_to_the_actor(tmp_db, sim_dir, capsys):
    assert cli.run_main(["--dir", str(sim_dir), "--actor", "nightly_job", "--notes", "scheduled"]) == 0
    run = repository.list_runs().iloc[0]
    assert run["status"] == "COMPLETED" and run["notes"] == "scheduled"
    assert set(repository.audit_trail(run["run_id"])["actor"]) == {"nightly_job"}


def test_run_cli_accepts_explicit_file_paths(sim_dir, capsys):
    code = cli.run_main(["--bank", str(sim_dir / "bank.xlsx"), "--ledger", str(sim_dir / "ledger.csv"), "--no-db"])
    assert code == 0 and "BANK_VS_LEDGER" in capsys.readouterr().out


def test_run_cli_fails_cleanly_with_exit_code_2(tmp_path, capsys):
    assert cli.run_main(["--dir", str(tmp_path), "--no-db"]) == 2
    assert "Run failed" in capsys.readouterr().err


def test_evaluate_cli_writes_metrics_json(sim_dir, tmp_path, capsys):
    out = tmp_path / "metrics.json"
    assert cli.evaluate_main(["--dir", str(sim_dir), "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert {"matching", "ingestion", "anomaly", "headline"} <= set(data)
    assert data["ingestion"]["detection_rate"] == 100.0
