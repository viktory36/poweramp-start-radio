import json
from pathlib import Path

from click.testing import CliRunner

from poweramp_indexer.cli import cli
from poweramp_indexer.server_indexer import server_instance_lock


def test_server_init_and_status_cli_do_not_load_models(tmp_path: Path):
    listen = tmp_path / "listen"
    runtime = tmp_path / "runtime"
    listen.mkdir()
    runtime.mkdir()
    config = tmp_path / "server.toml"
    config.write_text(
        f"""
[server]
state_db = "{runtime / 'state.db'}"
bundle_db = "{runtime / 'bundle.db'}"
cache_dir = "{runtime / 'cache'}"
settle_seconds = 0

[[listen_roots]]
id = "test"
path = "{listen}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    initialized = runner.invoke(
        cli, ["server", "init", "--config", str(config), "--baseline-existing"]
    )
    assert initialized.exit_code == 0, initialized.output
    assert "Baseline recorded: 0 existing" in initialized.output

    with server_instance_lock(runtime / "state.db"):
        status = runner.invoke(cli, ["server", "status", "--config", str(config)])
        json_status = runner.invoke(
            cli,
            ["server", "status", "--config", str(config), "--json"],
        )
    assert status.exit_code == 0, status.output
    assert "Poweramp server indexer" in status.output
    assert "Process: running" in status.output
    assert f"State: {runtime / 'state.db'}" in status.output
    assert f"test: {listen} (baseline recorded)" in status.output
    assert "Library: 0 present" in status.output
    assert "Queue: empty" in status.output
    assert f"Export: {runtime / 'bundle.db'}" in status.output
    assert "Speed: no completed non-reused timing samples yet" in status.output

    assert json_status.exit_code == 0, json_status.output
    document = json.loads(json_status.output)
    assert document["writer_active"] is True
    assert document["state"] == {
        "path": str(runtime / "state.db"),
        "schema_version": 3,
    }
    assert document["roots"][0]["baselined"] is True
    assert document["library"]["present"] == 0
    assert document["bundle"]["path"] == str(runtime / "bundle.db")
    assert document["timing"]["median_seconds_per_track"] is None
