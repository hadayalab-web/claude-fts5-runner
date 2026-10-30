"""Runner (Path A / Path B) and CLI tests."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from claude_fts5_runner import __version__, runner
from claude_fts5_runner.cli import main
from claude_fts5_runner.indexer import index_directory
from claude_fts5_runner.runner import (
    RunnerError,
    build_command,
    invoke_claude_cli,
    plan_prompt,
    run,
)

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture()
def db(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "winerror.md").write_text(
        "# WinError 10061\n\nConnection refused on localhost: start the server before the client.\n",
        encoding="utf-8",
    )
    (notes / "cooking.md").write_text("# Pasta\n\nBoil water and add salt.\n", encoding="utf-8")
    (notes / "jp.md").write_text("# 設定\n\nタイムアウト の原因と対策。\n", encoding="utf-8")
    path = tmp_path / "idx.db"
    index_directory(path, notes)
    return path, notes


# ------------------------------------------------------------- prompt paths


def test_path_a_direct_passes_prompt_through_without_db():
    plan = plan_prompt("Refactor utils.py", mode="direct")
    assert plan.prompt == "Refactor utils.py"
    assert plan.grounded is False and plan.hits == ()


def test_path_b_grounded_prepends_block_and_keeps_prompt_verbatim(db):
    path, _ = db
    prompt = "Why do I get WinError 10061 connection refused?"
    plan = plan_prompt(prompt, mode="grounded", db_path=path, top_k=2)
    assert plan.grounded is True
    assert plan.hits[0].path == "winerror.md"
    assert plan.prompt.startswith("<FTS5_KNOWLEDGE_GROUNDING")
    assert plan.prompt.endswith("</FTS5_KNOWLEDGE_GROUNDING>\n\n" + prompt)
    assert "start the server before the client" in plan.prompt


def test_path_b_with_no_hits_falls_back_to_unchanged_prompt(db):
    path, _ = db
    plan = plan_prompt("zzzqqq xxyyzz", mode="grounded", db_path=path)
    assert plan.prompt == "zzzqqq xxyyzz" and plan.grounded is False


def test_path_b_handles_punctuation_heavy_prompts(db):
    path, _ = db
    plan = plan_prompt("Error: ERR-404 -- (see: foo/bar) NOT working?!", mode="grounded", db_path=path)
    assert plan.mode == "grounded"


def test_plan_prompt_validation(tmp_path):
    with pytest.raises(RunnerError, match="unknown mode"):
        plan_prompt("x", mode="turbo")
    with pytest.raises(RunnerError, match="empty"):
        plan_prompt("   ", mode="direct")
    with pytest.raises(RunnerError, match="requires a database"):
        plan_prompt("x", mode="grounded")
    with pytest.raises(RunnerError, match="not found"):
        plan_prompt("x", mode="grounded", db_path=tmp_path / "nope.db")


# --------------------------------------------------------- invoke_claude_cli


def test_dry_run_never_spawns_a_process(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("subprocess must not run in dry-run mode")

    monkeypatch.setattr(runner.subprocess, "run", boom)
    result = invoke_claude_cli("hi", claude_bin="definitely-not-installed", dry_run=True, model="m")
    assert result.dry_run is True and result.returncode == 0
    assert result.command == ("definitely-not-installed", "-p", "--model", "m")


def test_build_command_extra_args_order():
    assert build_command("claude", extra_args=["--max-turns=2"]) == ["claude", "-p", "--max-turns=2"]


def test_missing_binary_raises_runner_error():
    with pytest.raises(RunnerError, match="not found"):
        invoke_claude_cli("hi", claude_bin="definitely-not-installed-xyz")


def test_prompt_sent_on_stdin_as_utf8_bytes(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"], captured["kwargs"] = cmd, kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="応答 ok".encode("utf-8"), stderr=b"")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/bin/claude")
    result = invoke_claude_cli("日本語 prompt ✓")
    assert captured["kwargs"]["input"] == "日本語 prompt ✓".encode("utf-8")
    assert captured["cmd"] == ["/bin/claude", "-p"]
    assert result.stdout == "応答 ok" and result.returncode == 0


def test_real_subprocess_utf8_round_trip():
    echo = ("-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())")
    text = "grounded → 日本語 café ✓"
    result = invoke_claude_cli(text, claude_bin=sys.executable, base_args=echo)
    assert result.returncode == 0 and result.stdout == text


def test_timeout_is_reported_as_runner_error():
    sleeper = ("-c", "import time; time.sleep(5)")
    with pytest.raises(RunnerError, match="timed out"):
        invoke_claude_cli("x", claude_bin=sys.executable, base_args=sleeper, timeout=0.3)


def test_run_end_to_end_grounded_with_fake_cli(db):
    path, _ = db
    echo = ("-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())")
    plan = plan_prompt("WinError 10061 refused", mode="grounded", db_path=path)
    inv = invoke_claude_cli(plan.prompt, claude_bin=sys.executable, base_args=echo)
    assert inv.stdout == plan.prompt
    assert run("p", mode="direct", dry_run=True).invocation.dry_run is True


# ----------------------------------------------------------------------- CLI


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_index_then_reindex_reports_unchanged(tmp_path, capsys):
    notes = tmp_path / "n"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nalpha\n", encoding="utf-8")
    db_path = str(tmp_path / "i.db")

    assert main(["index", str(notes), "--db", db_path, "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["added"] == 1 and first["total_chunks"] == 1

    assert main(["index", str(notes), "--db", db_path]) == 0
    assert "1 unchanged" in capsys.readouterr().out


def test_cli_search_json_and_text_with_hyphen_colon_query(db, capsys):
    path, _ = db
    assert main(["search", "WinError-10061:", "--db", str(path), "--json", "--top-k", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"][0]["path"] == "winerror.md"

    assert main(["search", "タイムアウト", "--db", str(path)]) == 0
    out = capsys.readouterr().out
    assert "jp.md" in out and "設定" in out

    assert main(["search", "nonexistentterm", "--db", str(path)]) == 0
    assert "no results" in capsys.readouterr().err


def test_cli_run_dry_run_direct_and_grounded(db, capsys):
    path, _ = db
    assert main(["run", "Fix my script", "--dry-run", "--json"]) == 0
    direct = json.loads(capsys.readouterr().out)
    assert direct["prompt"] == "Fix my script" and direct["grounded"] is False
    assert direct["command"] == ["claude", "-p"] and direct["dry_run"] is True

    argv = ["run", "WinError 10061 connection refused", "--mode", "grounded", "--db", str(path)]
    assert main([*argv, "--top-k", "1", "--dry-run", "--json"]) == 0
    grounded = json.loads(capsys.readouterr().out)
    assert grounded["grounded"] is True and len(grounded["hits"]) == 1
    assert grounded["prompt"].startswith("<FTS5_KNOWLEDGE_GROUNDING")

    assert main([*argv, "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "[dry-run] mode=grounded grounded=true hits=" in text and "--- prompt ---" in text


def test_nonzero_exit_code_and_stderr_are_propagated():
    script = "import sys; sys.stdin.buffer.read(); sys.stderr.write('boom'); sys.exit(3)"
    inv = invoke_claude_cli("hello", claude_bin=sys.executable, base_args=("-c", script))
    assert inv.returncode == 3 and inv.stderr == "boom" and inv.dry_run is False


def test_cli_run_reads_prompt_from_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("prompt from stdin"))
    assert main(["run", "-", "--dry-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["prompt"] == "prompt from stdin"


def test_cli_errors_return_exit_code_2(tmp_path, capsys):
    missing = str(tmp_path / "none.db")
    assert main(["run", "x", "--mode", "grounded", "--db", missing, "--dry-run"]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["search", "x", "--db", missing]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["run", "x", "--claude-bin", "definitely-not-installed-xyz"]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["index", str(tmp_path / "no_such_dir"), "--db", str(tmp_path / "i.db")]) == 2
    capsys.readouterr()


def test_module_entrypoint_emits_utf8_even_with_legacy_stdio_encoding(db):
    path, _ = db
    env = {
        **os.environ,
        "PYTHONPATH": SRC,
        "PYTHONUTF8": "0",
        "PYTHONIOENCODING": "cp1252",  # simulate a legacy Windows console code page
    }
    proc = subprocess.run(
        [sys.executable, "-m", "claude_fts5_runner", "search", "タイムアウト", "--db", str(path), "--json"],
        capture_output=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    payload = json.loads(proc.stdout.decode("utf-8"))
    assert payload["hits"][0]["heading"] == "設定"
