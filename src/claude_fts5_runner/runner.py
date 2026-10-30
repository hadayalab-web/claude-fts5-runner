"""Dual-path prompt runner.

Path A (``direct``)   - the prompt is passed to Claude Code unchanged.
Path B (``grounded``) - the prompt is used as an FTS5 query; the top hits are
                        prepended as a ``<FTS5_KNOWLEDGE_GROUNDING>`` block.

The prompt is sent to ``claude -p`` on **stdin** as UTF-8 bytes, which avoids
Windows command-line length limits, shell quoting, and code-page issues.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .search import SearchHit, format_grounding_block, search

MODES = ("direct", "grounded")
MAX_QUERY_TERMS = 24  # cap on terms taken from a long prompt when retrieving


class RunnerError(RuntimeError):
    """Invalid runner input or an unusable Claude CLI."""


@dataclass(frozen=True)
class PromptPlan:
    mode: str
    original_prompt: str
    prompt: str  # what will actually be sent to the model
    hits: tuple[SearchHit, ...]
    grounded: bool  # True only if a grounding block was actually prepended


@dataclass(frozen=True)
class InvocationResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    dry_run: bool


@dataclass(frozen=True)
class RunResult:
    plan: PromptPlan
    invocation: InvocationResult


def build_direct_prompt(prompt: str) -> str:
    """Path A: pass the prompt through untouched."""
    return prompt


def build_grounded_prompt(
    prompt: str, hits: Sequence[SearchHit], *, max_chars: int | None = 2000
) -> str:
    """Path B: ``<grounding block>`` + blank line + the original prompt.

    With no hits the prompt is returned unchanged (nothing to ground on).
    """
    block = format_grounding_block(hits, max_chars=max_chars)
    return f"{block}\n\n{prompt}" if block else prompt


def plan_prompt(
    prompt: str,
    *,
    mode: str = "direct",
    db_path: str | os.PathLike[str] | None = None,
    top_k: int = 5,
    max_chars: int | None = 2000,
) -> PromptPlan:
    """Assemble the final prompt for ``mode`` (does not call Claude)."""
    if mode not in MODES:
        raise RunnerError(f"unknown mode {mode!r}; expected one of {MODES}")
    if not prompt or not prompt.strip():
        raise RunnerError("prompt is empty")

    if mode == "direct":
        return PromptPlan(mode, prompt, build_direct_prompt(prompt), (), False)

    if db_path is None:
        raise RunnerError("grounded mode requires a database path (--db)")
    if not Path(db_path).is_file():
        raise RunnerError(f"index database not found: {db_path} (run `claude-fts5 index` first)")
    hits = tuple(search(db_path, prompt, top_k=top_k, match="any", max_terms=MAX_QUERY_TERMS))
    final = build_grounded_prompt(prompt, hits, max_chars=max_chars)
    return PromptPlan(mode, prompt, final, hits, bool(hits))


def build_command(
    claude_bin: str = "claude",
    *,
    model: str | None = None,
    extra_args: Sequence[str] = (),
    base_args: Sequence[str] = ("-p",),
) -> list[str]:
    """Build the argv for a non-interactive (print-mode) Claude Code call."""
    cmd = [claude_bin, *base_args]
    if model:
        cmd += ["--model", model]
    cmd += list(extra_args)
    return cmd


def invoke_claude_cli(
    prompt: str,
    *,
    claude_bin: str = "claude",
    model: str | None = None,
    extra_args: Sequence[str] = (),
    base_args: Sequence[str] = ("-p",),
    dry_run: bool = False,
    timeout: float | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> InvocationResult:
    """Run ``claude -p`` with ``prompt`` on stdin (UTF-8 in, UTF-8 out).

    With ``dry_run`` nothing is executed and the binary need not exist; the
    would-be command is returned with ``returncode=0``.
    """
    cmd = build_command(claude_bin, model=model, extra_args=extra_args, base_args=base_args)
    if dry_run:
        return InvocationResult(tuple(cmd), 0, "", "", True)

    resolved = shutil.which(claude_bin)
    if resolved is None:
        raise RunnerError(
            f"Claude CLI not found: {claude_bin!r}. Install Claude Code or pass --claude-bin."
        )
    cmd[0] = resolved
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(f"Claude CLI timed out after {timeout}s") from exc
    except OSError as exc:
        raise RunnerError(f"could not start {resolved!r}: {exc}") from exc
    return InvocationResult(
        tuple(cmd),
        proc.returncode,
        proc.stdout.decode("utf-8", errors="replace"),
        proc.stderr.decode("utf-8", errors="replace"),
        False,
    )


def run(
    prompt: str,
    *,
    mode: str = "direct",
    db_path: str | os.PathLike[str] | None = None,
    top_k: int = 5,
    dry_run: bool = False,
    claude_bin: str = "claude",
    model: str | None = None,
    extra_args: Sequence[str] = (),
    timeout: float | None = None,
    max_chars: int | None = 2000,
) -> RunResult:
    """Plan the prompt for ``mode`` and invoke the Claude CLI with it."""
    plan = plan_prompt(prompt, mode=mode, db_path=db_path, top_k=top_k, max_chars=max_chars)
    invocation = invoke_claude_cli(
        plan.prompt,
        claude_bin=claude_bin,
        model=model,
        extra_args=extra_args,
        dry_run=dry_run,
        timeout=timeout,
    )
    return RunResult(plan, invocation)
