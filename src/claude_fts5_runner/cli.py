"""Command-line interface: ``claude-fts5 {index,search,run}``."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sqlite3
import sys
from typing import Sequence

from . import __version__
from .indexer import (
    DEFAULT_GLOBS,
    Fts5UnavailableError,
    SchemaVersionError,
    configure_utf8_stdio,
    index_directory,
)
from .runner import MODES, RunnerError, run
from .search import QueryError, search

DEFAULT_DB = "claude_fts5.db"
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def _emit_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _add_db_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default=os.environ.get("CLAUDE_FTS5_DB", DEFAULT_DB),
        help="path to the SQLite index (default: $CLAUDE_FTS5_DB or %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-fts5",
        description="Ground Claude Code prompts with a local SQLite FTS5 index.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{index,search,run}")

    p_index = sub.add_parser("index", help="build or incrementally update the index")
    p_index.add_argument("path", help="directory (or single file) of Markdown notes")
    _add_db_option(p_index)
    p_index.add_argument(
        "--glob",
        action="append",
        dest="globs",
        metavar="PATTERN",
        help=f"file name pattern, repeatable (default: {' '.join(DEFAULT_GLOBS)})",
    )
    p_index.add_argument(
        "--no-prune", action="store_true", help="keep index entries for files that no longer exist"
    )
    p_index.add_argument("--json", action="store_true", help="machine-readable output")

    p_search = sub.add_parser("search", help="BM25-ranked search over the index")
    p_search.add_argument("query", help="free-text query (punctuation is safe)")
    _add_db_option(p_search)
    p_search.add_argument("--top-k", type=int, default=5, help="max results (default: %(default)s)")
    p_search.add_argument(
        "--match",
        choices=("all", "any"),
        default="all",
        help="require all terms (AND) or any term (OR) (default: %(default)s)",
    )
    p_search.add_argument("--json", action="store_true", help="machine-readable output")

    p_run = sub.add_parser("run", help="send a prompt to the Claude CLI (direct or grounded)")
    p_run.add_argument("prompt", help="prompt text, or '-' to read it from stdin")
    _add_db_option(p_run)
    p_run.add_argument(
        "--mode",
        choices=MODES,
        default="direct",
        help="direct = Path A pass-through, grounded = Path B FTS5 context (default: %(default)s)",
    )
    p_run.add_argument("--top-k", type=int, default=5, help="grounding hits (default: %(default)s)")
    p_run.add_argument(
        "--dry-run", action="store_true", help="print the assembled prompt and command; do not call claude"
    )
    p_run.add_argument("--json", action="store_true", help="machine-readable output")
    p_run.add_argument("--claude-bin", default="claude", help="Claude CLI executable (default: %(default)s)")
    p_run.add_argument("--model", help="forwarded to the Claude CLI as --model")
    p_run.add_argument(
        "--claude-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra argument for the Claude CLI, repeatable (use --claude-arg=--flag)",
    )
    p_run.add_argument("--timeout", type=float, help="seconds before the Claude CLI call is aborted")
    return parser


# ------------------------------------------------------------------ commands


def _cmd_index(args: argparse.Namespace) -> int:
    stats = index_directory(
        args.db,
        args.path,
        globs=tuple(args.globs) if args.globs else DEFAULT_GLOBS,
        prune=not args.no_prune,
    )
    if args.json:
        _emit_json(stats.as_dict())
    else:
        print(
            f"indexed {args.path} -> {args.db}: "
            f"{stats.added} added, {stats.updated} updated, {stats.unchanged} unchanged, "
            f"{stats.removed} removed ({stats.total_files} files, {stats.total_chunks} chunks)"
        )
    for err in stats.errors:
        print(f"warning: could not read {err}", file=sys.stderr)
    return EXIT_FAILURE if stats.errors else EXIT_OK


def _cmd_search(args: argparse.Namespace) -> int:
    hits = search(args.db, args.query, top_k=args.top_k, match=args.match)
    if args.json:
        _emit_json({"query": args.query, "hits": [h.as_dict() for h in hits]})
        return EXIT_OK
    if not hits:
        print("no results", file=sys.stderr)
        return EXIT_OK
    for h in hits:
        print(f"{h.rank}. {h.path}:{h.start_line} :: {h.heading_path}  (score {h.score:.3f})")
        print(f"   {h.snippet}")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
    result = run(
        prompt,
        mode=args.mode,
        db_path=args.db if args.mode == "grounded" else None,
        top_k=args.top_k,
        dry_run=args.dry_run,
        claude_bin=args.claude_bin,
        model=args.model,
        extra_args=tuple(args.claude_arg),
        timeout=args.timeout,
    )
    plan, inv = result.plan, result.invocation
    if args.json:
        _emit_json(
            {
                "mode": plan.mode,
                "grounded": plan.grounded,
                "dry_run": inv.dry_run,
                "hits": [h.as_dict() for h in plan.hits],
                "prompt": plan.prompt,
                "command": list(inv.command),
                "returncode": inv.returncode,
                "stdout": inv.stdout,
                "stderr": inv.stderr,
            }
        )
        return inv.returncode
    if inv.dry_run:
        print(f"[dry-run] mode={plan.mode} grounded={str(plan.grounded).lower()} hits={len(plan.hits)}")
        print(f"command: {shlex.join(inv.command)}  (prompt on stdin)")
        print("--- prompt ---")
        print(plan.prompt)
        return EXIT_OK
    sys.stdout.write(inv.stdout)
    if inv.stderr:
        sys.stderr.write(inv.stderr)
    return inv.returncode


_COMMANDS = {"index": _cmd_index, "search": _cmd_search, "run": _cmd_run}


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (
        RunnerError,
        QueryError,
        SchemaVersionError,
        Fts5UnavailableError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except sqlite3.Error as exc:
        print(f"error: database failure: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
