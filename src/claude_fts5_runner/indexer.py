"""SQLite FTS5 indexer: Markdown section chunking with incremental updates.

Design notes
------------
* One row per Markdown *section* (split on ATX headings, fenced code aware).
* ``files`` stores the sha256 of the raw file bytes; a file whose hash is
  unchanged is skipped entirely, so re-indexing a large notes folder is cheap.
* ``chunks_fts`` is an *external-content* FTS5 table (``content='chunks'``)
  kept in sync by triggers, so text is stored once.
* Tokenizer is ``porter unicode61`` (Unicode-aware, English stemming).
* Paths are stored as POSIX-style paths relative to the indexed root. Use one
  database per corpus root.
* All file reads are byte-level + explicit UTF-8 decoding, so the result never
  depends on the Windows ANSI code page.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

SCHEMA_VERSION = 1
TOKENIZER = "porter unicode61"
DEFAULT_GLOBS: tuple[str, ...] = ("*.md", "*.markdown")
DEFAULT_MAX_CHARS = 4000
SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache"}
)

_DDL = f"""
CREATE TABLE files (
    path       TEXT PRIMARY KEY,
    sha256     TEXT NOT NULL,
    n_chunks   INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE chunks (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    heading      TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    level        INTEGER NOT NULL,
    start_line   INTEGER NOT NULL,
    body         TEXT NOT NULL
);
CREATE INDEX idx_chunks_path ON chunks(path);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    heading, body,
    content='chunks', content_rowid='id',
    tokenize='{TOKENIZER}'
);
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, heading, body) VALUES (new.id, new.heading, new.body);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading, body)
    VALUES ('delete', old.id, old.heading, old.body);
END;
PRAGMA user_version = {SCHEMA_VERSION};
"""


class SchemaVersionError(RuntimeError):
    """The database is not a claude-fts5 index, or has an unsupported schema version."""


class Fts5UnavailableError(RuntimeError):
    """The running SQLite build does not include the FTS5 extension."""


# --------------------------------------------------------------------------- I/O


def configure_utf8_stdio() -> None:
    """Force UTF-8 on stdin/stdout/stderr (Windows consoles default to a legacy code page)."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if encoding == "utf8" or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # stream already in use / closed
            pass


def read_file(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Read ``path`` as UTF-8 and return ``(text, sha256_hex_of_raw_bytes)``.

    A UTF-8 BOM is stripped, undecodable bytes become U+FFFD, and newlines are
    normalised to ``\\n``. The hash covers the raw bytes, so any edit is detected.
    """
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    text = data.decode("utf-8-sig", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n"), digest


# ------------------------------------------------------------------- chunking


@dataclass(frozen=True)
class Chunk:
    source: str  # POSIX path relative to the indexed root
    heading: str  # nearest heading title
    heading_path: str  # "Parent > Child" breadcrumb
    level: int  # 1-6 for headings, 0 for text before the first heading
    ordinal: int  # 0-based position within the file
    start_line: int  # 1-based line of the heading (or first line for preamble)
    text: str


_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _next_fence(line: str, fence: tuple[str, int] | None) -> tuple[str, int] | None:
    """Advance the fenced-code-block state machine by one line."""
    m = _FENCE_RE.match(line)
    if fence is None:
        if m is None:
            return None
        marker, info = m.group(1), m.group(2)
        if marker[0] == "`" and "`" in info:  # inline code span, not a fence
            return None
        return (marker[0], len(marker))
    if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1] and not m.group(2).strip():
        return None
    return fence


def _pack(units: Sequence[str], sep: str, max_chars: int) -> list[str]:
    """Greedily join ``units`` with ``sep`` into pieces of at most ``max_chars``."""
    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else current + sep + unit
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_long(body: str, max_chars: int) -> list[str]:
    """Split an over-long section on paragraph, then line, boundaries."""
    if len(body) <= max_chars:
        return [body]
    out: list[str] = []
    for para in _pack(body.split("\n\n"), "\n\n", max_chars):
        if len(para) <= max_chars:
            out.append(para)
        else:  # a single huge paragraph: fall back to line packing
            out.extend(_pack(para.split("\n"), "\n", max_chars))
    return out


def chunk_markdown(
    text: str,
    source: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    default_title: str | None = None,
) -> list[Chunk]:
    """Split Markdown ``text`` into one :class:`Chunk` per heading section.

    * ATX headings (``#`` .. ``######``) start a new section; ``#`` lines inside
      fenced code blocks are ignored.
    * Text before the first heading is kept under ``default_title`` (default:
      the source path).
    * Sections with no body text are dropped.
    * Sections longer than ``max_chars`` are split on paragraph/line boundaries;
      the pieces share the same heading. A single line longer than
      ``max_chars`` is kept whole.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    preamble_title = default_title or source

    sections: list[tuple[str, str, int, int, list[str]]] = []  # title, path, level, line, body
    stack: list[tuple[int, str]] = []
    title, path, level, start = preamble_title, preamble_title, 0, 1
    body: list[str] = []
    fence: tuple[str, int] | None = None

    for lineno, line in enumerate(text.split("\n"), start=1):
        in_fence_before = fence is not None
        fence = _next_fence(line, fence)
        m = None if (in_fence_before or fence is not None) else _HEADING_RE.match(line)
        if m is None:
            body.append(line)
            continue
        sections.append((title, path, level, start, body))
        lvl = len(m.group(1))
        new_title = m.group(2).strip() or "(untitled)"
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        stack.append((lvl, new_title))
        title, path, level, start = new_title, " > ".join(t for _, t in stack), lvl, lineno
        body = []
    sections.append((title, path, level, start, body))

    chunks: list[Chunk] = []
    for s_title, s_path, s_level, s_start, s_body in sections:
        content = "\n".join(s_body).strip()
        if not content:
            continue
        for piece in _split_long(content, max_chars):
            chunks.append(
                Chunk(
                    source=source,
                    heading=s_title,
                    heading_path=s_path,
                    level=s_level,
                    ordinal=len(chunks),
                    start_line=s_start,
                    text=piece,
                )
            )
    return chunks


# ------------------------------------------------------------------- database


def connect(db_path: str | os.PathLike[str], *, create: bool = True) -> sqlite3.Connection:
    """Open (and, if ``create``, initialise) an index database.

    Raises :class:`FileNotFoundError` when ``create`` is false and the file is missing,
    :class:`SchemaVersionError` for foreign or incompatible databases, and
    :class:`Fts5UnavailableError` if SQLite lacks FTS5.
    """
    path = Path(db_path)
    if not create and not path.is_file():
        raise FileNotFoundError(f"index database not found: {path}")
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            existing = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            if existing:
                raise SchemaVersionError(
                    f"{path} is an existing SQLite database that is not a claude-fts5 index"
                )
            if not create:
                raise SchemaVersionError(f"{path} is empty; run `claude-fts5 index` first")
            try:
                conn.executescript(_DDL)
            except sqlite3.OperationalError as exc:
                if "fts5" in str(exc).lower():
                    raise Fts5UnavailableError(
                        "this Python's SQLite was built without FTS5 support"
                    ) from exc
                raise
        elif version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"{path} has schema version {version}; this release supports {SCHEMA_VERSION}"
            )
    except BaseException:
        conn.close()
        raise
    return conn


# ------------------------------------------------------------------- indexing


@dataclass
class IndexStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks_written: int = 0
    total_files: int = 0
    total_chunks: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def discover_files(
    root: str | os.PathLike[str], globs: Sequence[str] = DEFAULT_GLOBS
) -> list[tuple[Path, str]]:
    """Return sorted ``(absolute_path, posix_relative_path)`` pairs under ``root``."""
    root_path = Path(root)
    if root_path.is_file():
        return [(root_path, root_path.name)]
    if not root_path.is_dir():
        raise FileNotFoundError(f"path not found: {root_path}")
    found: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if any(fnmatch.fnmatch(name.lower(), g.lower()) for g in globs):
                full = Path(dirpath) / name
                found.append((full, full.relative_to(root_path).as_posix()))
    found.sort(key=lambda item: item[1])
    return found


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _delete_file_rows(conn: sqlite3.Connection, rel: str) -> None:
    conn.execute("DELETE FROM chunks WHERE path = ?", (rel,))
    conn.execute("DELETE FROM files WHERE path = ?", (rel,))


def index_directory(
    db_path: str | os.PathLike[str],
    root: str | os.PathLike[str],
    *,
    globs: Sequence[str] = DEFAULT_GLOBS,
    prune: bool = True,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> IndexStats:
    """Incrementally index Markdown files under ``root`` into ``db_path``.

    A file is re-chunked only when the sha256 of its bytes differs from the
    stored hash. With ``prune`` (directories only), files that disappeared
    from ``root`` are removed from the index.
    """
    root_path = Path(root)
    files = discover_files(root_path, globs)
    stats = IndexStats()
    seen: set[str] = set()

    conn = connect(db_path, create=True)
    try:
        stored = {r["path"]: r["sha256"] for r in conn.execute("SELECT path, sha256 FROM files")}
        for full, rel in files:
            seen.add(rel)
            try:
                text, digest = read_file(full)
            except OSError as exc:
                stats.errors.append(f"{rel}: {exc}")
                continue
            if stored.get(rel) == digest:
                stats.unchanged += 1
                continue
            chunks = chunk_markdown(text, rel, max_chars=max_chars, default_title=full.stem)
            with conn:  # one transaction per file
                _delete_file_rows(conn, rel)
                conn.executemany(
                    "INSERT INTO chunks(path, ordinal, heading, heading_path, level, start_line, body)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (c.source, c.ordinal, c.heading, c.heading_path, c.level, c.start_line, c.text)
                        for c in chunks
                    ],
                )
                conn.execute(
                    "INSERT INTO files(path, sha256, n_chunks, indexed_at) VALUES (?, ?, ?, ?)",
                    (rel, digest, len(chunks), _utc_now()),
                )
            stats.chunks_written += len(chunks)
            if rel in stored:
                stats.updated += 1
            else:
                stats.added += 1

        if prune and root_path.is_dir():
            for rel in sorted(set(stored) - seen):
                with conn:
                    _delete_file_rows(conn, rel)
                stats.removed += 1

        stats.total_files = conn.execute("SELECT count(*) FROM files").fetchone()[0]
        stats.total_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()
    return stats


def iter_chunk_rows(conn: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    """Yield every stored chunk row (debugging / export helper)."""
    yield from conn.execute("SELECT * FROM chunks ORDER BY path, ordinal")
