"""FTS5 query sanitising, BM25-ranked search, and grounding-block formatting."""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from typing import Sequence
from xml.sax.saxutils import escape, quoteattr

from .indexer import connect

HEADING_WEIGHT = 3.0  # bm25 column weight for the heading
BODY_WEIGHT = 1.0  # bm25 column weight for the body

_TERM_RE = re.compile(r'"([^"]*)"|(\S+)')
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


class QueryError(ValueError):
    """SQLite rejected the (sanitised) MATCH expression."""


@dataclass(frozen=True)
class SearchHit:
    rank: int  # 1-based position in the result list
    path: str
    heading: str
    heading_path: str
    ordinal: int
    start_line: int
    score: float  # -bm25(); higher is more relevant
    snippet: str
    text: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def sanitize_query(query: str, *, match: str = "all", max_terms: int | None = None) -> str:
    """Turn free text into a *safe* FTS5 MATCH expression.

    Every whitespace-separated term is reduced to its alphanumeric word parts
    and emitted as a double-quoted FTS5 string, so characters such as ``-``,
    ``:``, ``*``, ``^``, ``(``, ``)`` and the keywords ``AND``/``OR``/``NOT``/
    ``NEAR`` are always literals and can never raise an FTS5 syntax error.

    * ``ERR-404:`` becomes the phrase ``"ERR 404"`` (matches how the tokenizer
      indexes the original text).
    * A user-supplied ``"quoted phrase"`` stays a phrase.
    * ``match="all"`` joins terms with an implicit AND; ``"any"`` joins with OR
      (useful for long natural-language prompts; BM25 still ranks the results).
    * Duplicate terms are dropped; ``max_terms`` caps the number of terms.

    Returns ``""`` if the input contains no word characters.
    """
    if match not in ("all", "any"):
        raise ValueError("match must be 'all' or 'any'")
    phrases: list[str] = []
    seen: set[str] = set()
    for m in _TERM_RE.finditer(query):
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        words = _WORD_RE.findall(raw)
        if not words:
            continue
        phrase = " ".join(words)
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
        if max_terms is not None and len(phrases) >= max_terms:
            break
    quoted = ['"' + p.replace('"', '""') + '"' for p in phrases]
    return (" OR " if match == "any" else " ").join(quoted)


_SQL = """
SELECT c.path, c.heading, c.heading_path, c.ordinal, c.start_line, c.body,
       bm25(chunks_fts, ?, ?) AS s,
       snippet(chunks_fts, 1, '[', ']', '...', 24) AS snip
FROM chunks_fts
JOIN chunks AS c ON c.id = chunks_fts.rowid
WHERE chunks_fts MATCH ?
ORDER BY s, c.path, c.ordinal
LIMIT ?
"""


def search(
    db: str | os.PathLike[str] | sqlite3.Connection,
    query: str,
    *,
    top_k: int = 5,
    match: str = "all",
    max_terms: int | None = None,
) -> list[SearchHit]:
    """BM25-ranked search. ``db`` is a database path or an open connection.

    Returns at most ``top_k`` hits, best first; an empty/unsearchable query
    returns ``[]``. Heading matches are weighted above body matches.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    fts_query = sanitize_query(query, match=match, max_terms=max_terms)
    if not fts_query:
        return []

    owns = not isinstance(db, sqlite3.Connection)
    conn = connect(db, create=False) if owns else db
    try:
        try:
            rows = conn.execute(_SQL, (HEADING_WEIGHT, BODY_WEIGHT, fts_query, top_k)).fetchall()
        except sqlite3.OperationalError as exc:
            raise QueryError(f"FTS5 rejected query {fts_query!r}: {exc}") from exc
    finally:
        if owns:
            conn.close()

    return [
        SearchHit(
            rank=i,
            path=r["path"],
            heading=r["heading"],
            heading_path=r["heading_path"],
            ordinal=r["ordinal"],
            start_line=r["start_line"],
            score=-float(r["s"]),
            snippet=" ".join(r["snip"].split()),
            text=r["body"],
        )
        for i, r in enumerate(rows, start=1)
    ]


GROUNDING_TAG = "FTS5_KNOWLEDGE_GROUNDING"


def format_grounding_block(
    hits: Sequence[SearchHit],
    *,
    query: str | None = None,
    max_chars: int | None = 2000,
) -> str:
    """Render hits as a ``<FTS5_KNOWLEDGE_GROUNDING>`` XML block.

    Text and attributes are XML-escaped, so indexed notes cannot close the
    block early (e.g. a note containing ``</FTS5_KNOWLEDGE_GROUNDING>``).
    Each hit's body is cut to ``max_chars`` characters (``None`` disables the
    cut) and marked ``truncated="true"``. Returns ``""`` when there are no hits.
    """
    if not hits:
        return ""
    query_attr = f" query={quoteattr(query)}" if query else ""
    lines = [f'<{GROUNDING_TAG}{query_attr} hits="{len(hits)}">']
    for hit in hits:
        body = hit.text.strip()
        truncated = bool(max_chars) and len(body) > max_chars
        if truncated:
            body = body[:max_chars].rstrip()
        flag = ' truncated="true"' if truncated else ""
        lines.append(
            f'  <hit rank="{hit.rank}" source={quoteattr(hit.path)}'
            f' line="{hit.start_line}" heading={quoteattr(hit.heading_path)}'
            f' score="{hit.score:.4f}"{flag}>'
        )
        lines.append(escape(body))
        lines.append("  </hit>")
    lines.append(f"</{GROUNDING_TAG}>")
    return "\n".join(lines)
