# claude-fts5-runner

A small, standard-library-only CLI that grounds [Claude Code](https://docs.anthropic.com/en/docs/claude-code) prompts with local knowledge retrieved from a SQLite FTS5 index.

> **Status: v0.1.0, early but working.** Everything here is built with Claude Code in the open; nothing proprietary lives in this repo. APIs and the on-disk schema may change before 1.0.

## Idea

Two ways to call a coding agent, chosen per task:

| Path | When | What happens |
| :--- | :--- | :--- |
| **A - direct** | Local, well-scoped task where the agent already has the files in context | The prompt goes straight through. No retrieval overhead. |
| **B - grounded** | Unknown error, cross-repo design question, anything where past notes matter | The prompt is used as a query against a local FTS5 index; the top BM25 hits are prepended as a `<FTS5_KNOWLEDGE_GROUNDING>` block. |

The prompt is delivered to `claude -p` on **stdin** as UTF-8 bytes, so long grounded prompts do not hit Windows command-line limits and non-ASCII text survives legacy code pages.

## Installation

Requires Python 3.10+ whose `sqlite3` module was built with FTS5 (true for the official CPython builds on Windows, macOS and most Linux distributions). There are no runtime dependencies. The Claude Code CLI (`claude`) is only needed for real (non-`--dry-run`) calls.

```bash
git clone https://github.com/hadayalab-web/claude-fts5-runner.git
cd claude-fts5-runner
pip install -e ".[dev]"      # installs the `claude-fts5` command + pytest
pytest                       # run the test suite
```

Without installing, `PYTHONPATH=src python -m claude_fts5_runner ...` works too.

## Usage

```bash
# 1. Index a folder of Markdown notes (incremental: unchanged files are skipped by sha256)
claude-fts5 index ./notes --db notes.db

# 2. Search it (BM25-ranked; punctuation such as ERR-404: is safe)
claude-fts5 search "WinError 10061" --db notes.db --top-k 3
claude-fts5 search "ERR-404: not-found" --db notes.db --match any --json

# 3a. Path A - pass the prompt straight to `claude -p`
claude-fts5 run "Refactor utils.py to remove the global cache" --mode direct

# 3b. Path B - ground the prompt with the top-k notes first
claude-fts5 run "Why do I get WinError 10061 on localhost?" \
    --mode grounded --db notes.db --top-k 5

# Inspect exactly what would be sent, without calling Claude
claude-fts5 run "Why do I get WinError 10061 on localhost?" \
    --mode grounded --db notes.db --dry-run
echo "prompt from a pipe" | claude-fts5 run - --dry-run --json
```

Useful options: `--model`, `--claude-bin`, `--claude-arg=--max-turns=3` (repeatable), `--timeout`. The database path defaults to `$CLAUDE_FTS5_DB`, then `claude_fts5.db`. Exit codes: `0` success, `2` usage/input error (bad mode, missing index, Claude CLI not found), `1` I/O or database failure; a real `run` returns the Claude CLI's own exit code.

A grounded prompt looks like this (the score is 0 only because this example has a one-note corpus, which gives BM25 no IDF signal; note text is XML-escaped, so an indexed note cannot close the block early):

```xml
<FTS5_KNOWLEDGE_GROUNDING hits="1">
  <hit rank="1" source="winerror.md" line="1" heading="WinError 10061" score="0.0000">
Connection refused on localhost: start the server before the client.
  </hit>
</FTS5_KNOWLEDGE_GROUNDING>

Why do I get WinError 10061 on localhost?
```

## How it works

- **Indexer** (`indexer.py`) - splits Markdown on ATX headings (ignoring `#` inside fenced code), keeps source path, heading title, `Parent > Child` breadcrumb and start line per chunk, and stores a sha256 of each file's raw bytes. Re-running `index` only re-chunks files whose hash changed and prunes files that disappeared. Storage is an external-content FTS5 table with the `porter unicode61` tokenizer, kept in sync by triggers. The schema carries a `user_version`; foreign or newer databases are refused rather than touched.
- **Search** (`search.py`) - `sanitize_query()` reduces every term to alphanumeric words and emits double-quoted FTS5 strings, so `-`, `:`, `*`, `(`, `AND`, `NOT`, `NEAR` and friends are always literals. `ERR-404` becomes the phrase `"ERR 404"`, which is how the tokenizer indexed it. Results come back as `SearchHit` dataclasses ranked by `bm25()` (heading weighted 3x body).
- **Runner** (`runner.py`) - `plan_prompt()` builds the Path A / Path B prompt; `invoke_claude_cli()` runs `claude -p` (or just describes the command with `dry_run=True`). Path B retrieves with OR semantics over at most 24 terms from the prompt, and falls back to the unchanged prompt when nothing matches.

The same pieces are importable:

```python
from claude_fts5_runner.indexer import index_directory
from claude_fts5_runner.search import search, format_grounding_block

index_directory("notes.db", "./notes")
hits = search("notes.db", "connection refused", top_k=3)
print(format_grounding_block(hits))
```

## Known limitations

- `unicode61` does not segment CJK text: an unbroken run of CJK characters is a single token, so a query only matches if it equals a whole run (space- or punctuation-delimited words work). This is covered by a test so the behaviour is explicit.
- Only Markdown (`*.md`, `*.markdown`) is indexed; code files are not chunked structurally yet.
- One database per corpus root (paths are stored relative to the indexed root).
- Grounding is capped per hit by characters (`max_chars`, default 2000), not by tokens.
- No latency or relevance benchmarks have been run yet, so this README makes no performance claims.

## Roadmap

### v0.1.0 (this release)

- [x] `index` command: incremental FTS5 index (`porter unicode61`) from a folder of Markdown notes, with sha256 change detection and pruning
- [x] Markdown section chunker: heading-based, fence-aware, with source path, heading breadcrumb and start line
- [x] `search` command: BM25-ranked results as structured `SearchHit` objects (text or `--json`)
- [x] FTS5 query sanitizer: hyphens, colons, quotes and reserved words can never cause FTS5 syntax errors
- [x] `run` command: Path A (direct) and Path B (FTS5-grounded) wrapper around the `claude` CLI, with `--dry-run`
- [x] `<FTS5_KNOWLEDGE_GROUNDING>` block with XML escaping
- [x] Windows-safe I/O (explicit UTF-8 for files, stdio and the subprocess pipe)
- [x] Unit and CLI tests for chunking, incremental indexing, sanitizing, ranking, prompt assembly and the subcommands

### v0.2.0 (planned)

- [ ] Automated schema migration between index versions (today an unsupported version is refused)
- [ ] Token-budget truncation and fallback routing (budget the whole grounding block; fall back to Path A when it does not fit)
- [ ] AST-aware code chunking (index source files by function/class, not just Markdown sections)
- [ ] Latency benchmark suite (indexing and query timings on reproducible corpora)

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
