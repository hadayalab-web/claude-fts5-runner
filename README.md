# claude-fts5-runner

A small, dependency-light CLI that grounds [Claude Code](https://docs.anthropic.com/en/docs/claude-code) prompts with local knowledge retrieved from a SQLite FTS5 index.

> **Status: early stage.** This repository is the working project for the Alignerr "build your own project with Claude Code" program. Everything here is being built with Claude Code in the open; nothing proprietary lives in this repo.

## Idea

Two ways to call a coding agent, chosen per task:

| Path | When | What happens |
| :--- | :--- | :--- |
| **A - direct** | Local, well-scoped task where the agent already has the files in context | Prompt goes straight through. No retrieval overhead. |
| **B - grounded** | Unknown error, cross-repo design question, anything where past notes matter | The runner queries a local SQLite FTS5 index and prepends the top hits to the prompt as a `<FTS5_KNOWLEDGE_GROUNDING>` block. |

## Planned scope

- [ ] `index` command: build an FTS5 index from a folder of Markdown notes
- [ ] `search` command: query the index, return ranked snippets
- [ ] `run` command: Path A / Path B wrapper around the `claude` CLI
- [ ] Unit tests for ranking, snippet building, and prompt assembly
- [ ] Windows-safe I/O (UTF-8 everywhere)

## License

MIT. See [LICENSE](LICENSE).
