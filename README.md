# Encore — Music Agent (uv starter)

Runnable starter project for an agent over your own ListenBrainz account and MusicBrainz. This is an implementation scaffold, **not** a verified live-account deployment. Tests run with mocks, without changing the user's music account.

## Setup (from repository root)

```bash
cp .env.example .env
# edit LISTENBRAINZ_USER, LISTENBRAINZ_TOKEN, MB_CONTACT
uv sync --extra dev
ollama pull qwen3-coder:30b # or set OLLAMA_MODEL to an installed tool-calling model
uv run encore recent
uv run encore init-knowledge
uv run encore eval-retrieval --k 3
uv run pytest -q
uv run encore stats artists --range this_month  # direct API/fallback test, no LLM
uv run encore ask "What are my most-played artists this month?"
uv run encore ask "Suggest a jazz playlist based on my listens"
```

### Write examples

```bash
uv run encore ask "Create a 10-track jazz playlist I haven't played in the last seven days"
uv run encore ask "Export my playlist 11111111-2222-3333-4444-555555555555 to Spotify"
```

The write operations request a separate literal `YES` on stdin. You must connect Spotify/Apple Music at https://listenbrainz.org/settings/ first. Approval is never implied from a natural-language request. If the terminal is noninteractive, writes are denied.

## Architecture and assignment mapping

- **Working memory:** `agent.py` retains model+tool messages for one 12-action run. Not persisted as raw tool output.
- **Episodic memory:** `memory.py` SQLite sessions; FTS5 retrieval with a 90-day TTL; summaries of completed goals.
- **Semantic memory:** trusted, versioned Markdown API knowledge, 110-word chunks with 20-word overlap, SQLite FTS5/BM25. Readonly to the agent; `uv run encore init-knowledge` refreshes corpus atomically by source.
- **Procedural memory:** immutable `SYSTEM` prompt plus hard-coded approval, schema validation, verified MBIDs, action budget, and no automatic write retries.
- **Preferences:** separate explicitly sourced user statements, with evidence and a `forget_preference` tool.
- **Evaluation:** `eval-retrieval` computes a small held-out Recall@k baseline. `eval-agent` repeats read-only prompts (tool-invocation proxy accuracy, latency, local-model token cost, outcome variance). Do human scoring for factual answer correctness.
- **Safety:** `ToolResult` envelopes expose `ok`, `data`, `error_type`, `message`. Classifies `invalid`, `resource`, `transient`, `auth`, `not_found`, `fatal`, `uncertain_write`, `policy_halt`. GET retries are bounded; POST has no automatic retry. A SQLite write ledger blocks replay of the same operation within the run after a timeout/denial. Checks user approval *on every write*.

Tool router only limits the visible tool menu for simple queries; **the model makes each choice** based on the previous tool result. No preprogrammed action chain.

## Known limitations / improvements for submission

1. Verify actual live ListenBrainz + MusicBrainz responses with your own account before writing. Read-only tools are implemented, but only offline mocked tests have been run for this scaffold.
2. The `get_listens_since` scan is deliberately capped at 1,000 listens (10 pages). When incomplete, Encore must not claim a guaranteed seven-day exclusion. Full history should use ListenBrainz async export rather than unbounded pagination.
3. Song duration is from MusicBrainz and may be missing. Neither tag nor metadata reliably guarantees instrumental/no vocals; checking against a streaming catalog or curated features would improve results.
4. The write ledger handles replay only inside a session ID, and records `uncertain` after a write exception; an operator must check ListenBrainz before trying again. This is **not** distributed exactly-once execution across crashed/restarted sessions.
5. Retrieval is keyword/BM25 baseline, not a tested embedding-hybrid solution; compare chunk settings, recall@k and optionally add embeddings as an ablation. Documentation is a seeded, selectively curated corpus, not a complete API reference.
6. `eval-agent` is *read only* by design and does not establish factual task success. Record a human-labeled accuracy score and counterbalanced prompts for your final write-up.
7. Agent may use more than 12 individual **HTTP requests** (one tool can call several APIs); limit applies to LLM tool-action steps.
8. SQLite stores conversational goals and summaries locally, unencrypted. Do not include sensitive data in prompts. `uv run encore delete-data` removes local records, not remote ListenBrainz content. Secrets stay in `.env` and are excluded from version control.
9. Read requests to MusicBrainz are held to ~1/s; ListenBrainz GET retries honor Retry-After for 429 but further rate-limit header handling could be improved.

## File tree

```text
src/encore/
  agent.py          # Ollama JSON tool calls + bounded ReAct
  agent_eval.py     # repeatable read-only agent evaluation
  clients.py        # ListenBrainz/MusicBrainz HTTP clients
  config.py         # .env settings
  memory.py         # SQLite episodic/semantic/preferences + write ledger
  retrieval_eval.py # Recall@k over held-out queries
  tools.py          # tool JSON schemas, validation, approval policy
  cli.py            # entrypoint
knowledge/          # curated Markdown source (trusted ingestion only)
evals/              # held-out example inputs
tests/              # offline tests
```

## Authoritative documentation

- https://listenbrainz.readthedocs.io/en/latest/users/api/
- https://listenbrainz.readthedocs.io/en/latest/users/api/playlist.html
- https://listenbrainz.readthedocs.io/en/latest/users/api/misc.html
- https://musicbrainz.org/doc/MusicBrainz_API
- https://docs.ollama.com/capabilities/tool-calling

## Current-month statistics and fallback

`this_month` means the current calendar month. `month` is the previous calendar month.
Set `ENCORE_TIMEZONE` in `.env` (for example, `America/Denver`) to define
calendar boundaries for the raw-scrobble fallback. When ListenBrainz returns
empty/204 aggregate stats, `get_top_artists` and `get_top_recordings` automatically
count scrobbles for `this_month` using at most 10 pages (100 per page).
If the bounded scan cannot cover the entire month, the result explicitly marks
`complete: false`; Encore must not present it as a definitive ranking. All other
periods report a resource error rather than fabricating stats when unavailable.

For inspecting an API response without LLM interpretation:

```bash
uv run python - <<'PYEX'
from encore.config import get_config
from encore.clients import Http, ListenBrainz
cfg = get_config()
lb = ListenBrainz(Http('Encore-Debug/0.1'), cfg.username, cfg.token, timezone_name=cfg.timezone)
print(lb.top('artists', 10, 'this_month'))
PYEX
```
