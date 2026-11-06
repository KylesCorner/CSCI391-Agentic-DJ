# Encore DJ — Concise Student Workbook

## A. Design Decisions

| Question | Answer |
|---|---|
| Open-ended trajectory? | Yes. Tool calls depend on the request and API results. |
| Variable inputs? | Yes. Natural-language requests vary in intent and constraints. |
| Open output space? | Yes. Answers, recommendations, and playlist edits vary. |
| Real-world action? | Yes. Approved writes modify ListenBrainz data. |
| Model | Local Qwen via Ollama; separate execution and final-review tiers. |
| Tools | ListenBrainz, MusicBrainz, and memory; domain-based routing. |
| Memory | Working, episodic, semantic, and procedural stores. |
| Loop | Bounded ReAct loop; configurable step limit and retries. |
| Termination | Executor sets `answered`, `answered_low_confidence`, `stuck`, `budget_exceeded`, etc. |

**Memory configuration:** Demonstrated context budget: 12,000 tokens. Compression trigger, recency α, dedup threshold, TTL/cap, and embedding version require code verification. No compressions occurred in the 40 benchmark runs.

## B. Memory Evidence

| Behavior | Evidence |
|---|---|
| Episodic retrieval affected response | 5/5 memory scenarios show retrieval and expected cues; exact trace lines unavailable. |
| Compression preserved playlist | Not demonstrated; 0 compressions. |
| `search_music_knowledge` selected | Not verified. |
| Stale memory questioned | M05 distinguishes old swing from new folk preference. |
| Polluted summary flagged | `quality_flag=verify_before_using` appears; deliberate test unverified. |
| Duplicate summary merged | M02 shows `merged=true`; trace unverified. |
| `--forget` cleared stores | Not verified. |

**Semantic retrieval:** Two chunking strategies and top-3 recall scores are not in the supplied evaluation.

**1. Episodic summary:** Store reusable preferences and unfinished tasks, not a transcript. M02 preserves a dislike of heavy metal while reading. Add timestamps and distinguish user statements from inferences. An actual stored summary still needs to be pasted.

**2. Malicious playlist title:** Treat the title as untrusted data, never as a deletion command. Require an explicit request, scope checks, and approval before writes.

**3. Few-shot examples:** Keep tool-selection, verification-before-write, ambiguous-reference, and refusal examples. Exact retained examples require prompt-file verification.

## C. Evaluation Results

| Category | Runs | Correct | Incorrect | Ungraded | Mean latency | Mean peak input tokens |
|---|---:|---:|---:|---:|---:|---:|
| Happy | 14 | 10 | 4 | 0 | 58.31 s | 2,656 |
| Boundary | 12 | 4 | 7 | 1 | 56.85 s | 2,960 |
| Failure | 14 | 11 | 3 | 0 | 30.85 s | 2,071 |

**Overall:** 40 runs; 25 correct, 14 incorrect, 1 ungraded; mean latency **48.26 s**. Recorded API cost: **$0** (local Ollama; hardware costs excluded). Correctness: **25/39 (64.1%)** graded, or **25/40 (62.5%)** counting ungraded as incorrect.

### Three-run variance

| Request | Statuses | Tool calls | Same answer? |
|---|---|---|---|
| H02 | answered / answered / answered | 1–1 | Yes |
| H05 | answered / answered / answered | 2–4 | No |
| B05 | answered_low_confidence / stuck / stuck | 1–1 | No |
| F01 | answered / answered / answered | 1–1 | Yes |
| F08 | stuck / stuck / stuck | 1–1 | Yes |

### Memory scenarios

| Scenario | Session 1 | Session 2 | Episodic read | Expected cues |
|---|---|---|---|---|
| M01 | answered | answered_low_confidence | Yes | Yes |
| M02 | answered | answered | Yes | Yes |
| M03 | policy_halt | answered | Yes | Yes |
| M04 | answered | answered_low_confidence | Yes | Yes |
| M05 | stuck | answered_low_confidence | Yes | Yes |

Memory-scenario correctness labels are blank; recall cues do not prove overall correctness.

**Failure 1 — H08:** Artist-relationship request hit `budget_exceeded` after 4 tool calls, 15 LLM calls, and 91.82 s. Reduce redundant calls and terminate when sufficient evidence is available.

**Failure 2 — B05:** Prior-preference recall returned one low-confidence answer and two `stuck` outcomes. Improve memory grounding and final-answer fallback.

## D. Final Reflection

**1. Where did Encore forget?** B05 inconsistently used prior preferences: one low-confidence answer and two stuck runs. Memory reads occurred, so the issue appears to be using retrieved information, not proven data loss.

**2. Why separate history and episodic memory?** ListenBrainz history records actual listens; episodic memory stores agent summaries. Separation prevents inferred summaries from becoming authoritative listening facts.

**3. Did tiering pay off?** Tiering may reduce costly model calls, but these evaluations do not isolate its impact. A controlled single-model comparison is needed.

**4. What would make Spotify trustworthy?** Scoped OAuth, approval before writes, change previews, verification, undo capability, audit logs, and secure token handling.

---

*Source: uploaded Encore evaluation archive, as summarized in the supplied workbook draft. Unverified configuration and trace fields remain explicitly marked.*
