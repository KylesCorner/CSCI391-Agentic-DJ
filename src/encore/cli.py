"""Command-line entrypoint: uv run encore ..."""
from __future__ import annotations
import argparse
import json
import os
import shutil
from uuid import uuid4
from pathlib import Path
from datetime import datetime, timezone
from .agent import Ollama, run_agent
from .clients import Http, ListenBrainz, MusicBrainz
from .config import get_config
from .memory import Memory
from .episodic import EpisodicStore, OllamaEmbedder, SessionSummarizer, SUMMARY_FIELDS, _strip_secrets
from .retrieval_eval import evaluate
from .semantic import SemanticStore
from .tools import Tools
from .faults import FaultInjector
from .observability import SessionTrace, safe_user_dir


def approval(description: str) -> bool:
    print(f"\nACCOUNT WRITE APPROVAL REQUIRED:\n{description}")
    try:
        return input("Approve this action? Type YES: ").strip() == "YES"
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    parser = argparse.ArgumentParser(prog="encore")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="Run a natural-language goal against the agent")
    ask.add_argument("goal", nargs="+", help="Free-text request")
    ask.add_argument("--steps", type=int, default=12, help="Maximum agent actions (<=12)")
    ask.add_argument("--debug", action="store_true", help="Include short tool result previews and model no-tool-call text in trace")
    ask.add_argument("--context-budget", type=int, default=8000,
                     help="Approximate model context budget in tokens; compression triggers at 70 percent")
    ask.add_argument("--inject", type=Path,
                     help="JSON fault configuration: transient_rate, seed, max_steps")
    ask.add_argument("--read-attempts", type=int, default=3,
                     help="Max read-only tool attempts after transient errors (1..5); writes never retry")
    commands.add_parser("recent", help="Test ListenBrainz read-only connection")
    stats = commands.add_parser("stats", help="Inspect ListenBrainz stats directly, without Ollama")
    stats.add_argument("kind", choices=["artists", "recordings"])
    stats.add_argument("--range", default="this_month", dest="span")
    stats.add_argument("--count", type=int, default=10)
    commands.add_parser("init-knowledge", help="Index trusted local documentation")
    ev = commands.add_parser("eval-retrieval", help="Evaluate semantic memory Recall@k")
    ev.add_argument("--k", type=int, default=3)
    ev.add_argument("--output", default="evals/semantic_results.json", help="Write complete 3-way retrieval metrics as JSON")
    live_eval = commands.add_parser("eval-agent", help="Evaluate read-only prompts for latency/variance/tool choice")
    live_eval.add_argument("--repeats", type=int, default=3)
    commands.add_parser("delete-data", help="Delete local Encore memories/write logs")
    commands.add_parser("forget", help="Delete all local user memories (alias for delete-data)")
    episodes = commands.add_parser("episodes", help="Search prior episodic memories")
    episodes.add_argument("query", nargs="+", help="Memory topic to search")
    pruning = commands.add_parser("prune-episodes", help="Run memory TTL/cap pruning; supports simulated time")
    pruning.add_argument("--now", help="Simulated UTC time, e.g. 2027-04-01T00:00:00+00:00")
    args = parser.parse_args()
    cfg = get_config()
    trace_root = Path(os.getenv("ENCORE_TRACE_DIR", "./traces"))
    session_trace = None
    mem = Memory(cfg.db)
    mem.prune(days=90)
    # Public semantic knowledge is independent from user-scoped episodic memory.
    # Keep the collection separate even when the same Chroma directory is used.
    if args.command in {"ask", "init-knowledge", "eval-retrieval"}:
        mem.enable_semantic(SemanticStore(
            cfg.episodic_path, OllamaEmbedder(cfg.ollama_host, cfg.embedding_model)
        ))
    # For ask/episodes/deletion use the real user-scoped Chroma store.
    if args.command in {"ask", "episodes", "prune-episodes", "delete-data", "forget"}:
        store = EpisodicStore(
            cfg.episodic_path, cfg.user_id,
            OllamaEmbedder(cfg.ollama_host, cfg.embedding_model),
        )
        mem.enable_episodic(store)
        if args.command != "prune-episodes":
            store.prune()
    try:
        if args.command == "init-knowledge":
            corpus = Path(__file__).resolve().parents[2] / "knowledge"
            if not corpus.is_dir():
                corpus = Path("knowledge")
            report = mem.semantic.index(corpus)
            print(json.dumps(report, indent=2))
            return
        if args.command == "eval-retrieval":
            path = Path(__file__).resolve().parents[2] / "evals" / "retrieval_questions.json"
            if not path.exists():
                path = Path("evals/retrieval_questions.json")
            report = evaluate(mem, path, args.k)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"report":str(output_path),
                              "recall_at_k":{k:v["recall_at_k"] for k,v in report["strategies"].items()},
                              "n":report["question_count"],"k":args.k},indent=2))
            return
        if args.command == "prune-episodes":
            when = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(timezone.utc)
            if when.tzinfo is None:
                parser.error("--now must include a timezone, e.g. +00:00")
            print(json.dumps({"removed": mem.episodic.prune(now=when), "now": when.isoformat()}, indent=2))
            return
        if args.command == "episodes":
            print(json.dumps(mem.search_sessions(" ".join(args.query), limit=3), indent=2, ensure_ascii=False))
            return
        if args.command in {"delete-data", "forget"}:
            if approval("DELETE ALL local Encore sessions, preferences, knowledge index, and write ledger"):
                counts = mem.delete_personal()
                trace_path = safe_user_dir(trace_root, cfg.user_id or cfg.username)
                counts["trace_files"] = len(list(trace_path.glob("*.jsonl"))) if trace_path.exists() else 0
                if trace_path.exists():
                    shutil.rmtree(trace_path)
                print("Deleted local user data:", json.dumps(counts, indent=2))
                print("Existing remote ListenBrainz playlists are unchanged.")
            return
        if not cfg.username:
            parser.error("LISTENBRAINZ_USER missing; copy .env.example to .env")
        if args.command == "ask":
            if not 1 <= args.read_attempts <= 5:
                parser.error("--read-attempts must be between 1 and 5")
            if not 1 <= args.steps <= 12:
                parser.error("--steps must be between 1 and 12")
            try:
                injection = FaultInjector.load(args.inject) if args.inject else None
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                parser.error(f"Invalid --inject file: {exc}")
            if injection and injection.max_steps is not None:
                args.steps = injection.max_steps
            session_trace = SessionTrace(
                trace_root, cfg.user_id or cfg.username, str(uuid4()),
                secrets=(cfg.token,),
            )
            session_trace.emit("session_start", command="ask", max_steps=args.steps,
                               model=cfg.model, summary_model=cfg.summary_model,
                               cheap_model=cfg.cheap_model, final_model=cfg.final_model,
                               router_model=cfg.router_model,
                               embedding_model=cfg.embedding_model,
                               context_budget=args.context_budget,
                               injected=bool(injection))
            if injection:
                injection.reporter = session_trace.emit
                session_trace.emit("fault_config", transient_rate=injection.transient_rate,
                                   seed=injection.seed, max_steps=injection.max_steps)
        else:
            injection = None
        http = Http("Encore-Educational/0.1 (ListenBrainz music-class project)",
                    injector=injection,
                    reporter=session_trace.emit if session_trace is not None else None)
        lb = ListenBrainz(http, cfg.username, cfg.token, timezone_name=cfg.timezone)
        if args.command == "recent":
            for item in lb.recent(10):
                print(f"{item['artist']} - {item['title']}")
            return
        if args.command == "stats":
            print(json.dumps(lb.top(args.kind, args.count, args.span), indent=2))
            return
        mb = MusicBrainz(http, cfg.mb_contact) if cfg.mb_contact else None
        if args.command == "eval-agent":
            from .agent_eval import evaluate_agent
            tasks = Path(__file__).resolve().parents[2] / "evals" / "agent_readonly.json"
            if not tasks.exists():
                tasks = Path("evals/agent_readonly.json")
            output = evaluate_agent(lb, mb, mem, Ollama(cfg.ollama_host, cfg.model),
                                    tasks, repeats=args.repeats)
            print(json.dumps(output, indent=2))
            return
        goal = " ".join(args.goal)
        def traced_approval(description: str) -> bool:
            approved = approval(description)
            session_trace.emit("approval", approved=approved, action=description[:800],
                               requires_user_confirmation=True)
            return approved

        tools = Tools(lb, mb, mem, approve=traced_approval,
                      current_goal=goal, session_id=session_trace.session_id)
        result = run_agent(goal, tools, Ollama(cfg.ollama_host, cfg.model, cfg.summary_model,
                                              cheap_model=cfg.cheap_model,
                                              final_model=cfg.final_model,
                                              router_model=cfg.router_model),
                           max_steps=args.steps, trace_data=args.debug,
                           context_budget_tokens=args.context_budget,
                           event_log=session_trace,
                           retry_seed=injection.seed if injection else 0,
                           max_read_attempts=args.read_attempts)
        # Always attempt to save a structured episode, including unsuccessful
        # runs. Never mistake free-form final_answer text for a session summary.
        safe_goal = _strip_secrets(goal, (cfg.token,))
        safe_answer = _strip_secrets(result.answer, (cfg.token,))
        summarizer = SessionSummarizer(
            cfg.ollama_host, cfg.summary_model, secrets=(cfg.token,)
        )
        session_trace.emit("memory_summary_start", model=cfg.summary_model, tier="cheap")
        try:
            episode_summary, episode_review = summarizer.summarize(
                safe_goal, safe_answer, result.status, result.tool_trace
            )
            session_trace.emit("memory_summary_end", success=True, model=cfg.summary_model, tier="cheap",
                               quality_score=episode_review.get("score"))
        except Exception as exc:
            session_trace.emit("memory_summary_end", success=False, model=cfg.summary_model,
                               tier="cheap", error_category=type(exc).__name__)
            print(f"[episodic] LLM summary/quality call failed: {type(exc).__name__}. "
                  "Saving a low-trust fallback rather than fabricating memory.")
            episode_summary = {key: [] for key in SUMMARY_FIELDS}
            episode_summary["open_threads"] = [f"Request: {safe_goal[:350]} | Outcome: {result.status}"]
            episode_summary["errors"] = [f"Summary/check unavailable: {type(exc).__name__}"]
            episode_review = {"consistent": False, "score": 0.0, "issues": ["Summary model unavailable"]}
        try:
            # Do not persist credentials even if the model unexpectedly echoes one.
            episode_summary = json.loads(_strip_secrets(json.dumps(episode_summary), (cfg.token,)))
            episode_result = mem.add_structured_session(safe_goal, episode_summary, result.status, episode_review)
            session_trace.emit("memory_write", success=True, store="episodic",
                               quality_flag=episode_result.get("quality_flag"),
                               merged=episode_result.get("merged"))
            print("Episodic memory:", json.dumps(episode_result, ensure_ascii=False))
        except Exception as exc:
            session_trace.emit("memory_write", success=False, store="episodic",
                               error_category=type(exc).__name__)
            print(f"[episodic] Memory write FAILED: {type(exc).__name__}: {exc}")
        print(f"\n{result.answer}\n")
        print(f"Status: {result.status} | confidence: {result.confidence} | actions: {result.steps} | seconds: {result.seconds}")
        if result.sources:
            print("Sources:", json.dumps(result.sources[:15], ensure_ascii=False))
        print(f"Ollama tokens: prompt={result.prompt_tokens}, generated={result.generated_tokens} (local, no API billing)")
        print("Tool trace:", json.dumps(result.tool_trace, indent=2))
        session_trace.emit("session_end", status=result.status,
                           actions=result.steps, latency_s=result.seconds,
                           tokens_in=result.prompt_tokens, tokens_out=result.generated_tokens)
        print("Saved trace:", session_trace.path)
    finally:
        if session_trace is not None:
            session_trace.close()
        mem.close()


if __name__ == "__main__":
    main()
