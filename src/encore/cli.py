"""Command-line entrypoint: uv run encore ..."""
from __future__ import annotations
import argparse
import json
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
                print("Deleted local user data:", json.dumps(counts, indent=2))
                print("Existing remote ListenBrainz playlists are unchanged.")
            return
        if not cfg.username:
            parser.error("LISTENBRAINZ_USER missing; copy .env.example to .env")
        http = Http("Encore-Educational/0.1 (ListenBrainz music-class project)")
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
        tools = Tools(lb, mb, mem, approve=approval, current_goal=goal)
        result = run_agent(goal, tools, Ollama(cfg.ollama_host, cfg.model, cfg.summary_model),
                           max_steps=args.steps, trace_data=args.debug,
                           context_budget_tokens=args.context_budget)
        # Always attempt to save a structured episode, including unsuccessful
        # runs. Never mistake free-form final_answer text for a session summary.
        safe_goal = _strip_secrets(goal, (cfg.token,))
        safe_answer = _strip_secrets(result.answer, (cfg.token,))
        summarizer = SessionSummarizer(
            cfg.ollama_host, cfg.summary_model, secrets=(cfg.token,)
        )
        try:
            episode_summary, episode_review = summarizer.summarize(
                safe_goal, safe_answer, result.status, result.tool_trace
            )
        except Exception as exc:
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
            print("Episodic memory:", json.dumps(episode_result, ensure_ascii=False))
        except Exception as exc:
            print(f"[episodic] Memory write FAILED: {type(exc).__name__}: {exc}")
        print(f"\n{result.answer}\n")
        print(f"Status: {result.status} | actions: {result.steps} | seconds: {result.seconds}")
        print(f"Ollama tokens: prompt={result.prompt_tokens}, generated={result.generated_tokens} (local, no API billing)")
        print("Tool trace:", json.dumps(result.tool_trace, indent=2))
    finally:
        mem.close()


if __name__ == "__main__":
    main()
