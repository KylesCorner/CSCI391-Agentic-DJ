
"""Command-line entrypoint: uv run encore ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import Ollama, run_agent
from .clients import Http, ListenBrainz, MusicBrainz
from .config import get_config
from .memory import Memory
from .retrieval_eval import evaluate
from .tools import Tools


def approval(description: str) -> bool:
    print(
        f"\nACCOUNT WRITE APPROVAL REQUIRED:\n"
        f"{description}"
    )

    try:
        return (
            input("Approve this action? Type YES: ").strip()
            == "YES"
        )
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    parser = argparse.ArgumentParser(
        prog="encore"
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ----------------------------------------
    # AGENT
    # ----------------------------------------
    ask = commands.add_parser(
        "ask",
        help="Run a natural-language goal against the agent",
    )

    ask.add_argument(
        "goal",
        nargs="+",
        help="Free-text request",
    )

    ask.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Maximum agent actions (<=12)",
    )

    ask.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Include tool result previews and "
            "model no-tool-call text in trace"
        ),
    )

    # ----------------------------------------
    # LISTENBRAINZ
    # ----------------------------------------
    commands.add_parser(
        "recent",
        help="Test ListenBrainz read-only connection",
    )

    stats = commands.add_parser(
        "stats",
        help="Inspect ListenBrainz stats without Ollama",
    )

    stats.add_argument(
        "kind",
        choices=["artists", "recordings"],
    )

    stats.add_argument(
        "--range",
        default="this_month",
        dest="span",
    )

    stats.add_argument(
        "--count",
        type=int,
        default=10,
    )

    # ----------------------------------------
    # MEMORY
    # ----------------------------------------
    commands.add_parser(
        "init-knowledge",
        help="Index trusted local documentation",
    )

    ev = commands.add_parser(
        "eval-retrieval",
        help="Evaluate semantic memory Recall@k",
    )

    ev.add_argument(
        "--k",
        type=int,
        default=3,
    )

    # ----------------------------------------
    # EVALUATION
    # ----------------------------------------
    live_eval = commands.add_parser(
        "eval-agent",
        help=(
            "Evaluate read-only prompts for "
            "latency, variance, and tool choice"
        ),
    )

    live_eval.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    commands.add_parser(
        "delete-data",
        help="Delete local Encore memories/write logs",
    )

    args = parser.parse_args()
    cfg = get_config()

    mem = Memory(cfg.db)
    mem.prune(days=90)

    try:
        # ------------------------------------
        # KNOWLEDGE INDEX
        # ------------------------------------
        if args.command == "init-knowledge":
            kb = (
                Path(__file__).resolve().parents[2]
                / "knowledge"
                / "listenbrainz.md"
            )

            if not kb.exists():
                kb = Path("knowledge/listenbrainz.md")

            n = mem.ingest(
                "listenbrainz-api",
                kb.read_text(),
            )

            print(
                f"Indexed {n} knowledge chunks from {kb}"
            )
            return

        # ------------------------------------
        # RETRIEVAL EVALUATION
        # ------------------------------------
        if args.command == "eval-retrieval":
            path = (
                Path(__file__).resolve().parents[2]
                / "evals"
                / "retrieval_questions.json"
            )

            if not path.exists():
                path = Path(
                    "evals/retrieval_questions.json"
                )

            print(
                json.dumps(
                    evaluate(mem, path, args.k),
                    indent=2,
                )
            )
            return

        # ------------------------------------
        # PRIVACY DELETION
        # ------------------------------------
        if args.command == "delete-data":
            if approval(
                "DELETE ALL local Encore sessions, "
                "preferences, knowledge index, "
                "and write ledger"
            ):
                mem.delete_personal()
                print(
                    "Local personal data cleared. "
                    "Existing remote playlists are unchanged."
                )
            return

        if not cfg.username:
            parser.error(
                "LISTENBRAINZ_USER missing; "
                "copy .env.example to .env"
            )

        http = Http(
            "Encore-Educational/0.1 "
            "(ListenBrainz music-class project)"
        )

        lb = ListenBrainz(
            http,
            cfg.username,
            cfg.token,
            timezone_name=cfg.timezone,
        )

        # ------------------------------------
        # RECENT LISTENS
        # ------------------------------------
        if args.command == "recent":
            for item in lb.recent(10):
                print(
                    f"{item['artist']} - {item['title']}"
                )
            return

        # ------------------------------------
        # STATISTICS
        # ------------------------------------
        if args.command == "stats":
            print(
                json.dumps(
                    lb.top(
                        args.kind,
                        args.count,
                        args.span,
                    ),
                    indent=2,
                )
            )
            return

        mb = (
            MusicBrainz(http, cfg.mb_contact)
            if cfg.mb_contact
            else None
        )

        # ------------------------------------
        # AGENT EVALUATION
        # ------------------------------------
        if args.command == "eval-agent":
            from .agent_eval import evaluate_agent

            tasks = (
                Path(__file__).resolve().parents[2]
                / "evals"
                / "agent_readonly.json"
            )

            if not tasks.exists():
                tasks = Path(
                    "evals/agent_readonly.json"
                )

            output = evaluate_agent(
                lb,
                mb,
                mem,
                Ollama(
                    cfg.ollama_host,
                    cfg.model,
                ),
                tasks,
                repeats=args.repeats,
            )

            print(
                json.dumps(
                    output,
                    indent=2,
                )
            )
            return

        # ------------------------------------
        # NATURAL-LANGUAGE AGENT
        # ------------------------------------
        goal = " ".join(args.goal)

        tools = Tools(
            lb,
            mb,
            mem,
            approve=approval,
            current_goal=goal,
        )

        result = run_agent(
            goal,
            tools,
            Ollama(
                cfg.ollama_host,
                cfg.model,
            ),
            max_steps=args.steps,
            trace_data=args.debug,
        )

        mem.add_session(
            goal,
            result.answer,
            result.status,
        )

        print(f"\n{result.answer}\n")

        print(
            f"Status: {result.status} "
            f"| actions: {result.steps} "
            f"| seconds: {result.seconds}"
        )

        print(
            f"Ollama tokens: "
            f"prompt={result.prompt_tokens}, "
            f"generated={result.generated_tokens} "
            f"(local, no API billing)"
        )

        print(
            "Tool trace:",
            json.dumps(
                result.tool_trace,
                indent=2,
            ),
        )

    finally:
        mem.close()


if __name__ == "__main__":
    main()

