"""Held-out read-only probe. Measures repeatability and proxy tool selection accuracy.

This does NOT prove answer factual correctness: manually inspect against ground truth.
"""
from __future__ import annotations
from collections import Counter
import json
import statistics
from pathlib import Path
from .agent import run_agent, Ollama
from .tools import Tools
from .memory import Memory
from .clients import ListenBrainz, MusicBrainz


def evaluate_agent(lb: ListenBrainz, mb: MusicBrainz | None, mem: Memory, model: Ollama,
                   tasks_file: Path, repeats: int = 3) -> dict:
    tasks = json.loads(tasks_file.read_text())
    results = []
    for task in tasks:
        for trial in range(repeats):
            toolset = Tools(lb, mb, mem, approve=lambda _: False, current_goal=task["goal"])
            outcome = run_agent(task["goal"], toolset, model)
            called = [x["tool"] for x in outcome.tool_trace]
            results.append({"goal": task["goal"], "trial": trial, "status": outcome.status,
                            "required_tool_called": task["expected_tool"] in called,
                            "seconds": outcome.seconds, "actions": outcome.steps,
                            "tokens": outcome.prompt_tokens + outcome.generated_tokens,
                            "answer": outcome.answer[:400]})
    outcomes_by_goal = [
        len({r["status"] for r in results if r["goal"] == task["goal"]})
        for task in tasks
    ]
    return {
        "n_trials": len(results),
        "tool_selection_accuracy_proxy": sum(x["required_tool_called"] for x in results)/len(results) if results else 0,
        "mean_seconds": statistics.mean(x["seconds"] for x in results) if results else 0,
        "mean_tokens": statistics.mean(x["tokens"] for x in results) if results else 0,
        "run_to_run_status_variation": sum(x > 1 for x in outcomes_by_goal),
        "estimated_api_cost_usd": 0,  # Assumes locally hosted Ollama; excludes hardware/electricity.
        "caveat": "Accuracy above measures expected tool invocation, NOT correctness of final answers; manually score factual results.",
        "trials": results,
    }
