"""Run retrieval evaluation BEFORE tuning the agent. No LLM/API calls."""
import json
from pathlib import Path
from .memory import Memory


def evaluate(mem: Memory, question_file: Path, k: int = 3) -> dict:
    examples = json.loads(question_file.read_text())
    rows = []
    for example in examples:
        results = mem.search_knowledge(example["query"], limit=k)
        hit = any(example["expected"].casefold() in x["text"].casefold() for x in results)
        rows.append({"question": example["query"], "hit": hit,
                     "sources": [x["source"] for x in results]})
    return {"recall_at_k": sum(row["hit"] for row in rows) / len(rows) if rows else 0,
            "k": k, "n": len(rows), "rows": rows}
