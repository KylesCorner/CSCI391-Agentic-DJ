"""Assignment Part 5 evaluation: two chunkers + hybrid improvement on same corpus."""
import json
from pathlib import Path
from .memory import Memory


def evaluate(mem: Memory, question_file: Path, k: int = 3) -> dict:
    """Existing SQLite-compatible evaluator; with semantic store, evaluate all strategies."""
    examples = json.loads(question_file.read_text(encoding="utf-8"))
    if not examples:
        raise ValueError("A nonempty retrieval question set is required")
    if getattr(mem, "semantic", None) is None:
        rows=[]
        for example in examples:
            results = mem.search_knowledge(example["query"], limit=k)
            expected = example.get("expected",example.get("expected_section",""))
            hit = any(expected.casefold() in x.get("text","").casefold() for x in results)
            rows.append({"question":example["query"],"hit":hit,
                         "sources":[x["source"] for x in results]})
        return {"recall_at_k":sum(x["hit"] for x in rows)/len(rows),"k":k,"n":len(rows),"rows":rows}

    store = mem.semantic
    configs = (("fixed_vector","fixed","vector"),
               ("section_vector","section","vector"),
               ("section_hybrid","section","hybrid"))
    runs = {}
    for label, strategy, mode in configs:
        rows=[]
        for example in examples:
            if "expected_section" not in example:
                raise ValueError("Semantic benchmark needs expected_section ground truth")
            results = store.search(example["query"],limit=k,strategy=strategy,mode=mode)
            target = example["expected_section"]
            hit = any(target in row["section_ids"] for row in results)
            rows.append({"id":example["id"],"question":example["query"],"expected_section":target,
                         "hit":hit,"retrieved":[r["section_id"] for r in results]})
        runs[label]={"recall_at_3":round(sum(row["hit"] for row in rows)/len(rows),4) if k==3 else None,
                     "recall_at_k":round(sum(row["hit"] for row in rows)/len(rows),4),
                     "k":k,"n":len(rows),"rows":rows}
    return {"question_count":len(examples),"evaluation":"grounded section-ID recall; same queries for each index",
            "chunk_sizes":"fixed=96 whitespace-words / overlap=16; section=160 words / overlap=16",
            "strategies":runs,"improvement":"section_hybrid compared with section_vector"}
