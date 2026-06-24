import json
import time
from pathlib import Path
from statistics import mean

from rag.generator import answer_question
from rag.retriever import LocalVectorStore


def load_eval_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    return any(keyword in answer for keyword in keywords)


def run_evaluation(vector_store: LocalVectorStore, eval_path: Path) -> dict:
    cases = load_eval_questions(eval_path)
    rows = []
    latencies = []

    for case in cases:
        start = time.perf_counter()
        result = answer_question(case["question"], vector_store)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        latencies.append(latency_ms)
        has_source = bool(result.sources and result.sources[0].score > 0)
        hit = keyword_hit(result.answer, case.get("expected_keywords", []))
        should_refuse = bool(case.get("should_refuse", False))
        refusal_ok = result.refused if should_refuse else not result.refused
        rows.append(
            {
                "question": case["question"],
                "answer": result.answer,
                "keyword_hit": hit,
                "has_source": has_source,
                "refusal_ok": refusal_ok,
                "latency_ms": latency_ms,
                "used_provider": result.used_provider,
            }
        )

    total = len(rows) or 1
    return {
        "total_cases": len(rows),
        "keyword_match_rate": round(sum(row["keyword_hit"] for row in rows) / total, 3),
        "source_hit_rate": round(sum(row["has_source"] for row in rows) / total, 3),
        "refusal_accuracy": round(sum(row["refusal_ok"] for row in rows) / total, 3),
        "avg_latency_ms": round(mean(latencies), 2) if latencies else 0,
        "cases": rows,
    }

