"""Deterministic RAG evaluation for ClaimIQ's local policy corpus."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.config import ARTIFACTS_DIR, get_logger
from src.rag.retriever import retrieve

LOGGER = get_logger(__name__)
REPORT_PATH = ARTIFACTS_DIR / "rag_evaluation.json"
QA_PATH = Path(__file__).with_name("policy_qa.json")


def evaluate(top_k: int = 3) -> dict[str, Any]:
    """Evaluate retrieval, deterministic grounded answers, and citations."""
    questions = json.loads(QA_PATH.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for qa in questions:
        started = time.perf_counter()
        passages = retrieve(qa["question"], top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        sources = [passage.source for passage in passages]
        answer = passages[0].text if passages else ""
        citation = passages[0].citation if passages else ""
        retrieval_hit = qa["expected_source"] in sources
        results.append(
            {
                "id": qa["id"],
                "question": qa["question"],
                "expected_source": qa["expected_source"],
                "retrieval_hit": retrieval_hit,
                "answer_correct": qa["expected_phrase"].lower() in answer.lower(),
                "citation_accurate": bool(citation and qa["expected_source"] in citation),
                "grounded": bool(answer and answer in [passage.text for passage in passages]),
                "latency_ms": round(latency_ms, 2),
            }
        )
    count = len(results)
    report = {
        "question_count": count,
        "top_k": top_k,
        "retrieval_hit_rate": round(sum(item["retrieval_hit"] for item in results) / count, 3),
        "answer_correctness": round(sum(item["answer_correct"] for item in results) / count, 3),
        "citation_accuracy": round(sum(item["citation_accurate"] for item in results) / count, 3),
        "groundedness": round(sum(item["grounded"] for item in results) / count, 3),
        "average_latency_ms": round(sum(item["latency_ms"] for item in results) / count, 2),
        "results": results,
    }
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote RAG evaluation report for %d questions", count)
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
