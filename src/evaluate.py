from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from config import RESULTS_DIR
from baseline import BaselineSystem
from chunking import format_chunk_for_display
from rag_chatbot import build_retriever, generate_answer


QUESTIONS_PATH = RESULTS_DIR / "instructor_questions.json"
OUTPUT_PATH = RESULTS_DIR / "evaluation_results.csv"


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _evidence_text(retrieved) -> str:
    return "\n\n".join(format_chunk_for_display(chunk) for chunk, _ in retrieved)


def create_evaluation_results() -> None:
    questions = load_questions()
    baseline = BaselineSystem()
    retriever = build_retriever()

    fieldnames = [
        "question_id",
        "question",
        "question_type",
        "expected_focus",
        "system",
        "retrieved_passages",
        "generated_response",
        "correctness_score",
        "grounding_score",
        "retrieval_relevance_score",
        "usefulness_score",
        "style_quality_score",
        "comments",
    ]

    rows = []
    for q in questions:
        question = q.get("question", "")
        base_retrieved = baseline.retrieve(question, top_k=1)
        rag_retrieved = retriever.retrieve(question, top_k=3)
        outputs = {
            "baseline": (base_retrieved, baseline.answer(question)),
            "rag": (rag_retrieved, generate_answer(question, rag_retrieved)),
        }

        for system_name, (retrieved, response) in outputs.items():
            rows.append({
                "question_id": q.get("question_id", ""),
                "question": question,
                "question_type": q.get("type", ""),
                "expected_focus": q.get("expected_focus", ""),
                "system": system_name,
                "retrieved_passages": _evidence_text(retrieved),
                "generated_response": response,
                "correctness_score": "",
                "grounding_score": "",
                "retrieval_relevance_score": "",
                "usefulness_score": "",
                "style_quality_score": "",
                "comments": "",
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote evaluation results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_evaluation_results()
