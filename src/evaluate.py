from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from config import RESULTS_DIR
from baseline import BaselineSystem
from chunking import format_chunk_for_display
from rag_chatbot import RAGChatbot, build_retriever, generate_answer


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
    chatbot = RAGChatbot()

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
        baseline_reply = baseline.query(question)
        bot_reply = chatbot.query(question)
        outputs = {
            "baseline": baseline_reply,
            "rag": bot_reply 
        }

        for system_name, response in outputs.items():
            rows.append({
                "question_id": q.get("question_id", ""),
                "question": question,
                "question_type": q.get("type", ""),
                "expected_focus": q.get("expected_focus", ""),
                "system": system_name,
                "generated_response": response,
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote evaluation results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_evaluation_results()
