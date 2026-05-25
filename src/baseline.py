from __future__ import annotations

from typing import Any, Dict, List, Tuple

from chunking import format_chunk_for_display
from rag_chatbot import build_retriever


Chunk = Dict[str, Any]


class BaselineSystem:
    def __init__(self) -> None:
        self.retriever = build_retriever()

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[Chunk, float]]:
        return self.retriever.retrieve(query, top_k=top_k)

    def answer(self, query: str) -> str:
        chunk, score = self.retrieve(query, top_k=1)[0]
        return (
            "Baseline answer: the most relevant Chroma passage is shown below. "
            "This baseline uses the same retrieval index as the RAG system, but does not "
            "synthesise a beginner-friendly explanation.\n\n"
            f"Similarity: {score:.4f}\n{format_chunk_for_display(chunk)}"
        )


def baseline_answer(query: str) -> str:
    return BaselineSystem().answer(query)


if __name__ == "__main__":
    question = "Who is Hamlet?"
    print("Question:", question)
    print(baseline_answer(question))
