from __future__ import annotations

from typing import Any, Dict, List, Tuple

from chunking import create_chunks, format_chunk_for_display
from data_loader import load_all_plays
from retrieval import SimpleTfidfVectorizer


Chunk = Dict[str, Any]


class BaselineSystem:
    def __init__(self) -> None:
        self.chunks = create_chunks(load_all_plays())
        self.vectorizer = SimpleTfidfVectorizer(max_features=12000)
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[Chunk, float]]:
        scores = (self.vectorizer.transform([query]) @ self.matrix.T)[0]
        order = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def answer(self, query: str) -> str:
        chunk, score = self.retrieve(query, top_k=1)[0]
        return (
            "Baseline answer: the most similar passage is shown below. "
            "This system does not generate a full explanation.\n\n"
            f"Similarity: {score:.4f}\n{format_chunk_for_display(chunk)}"
        )


def baseline_answer(query: str) -> str:
    return BaselineSystem().answer(query)


if __name__ == "__main__":
    question = "Who is Hamlet?"
    print("Question:", question)
    print(baseline_answer(question))
