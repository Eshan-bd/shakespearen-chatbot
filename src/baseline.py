from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from chunking import create_chunks, format_chunk_for_display
from data_loader import load_all_plays


Chunk = Dict[str, Any]


class SimpleTfidfVectorizer:
    def __init__(self, max_features: int = 12000) -> None:
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def _tokens(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']+", text) if len(w) > 2]

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        counts_list = [Counter(self._tokens(text)) for text in texts]
        df = Counter()
        for counts in counts_list:
            df.update(counts.keys())
        terms = [term for term, _ in df.most_common(self.max_features)]
        self.vocab = {term: i for i, term in enumerate(terms)}
        n_docs = max(1, len(texts))
        self.idf = np.array([math.log((1 + n_docs) / (1 + df[t])) + 1 for t in terms])
        return self._matrix(counts_list)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self._matrix([Counter(self._tokens(text)) for text in texts])

    def _matrix(self, counts_list: List[Counter]) -> np.ndarray:
        matrix = np.zeros((len(counts_list), len(self.vocab)), dtype=np.float32)
        for row, counts in enumerate(counts_list):
            total = sum(counts.values()) or 1
            for term, count in counts.items():
                col = self.vocab.get(term)
                if col is not None:
                    matrix[row, col] = count / total
        if self.idf is not None and matrix.size:
            matrix *= self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


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
