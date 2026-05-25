from __future__ import annotations

import json
import math
import pickle
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


Chunk = Dict[str, Any]


def expand_query(query: str) -> str:
    q = query.lower()
    extra: List[str] = []
    rules = [
        (["macbeth", "duncan"], "ambition prophecy crown king murder deed Lady Macbeth persuade"),
        (["hamlet", "revenge"], "ghost Claudius proof play conscience delay father murder"),
        (["who", "hamlet"], "Prince Denmark father ghost Claudius Gertrude revenge"),
        (["montague", "capulet"], "feud brawl houses Romeo Juliet enemy ancient grudge"),
        (["juliet", "romeo"], "Capulet Montague love enemy balcony family conflict"),
        (["lady macbeth"], "wife ambition persuade Duncan murder guilt sleepwalking"),
    ]
    for terms, addition in rules:
        if all(term in q for term in terms):
            extra.append(addition)
    return f"{query} {' '.join(extra)}".strip()


class SimpleTfidfVectorizer:
    def __init__(self, max_features: int = 20000) -> None:
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def _tokens(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']+", text) if len(w) > 2]

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        doc_counts = [Counter(self._tokens(text)) for text in texts]
        df = Counter()
        for counts in doc_counts:
            df.update(counts.keys())
        terms = [term for term, _ in df.most_common(self.max_features)]
        self.vocab = {term: i for i, term in enumerate(terms)}
        n_docs = max(1, len(texts))
        self.idf = np.array([math.log((1 + n_docs) / (1 + df[t])) + 1 for t in terms])
        return self._matrix(doc_counts)

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


class ChromaRetriever:
    def __init__(self, persist_dir: Path, embedding_model_name: str):
        self.persist_dir = Path(persist_dir)
        self.embedding_model_name = embedding_model_name
        self.backend = "chromadb"
        self.embedding_backend = "tfidf"
        self.collection_name = "shakespeare_chunks"
        self.client = None
        self.collection = None
        self.model = None
        self.vectorizer = SimpleTfidfVectorizer()
        self.fallback = None

        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(self.collection_name)
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(embedding_model_name, local_files_only=True)
                self.embedding_backend = "sentence-transformers"
            except Exception:
                self.model = None
        except Exception:
            self.backend = "tfidf-fallback"

    def build_index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks supplied to build_index().")

        if self.backend != "chromadb":
            self.fallback = TfidfRetriever()
            self.fallback.build_index(chunks)
            return

        shutil.rmtree(self.persist_dir, ignore_errors=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        import chromadb

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(self.collection_name)

        ids = [chunk["chunk_id"] for chunk in chunks]
        docs = [chunk.get("embedding_text") or chunk["text"] for chunk in chunks]
        if self.embedding_backend == "sentence-transformers":
            embeddings = self.model.encode(docs, show_progress_bar=True, normalize_embeddings=True).tolist()
        else:
            embeddings = self.vectorizer.fit_transform(docs).tolist()
        metadatas = [self._metadata(chunk) for chunk in chunks]

        for start in range(0, len(chunks), 128):
            end = start + 128
            self.collection.add(
                ids=ids[start:end],
                documents=docs[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if self.backend != "chromadb":
            return self.fallback.retrieve(query, top_k)

        expanded = expand_query(query)
        if self.embedding_backend == "sentence-transformers":
            query_embedding = self.model.encode([expanded], normalize_embeddings=True).tolist()[0]
        else:
            query_embedding = self.vectorizer.transform([expanded]).tolist()[0]
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        chunks = []
        for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
            chunk = self._chunk(metadata)
            score = 1.0 / (1.0 + float(distance))
            chunks.append((chunk, score))
        return chunks

    def save(self, path: Path) -> None:
        if self.backend != "chromadb":
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "persist_dir": self.persist_dir,
                "embedding_model_name": self.embedding_model_name,
                "embedding_backend": self.embedding_backend,
                "vectorizer": self.vectorizer,
            }, f)

    @classmethod
    def load(cls, path: Path, embedding_model_name: str | None = None) -> "ChromaRetriever":
        from config import EMBEDDING_MODEL_NAME

        with path.open("rb") as f:
            data = pickle.load(f)
        obj = cls(data["persist_dir"], embedding_model_name or data.get("embedding_model_name") or EMBEDDING_MODEL_NAME)
        obj.embedding_backend = data.get("embedding_backend", "tfidf")
        obj.vectorizer = data["vectorizer"]
        return obj

    def _metadata(self, chunk: Chunk) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.get("chunk_id", ""),
            "play": chunk.get("play", ""),
            "act": chunk.get("act", ""),
            "scene": chunk.get("scene", ""),
            "speaker": chunk.get("speaker", ""),
            "text": chunk.get("text", ""),
            "scene_summary": chunk.get("scene_summary", ""),
            "keywords": json.dumps(chunk.get("keywords", [])),
        }

    def _chunk(self, metadata: Dict[str, Any]) -> Chunk:
        return {
            "chunk_id": metadata.get("chunk_id", ""),
            "play": metadata.get("play", ""),
            "act": metadata.get("act", ""),
            "scene": metadata.get("scene", ""),
            "speaker": metadata.get("speaker", ""),
            "text": metadata.get("text", ""),
            "scene_summary": metadata.get("scene_summary", ""),
            "keywords": json.loads(metadata.get("keywords") or "[]"),
            "metadata": metadata,
        }


class TfidfRetriever:
    def __init__(self) -> None:
        self.backend = "tfidf-fallback"
        self.vectorizer = SimpleTfidfVectorizer()
        self.chunks: List[Chunk] = []
        self.matrix: np.ndarray | None = None

    def build_index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [chunk.get("embedding_text") or chunk["text"] for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if self.matrix is None:
            raise RuntimeError("Index has not been built.")
        query_vector = self.vectorizer.transform([expand_query(query)])
        scores = (query_vector @ self.matrix.T)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


EmbeddingRetriever = ChromaRetriever
