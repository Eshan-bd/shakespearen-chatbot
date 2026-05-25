from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


class ChromaRetriever:
    def __init__(self, persist_dir: Path, embedding_model_name: str):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is required. Install it with: pip install chromadb") from exc

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for Chroma embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.persist_dir = Path(persist_dir)
        self.embedding_model_name = embedding_model_name
        self.backend = "chromadb"
        self.collection_name = "shakespeare_chunks"
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.model = SentenceTransformer(embedding_model_name)

    def build_index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks supplied to build_index().")

        shutil.rmtree(self.persist_dir, ignore_errors=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        import chromadb

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(self.collection_name)

        ids = [chunk["chunk_id"] for chunk in chunks]
        docs = [chunk.get("embedding_text") or chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(docs, show_progress_bar=True, normalize_embeddings=True).tolist()
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
        query_embedding = self.model.encode([expand_query(query)], normalize_embeddings=True).tolist()[0]
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        chunks = []
        for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
            chunk = self._chunk(metadata)
            score = 1.0 / (1.0 + float(distance))
            chunks.append((chunk, score))
        return chunks

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "persist_dir": self.persist_dir,
                "embedding_model_name": self.embedding_model_name,
            }, f)

    @classmethod
    def load(cls, path: Path, embedding_model_name: str | None = None) -> "ChromaRetriever":
        from config import EMBEDDING_MODEL_NAME

        with path.open("rb") as f:
            data = pickle.load(f)
        model_name = embedding_model_name or data.get("embedding_model_name") or EMBEDDING_MODEL_NAME
        return cls(data["persist_dir"], model_name)

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


EmbeddingRetriever = ChromaRetriever
