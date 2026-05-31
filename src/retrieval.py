from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from chunking import create_chunks
from config import CHROMA_DIR, EMBEDDING_MODEL_NAME, INDEX_PATH
from data_loader import load_all_plays


Chunk = Dict[str, Any]


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
        self.collection_name = f"shakespeare_{self._safe_name(embedding_model_name)}"
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(self.collection_name)
        try:
            self.model = SentenceTransformer(embedding_model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model '{embedding_model_name}'. "
                "Check your internet connection, or download/cache the model before running the system."
            ) from exc

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

        # Expand query and generate a normalized vector embedding for semantic search
        query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()[0]

        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        chunks = []
        for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
            chunk = self._chunk(metadata)
            score = 1.0 / (1.0 + float(distance))
            chunks.append((chunk, score))
        return chunks

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "persist_dir": str(self.persist_dir),
            "embedding_model_name": self.embedding_model_name,
            "collection_name": self.collection_name,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, embedding_model_name: str | None = None) -> "ChromaRetriever":
        from config import EMBEDDING_MODEL_NAME

        data = json.loads(path.read_text(encoding="utf-8"))
        model_name = embedding_model_name or data.get("embedding_model_name") or EMBEDDING_MODEL_NAME
        return cls(Path(data["persist_dir"]), model_name)

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

    def count(self) -> int:
        return int(self.collection.count())

    def _safe_name(self, value: str) -> str:
        name = re.sub(r"[^A-Za-z0-9_]+", "_", value.lower()).strip("_")
        return name[:50] or "default"


EmbeddingRetriever = ChromaRetriever


def build_retriever() -> EmbeddingRetriever:
    if INDEX_PATH.exists():
        retriever = EmbeddingRetriever.load(INDEX_PATH, EMBEDDING_MODEL_NAME)
        if retriever.count() > 0:
            return retriever

    records = load_all_plays()
    chunks = create_chunks(records)
    retriever = EmbeddingRetriever(CHROMA_DIR, EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)
    retriever.save(INDEX_PATH)
    return retriever
