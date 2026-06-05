from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Sequence, Tuple

from config import CHROMA_DIR, DEFAULT_TOP_K_RETRIEVES, EMBEDDING_MODEL_NAME, INDEX_PATH, OLLAMA_MODEL, PROMPT_DIR, SEMANTIC_SIMILARITY_THRESHOLD
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever, build_retriever, get_retriever


Chunk = Dict[str, Any]

STYLE_INTENTS = [
    "Shakespearean tone",
    "Shakespearean style",
    "Shakespearean voice",
    "Shakespearean language",
    "Elizabethan tone",
    "Elizabethan style",
    "Elizabethan voice",
    "bard style",
    "bard voice",
    "poetic diction",
    "archaic diction",
    "elevated diction",
    "theatrical language",
    "dramatic verse",
    "dramatic monologue",
    "early modern English",
    "old English style",
    "thou thee language",
    "creative response",
    "stylised response",
]

def load_system_prompt() -> str:
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")

def load_stylised_prompt() -> str:
    prompt_path = PROMPT_DIR / "stylised_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")

def parse_context(retrieved):
    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Context {rank} | similarity={score:.4f}]\n"
            f"{format_chunk_for_display(chunk)}"
        )

    return "\n\n".join(context_blocks)

def build_stylised_prompt(query, retrieved):
    prompt = load_stylised_prompt()

    prompt = f"""{prompt}


    User request:
    {query}

    Retrieved context for inspiration only:
    {parse_context(retrieved)}

    Response:
    """

    return prompt


def build_rag_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    prompt = load_system_prompt()


    prompt = f"""{prompt}

    Retrieved context:
    {parse_context(retrieved)}

    User question:
    {query}

    Answer:
    """
    return prompt


def _sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return re.split(r"(?<=[.!?])\s+", cleaned)


def _best_evidence_lines(query: str, retrieved: List[Tuple[Chunk, float]], model: Any, limit: int = 3) -> List[str]:
    candidates = []
    for chunk, _ in retrieved:
        for sent in _sentences(chunk["text"]):
            sent = sent.strip()
            if len(sent) < 60 or sent.isupper():
                continue
            candidates.append((sent, chunk))

    if candidates:
        texts = [query, *[sent for sent, _ in candidates]]
        embeddings = model.encode(texts, normalize_embeddings=True)
        query_embedding = embeddings[0]
        scored = [
            (_cosine(query_embedding, sentence_embedding), sent, chunk)
            for sentence_embedding, (sent, chunk) in zip(embeddings[1:], candidates)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
    else:
        scored = []

    lines = []
    for _, sent, chunk in scored[:limit]:
        lines.append(f"{chunk['play']} {chunk.get('act')}.{chunk.get('scene')}: {sent[:260]}")
    if not lines:
        for chunk, _ in retrieved[:limit]:
            excerpt = re.sub(r"\s+", " ", chunk["text"]).strip()[:260]
            lines.append(f"{chunk['play']} {chunk.get('act')}.{chunk.get('scene')}: {excerpt}")
    return lines


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _semantic_score(query: str, phrases: List[str], model: Any) -> float:
    embeddings = model.encode([query, *phrases], normalize_embeddings=True)
    query_embedding = embeddings[0]
    return max(_cosine(query_embedding, phrase_embedding) for phrase_embedding in embeddings[1:])


def _is_style_request(query: str, model: Any) -> bool:
    return _semantic_score(query, STYLE_INTENTS, model) >= SEMANTIC_SIMILARITY_THRESHOLD


def _ollama_answer(prompt: str) -> str | None:
    model = OLLAMA_MODEL
    if not model:
        return None
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("response", "").strip() or None
    except Exception:
        return None


def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]], embedding_model: Any | None = None) -> str:
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    if _is_style_request(query, embedding_model):
        return generate_stylised_answer(query, retrieved)
    else:
        return generate_evidence_answer(query, retrieved, embedding_model)

def generate_stylised_answer(query, retrieved):
    slm_answer = _ollama_answer(build_stylised_prompt(query, retrieved))
    if slm_answer:
        return slm_answer


def generate_evidence_answer(query: str, retrieved: List[Tuple[Chunk, float]], embedding_model: Any) -> str:
    answer = []

    slm_answer = _ollama_answer(build_rag_prompt(query, retrieved))
    if slm_answer:
        answer.append(slm_answer)


    summaries = []
    for chunk, score in retrieved:
        summary = chunk.get("scene_summary")
        if summary:
            summaries.append(f"{chunk['play']} Act {chunk.get('act')}, Scene {chunk.get('scene')}: {summary}")

    if not summaries:
        return "The retrieved evidence is too limited for a confident answer."

    evidence_lines = _best_evidence_lines(query, retrieved, embedding_model)
    answer += [
        " ".join(dict.fromkeys(summaries[:3])),
        "",
        "Scene in focus: " + " ".join(dict.fromkeys(summaries[:3])),
    ]
    if evidence_lines:
        answer.append("\nKey evidence:")
        answer.extend(f"- {line}" for line in evidence_lines)
    return "\n".join(answer)

class RAGChatbot:
    def __init__(self):
        self.retriever = get_retriever()

    def query(self, p_query):
        query = p_query.strip()

        self.retrieved = self.retriever.retrieve(query, top_k=DEFAULT_TOP_K_RETRIEVES)
        answer = generate_answer(query, self.retrieved, self.retriever.model)

        return answer

    def get_retriever(self):
        return self.retriever



def main() -> None:
    bot = RAGChatbot()
    print(f"Shakespeare-aware RAG chatbot. Retriever backend: {bot.get_retriever().backend}")
    print("Type 'quit' to exit.\n")
    while True:
        query = input("Query: ")
        if query.lower() in {"quit", "exit"}:
            break

        print("\n")
        print(bot.query(query))
        print("\n")

if __name__ == "__main__":
    main()