from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Sequence, Tuple

from config import CHROMA_DIR, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, INDEX_PATH, OLLAMA_MODEL, PROMPT_DIR, SEMANTIC_SIMILARITY_THRESHOLD
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever, build_retriever


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


def _best_evidence_lines(query: str, retrieved: List[Tuple[Chunk, float]], limit: int = 3) -> List[str]:
    q_terms = {w.lower() for w in re.findall(r"[A-Za-z']+", query) if len(w) > 3}
    candidates = []
    for chunk, _ in retrieved:
        for sent in _sentences(chunk["text"]):
            if len(sent) < 60 or re.fullmatch(r"[\[A-Z .:;'-]+", sent):
                continue
            terms = {w.lower() for w in re.findall(r"[A-Za-z']+", sent)}
            candidates.append((len(q_terms & terms), sent, chunk))
    candidates.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for _, sent, chunk in candidates[:limit]:
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
        return generate_evidence_answer(query, retrieved)

def generate_stylised_answer(query, retrieved):
    slm_answer = _ollama_answer(build_stylised_prompt(query, retrieved))
    if slm_answer:
        return slm_answer


def generate_evidence_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
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

    evidence_lines = _best_evidence_lines(query, retrieved)
    answer += [
        " ".join(dict.fromkeys(summaries[:3])),
        "",
        "Scene in focus: " + " ".join(dict.fromkeys(summaries[:3])),
    ]
    if evidence_lines:
        answer.append("\nKey evidence:")
        answer.extend(f"- {line}" for line in evidence_lines)
    return "\n".join(answer)


def main() -> None:
    retriever = build_retriever()

    print(f"Shakespeare-aware RAG chatbot. Retriever backend: {retriever.backend}")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        retrieved = retriever.retrieve(query, top_k=DEFAULT_TOP_K)
        answer = generate_answer(query, retrieved, retriever.model)

        print("\n")
        print(answer)
        print("\n")


if __name__ == "__main__":
    main()
