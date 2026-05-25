from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Sequence, Tuple

from config import CHROMA_DIR, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, INDEX_PATH, OLLAMA_MODEL, PROMPT_DIR
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


Chunk = Dict[str, Any]

STYLE_INTENTS = [
    "write a creative Shakespearean style response",
    "generate a stylised Elizabethan sounding answer",
    "reply poetically in Shakespeare's voice",
    "compose a short dramatic speech rather than factual evidence",
]

STYLE_TOPICS = {
    "Juliet's divided heart": [
        "Juliet is conflicted between love for Romeo and loyalty to the Capulet family",
        "a young woman loves someone from the enemy family",
    ],
    "Macbeth's troubled ambition": [
        "Macbeth is driven by ambition, prophecy, murder, guilt, and the crown",
        "a warrior is tempted to seize power and suffers for it",
    ],
    "Hamlet's grief and doubt": [
        "Hamlet grieves his father, doubts Claudius, and struggles with revenge",
        "a prince hesitates while thinking about death, proof, and revenge",
    ],
}


def load_system_prompt() -> str:
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def build_rag_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    system_prompt = load_system_prompt()

    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Context {rank} | similarity={score:.4f}]\n"
            f"{format_chunk_for_display(chunk)}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""{system_prompt}

    Retrieved context:
    {context}

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
    return _semantic_score(query, STYLE_INTENTS, model) >= 0.38


def _style_topic(query: str, retrieved: List[Tuple[Chunk, float]], model: Any) -> str:
    evidence_text = " ".join(chunk.get("scene_summary") or chunk.get("text", "")[:300] for chunk, _ in retrieved[:3])
    search_text = f"{query} {evidence_text}"
    scores = {
        topic: _semantic_score(search_text, phrases, model)
        for topic, phrases in STYLE_TOPICS.items()
    }
    topic, score = max(scores.items(), key=lambda item: item[1])
    return topic if score >= 0.30 else "this matter"


def _stylised_answer(query: str, retrieved: List[Tuple[Chunk, float]], model: Any) -> str:
    topic = _style_topic(query, retrieved, model)
    return (
        "Creative stylised response, not evidence:\n"
        f"O, {topic}, where love and duty pull one soul in twain. "
        "My heart would speak plain truth, yet fear and honour bind my tongue. "
        "If joy be born from peril, then let wisdom guide it, lest sweet desire "
        "turn bitter by the morning."
    )


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
    answer = []
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    if _is_style_request(query, embedding_model):
        return _stylised_answer(query, retrieved, embedding_model)
    else:
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
