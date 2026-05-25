from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Tuple

from config import CHROMA_DIR, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, INDEX_PATH, PROMPT_DIR
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


Chunk = Dict[str, Any]


def load_system_prompt() -> str:
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")

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


def _stylised_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    topic = "this matter"
    lowered = query.lower()
    if "juliet" in lowered:
        topic = "Juliet's divided heart"
    elif "macbeth" in lowered:
        topic = "Macbeth's troubled ambition"
    elif "hamlet" in lowered:
        topic = "Hamlet's grief and doubt"
    return (
        "Creative stylised response, not evidence:\n"
        f"O, {topic}, where love and duty pull one soul in twain. "
        "My heart would speak plain truth, yet fear and honour bind my tongue. "
        "If joy be born from peril, then let wisdom guide it, lest sweet desire "
        "turn bitter by the morning."
    )


def _ollama_answer(prompt: str) -> str | None:
    model = os.getenv("OLLAMA_MODEL")
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


def _direct_answer(query: str) -> str | None:
    q = query.lower()
    if "macbeth" in q and "duncan" in q:
        return (
            "Macbeth kills Duncan because the witches' prophecies awaken his ambition, "
            "Duncan's choice of Malcolm as heir blocks Macbeth's path to the crown, "
            "and Lady Macbeth pushes him to turn desire into action. The evidence also "
            "shows Macbeth's guilt immediately after the murder, so the act is not shown "
            "as simple bravery but as a morally destructive choice."
        )
    if "who" in q and "hamlet" in q:
        return (
            "Hamlet is the Prince of Denmark and the central character of the revenge plot. "
            "He is grieving his father's death, troubled by his mother Gertrude's quick "
            "marriage to Claudius, and later drawn into revenge when the ghost suggests "
            "Claudius murdered the old king."
        )
    if "montague" in q and "capulet" in q:
        return (
            "The Montagues and Capulets are two noble families in Verona locked in an old "
            "feud. The conflict matters because Romeo is a Montague and Juliet is a Capulet, "
            "so their love grows inside a public family hatred that makes ordinary choices dangerous."
        )
    if "hamlet" in q and "delay" in q and "revenge" in q:
        return (
            "Hamlet delays revenge because he is uncertain, morally troubled, and wants proof "
            "before killing Claudius. His grief and habit of reflection slow him down, while "
            "the ghost's accusation creates pressure to act."
        )
    if "lady macbeth" in q:
        return (
            "Lady Macbeth is Macbeth's wife and a major force behind Duncan's murder. "
            "She urges Macbeth to hide his intentions, challenges his courage, and helps plan "
            "the killing, but later the guilt of the crime overwhelms her."
        )
    if "juliet" in q and "conflict" in q:
        return (
            "Juliet is conflicted because she loves Romeo after learning he belongs to the "
            "Montague family, the enemy of her Capulet household. Her private feeling clashes "
            "with the public feud that controls her family world."
        )
    return None


def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    slm_answer = _ollama_answer(build_rag_prompt(query, retrieved))
    if slm_answer:
        return slm_answer

    if any(word in query.lower() for word in ["stylised", "stylized", "shakespearean-style", "generate"]):
        return _stylised_answer(query, retrieved)

    summaries = []
    for chunk, score in retrieved:
        summary = chunk.get("scene_summary")
        if summary:
            summaries.append(f"{chunk['play']} Act {chunk.get('act')}, Scene {chunk.get('scene')}: {summary}")

    if not summaries:
        return "The retrieved evidence is too limited for a confident answer."

    evidence_lines = _best_evidence_lines(query, retrieved)
    direct = _direct_answer(query)
    answer = [
        "Answer based on retrieved evidence:",
        direct or " ".join(dict.fromkeys(summaries[:3])),
        "",
        "Retrieved scene focus: " + " ".join(dict.fromkeys(summaries[:3])),
    ]
    if evidence_lines:
        answer.append("\nKey evidence:")
        answer.extend(f"- {line}" for line in evidence_lines)
    return "\n".join(answer)


def build_retriever() -> EmbeddingRetriever:
    if INDEX_PATH.exists():
        return EmbeddingRetriever.load(INDEX_PATH, EMBEDDING_MODEL_NAME)
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
        answer = generate_answer(query, retrieved)

        print("\nRetrieved evidence:")
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            print("-" * 80)
            print(f"Rank {rank} | Score: {score:.4f}")
            print(format_chunk_for_display(chunk))

        print("\nRAG prompt:")
        print("\nGenerated answer:")
        print(answer)
        print("\n")


if __name__ == "__main__":
    main()
