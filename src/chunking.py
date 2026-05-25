from __future__ import annotations

import re
from typing import Any, Dict, List

from config import CHUNK_OVERLAP, CHUNK_WORDS


Record = Dict[str, Any]
Chunk = Dict[str, Any]


def _get_text(record: Record) -> str:
    for key in ["text", "utterance", "excerpt", "content", "passage"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    parts = []
    for key in ["speaker", "summary", "modern_summary"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    return " ".join(parts).strip()


def _words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def _window_text(text: str, size: int, overlap: int) -> List[str]:
    words = _words(text)
    if len(words) <= size:
        return [text.strip()]

    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + size]).strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(words):
            break
    return chunks


def _embedding_text(record: Record, text: str) -> str:
    summary = record.get("scene_summary") or record.get("modern_summary") or ""
    keywords = ", ".join(record.get("keywords", [])) if isinstance(record.get("keywords"), list) else ""
    return f"{record.get('play', '')} Act {record.get('act', '')} Scene {record.get('scene', '')}. {summary} {keywords}. {text}"


def create_chunks(records: List[Record]) -> List[Chunk]:
    chunks: List[Chunk] = []

    for i, record in enumerate(records):
        text = _get_text(record)
        if not text:
            continue

        pieces = _window_text(text, CHUNK_WORDS, CHUNK_OVERLAP)
        for j, piece in enumerate(pieces):
            chunk_id = record.get("source_id") or record.get("scene_id") or record.get("id") or f"chunk_{i:06d}"
            chunks.append({
                "chunk_id": f"{chunk_id}_{j + 1:02d}" if len(pieces) > 1 else chunk_id,
                "play": record.get("play", record.get("play_key", "unknown")),
                "act": record.get("act"),
                "scene": record.get("scene"),
                "speaker": record.get("speaker", "MIXED"),
                "text": piece,
                "embedding_text": _embedding_text(record, piece),
                "scene_summary": record.get("scene_summary", ""),
                "keywords": record.get("keywords", []),
                "metadata": record,
            })

    return chunks


def format_chunk_for_display(chunk: Chunk) -> str:
    play = chunk.get("play", "Unknown play")
    act = chunk.get("act", "?")
    scene = chunk.get("scene", "?")
    speaker = chunk.get("speaker", "")

    header = f"{play}, Act {act}, Scene {scene}"
    if speaker:
        header += f", Speaker: {speaker}"

    return f"[{header}]\n{chunk.get('text', '')}"
