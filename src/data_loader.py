from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from config import PLAY_FILES, PROCESSED_DATA_DIR, RAW_DATA_DIR


Record = Dict[str, Any]


def _extract_records(obj: Any) -> List[Record]:
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["records", "utterances", "scenes", "chunks", "data"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

    raise ValueError(
        "Could not extract records. Expected a list or a dictionary containing "
        "one of: records, utterances, scenes, chunks, data."
    )


def _load_jsonl(path: Path) -> List[Record]:
    rows: List[Record] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json_records(path: Path) -> List[Record]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    return _extract_records(obj)


def load_all_plays() -> List[Record]:
    all_records: List[Record] = []

    for play_key, path in PLAY_FILES.items():
        processed = PROCESSED_DATA_DIR / f"{play_key}.json"
        scene_jsonl = RAW_DATA_DIR / f"{play_key}_scene_chunks.jsonl"

        if processed.exists():
            records = load_json_records(processed)
        elif path.exists():
            records = load_json_records(path)
        elif scene_jsonl.exists():
            records = _load_jsonl(scene_jsonl)
        else:
            raise FileNotFoundError(f"No dataset file found for {play_key}")

        for r in records:
            r.setdefault("play_key", play_key)
        all_records.extend(records)

    return all_records


if __name__ == "__main__":
    records = load_all_plays()
    print(f"Loaded {len(records)} records.")
    print("First record:")
    preview = {k: records[0].get(k) for k in ["play", "act", "scene", "scene_id", "scene_summary", "keywords"]}
    print(json.dumps(preview, indent=2, ensure_ascii=False))
