"""Évolution des probabilités de titre, une entrée par manche."""

import json
from pathlib import Path


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def upsert(history: list[dict], entry: dict) -> list[dict]:
    key = (entry["season"], entry["round"])
    kept = [e for e in history if (e["season"], e["round"]) != key]
    return sorted([*kept, entry], key=lambda e: (e["season"], e["round"]))


def save_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
