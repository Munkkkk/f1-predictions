from f1pred.history import load_history, save_history, upsert


def entry(season, rnd, p=0.5):
    return {"season": season, "round": rnd, "race_name": f"GP {rnd}",
            "probabilities": {"a": p, "b": 1 - p}}


def test_missing_file_gives_empty_history(tmp_path):
    assert load_history(tmp_path / "history.json") == []


def test_upsert_is_idempotent_and_sorted():
    h = upsert([], entry(2026, 3))
    h = upsert(h, entry(2026, 1))
    h = upsert(h, entry(2026, 3, p=0.9))
    assert [(e["season"], e["round"]) for e in h] == [(2026, 1), (2026, 3)]
    assert h[1]["probabilities"]["a"] == 0.9


def test_roundtrip(tmp_path):
    path = tmp_path / "history.json"
    save_history(path, [entry(2026, 1)])
    assert load_history(path) == [entry(2026, 1)]
