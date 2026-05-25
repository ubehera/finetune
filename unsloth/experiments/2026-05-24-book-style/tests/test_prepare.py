"""Plain-assert tests for prepare.py."""
from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from prepare import load_scenes_from_canon, split_train_val, write_jsonl


def _make_mini_canon(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE scenes (
            id TEXT PRIMARY KEY,
            series_id TEXT NOT NULL,
            book INTEGER NOT NULL,
            chapter INTEGER NOT NULL,
            scene INTEGER NOT NULL,
            pov_character TEXT,
            location TEXT,
            beat_id TEXT,
            text TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            judge_score REAL NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            claude_score REAL,
            claude_feedback TEXT,
            UNIQUE(series_id, book, chapter, scene, revision)
        ) STRICT;
    """)
    rows = [
        ("b1:c1:s1:r1", "ab", 1, 1, 1, "Mira", "Bridge", "b1.1.1.1", "ch1 scene1 rev1", 100, 0.7, 1, "t"),
        ("b1:c1:s1:r2", "ab", 1, 1, 1, "Mira", "Bridge", "b1.1.1.1", "ch1 scene1 rev2 LATEST", 110, 0.8, 2, "t"),
        ("b1:c1:s2", "ab", 1, 1, 2, "Voss", "CIC",    "b1.1.1.2", "ch1 scene2 text",  120, 0.7, 1, "t"),
        ("b1:c2:s1", "ab", 1, 2, 1, "Mira", "Hangar", "b1.1.2.1", "ch2 scene1 text",  130, 0.7, 1, "t"),
        ("b1:c3:s1", "ab", 1, 3, 1, "Voss", "Mess",   "b1.1.3.1", "ch3 scene1 text",  140, 0.7, 1, "t"),
        ("b2:c1:s1", "ab", 2, 1, 1, "X",    "Y",      "b2.1.1.1", "wrong book",       100, 0.7, 1, "t"),
    ]
    conn.executemany(
        "INSERT INTO scenes(id, series_id, book, chapter, scene, pov_character, location, "
        "beat_id, text, word_count, judge_score, revision, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_load_scenes_picks_latest_revision_per_scene_and_filters_book():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = pathlib.Path(tmpdir) / "canon.sqlite"
        _make_mini_canon(str(db_path))
        scenes = load_scenes_from_canon(str(db_path), book=1)
        assert len(scenes) == 4, f"expected 4 scenes, got {len(scenes)}"
        ch1_s1 = [s for s in scenes if s["chapter"] == 1 and s["scene"] == 1][0]
        assert ch1_s1["text"] == "ch1 scene1 rev2 LATEST", f"got: {ch1_s1['text']}"
        assert not any(s.get("book") == 2 for s in scenes)


def test_split_train_val_holds_out_val_chapter():
    scenes = [
        {"chapter": 1, "scene": 1}, {"chapter": 1, "scene": 2},
        {"chapter": 2, "scene": 1},
        {"chapter": 3, "scene": 1},
    ]
    train, val = split_train_val(scenes, val_chapter=3)
    assert len(train) == 3 and all(s["chapter"] != 3 for s in train)
    assert len(val) == 1 and all(s["chapter"] == 3 for s in val)


def test_write_jsonl_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        scenes = [
            {"book": 1, "chapter": 1, "scene": 1, "pov_character": "Mira",
             "location": "Bridge", "text": "scene text"},
        ]
        out = pathlib.Path(tmpdir) / "out.jsonl"
        write_jsonl(scenes, str(out))
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert "text" in row
        assert row["text"].startswith("[Book 1, Chapter 1, Scene 1")
        assert row["text"].endswith("scene text")


if __name__ == "__main__":
    test_load_scenes_picks_latest_revision_per_scene_and_filters_book()
    test_split_train_val_holds_out_val_chapter()
    test_write_jsonl_roundtrip()
    print("OK: all prepare.py tests passed")
