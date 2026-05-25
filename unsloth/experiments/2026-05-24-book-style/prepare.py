"""Build train/val JSONL files from the scriptorium canon sqlite.

Reads the latest revision per (book, chapter, scene) for a given book,
applies common.format_scene to add the metadata header, writes one
{"text": "<formatted scene>"} row per scene into train.jsonl/val.jsonl.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
from typing import Sequence

from common import format_scene


def load_scenes_from_canon(db_path: str, book: int) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("""
            SELECT s.book, s.chapter, s.scene, s.pov_character, s.location, s.text
            FROM scenes s
            JOIN (
                SELECT series_id, book, chapter, scene, MAX(revision) AS max_rev
                FROM scenes
                WHERE book = ?
                GROUP BY series_id, book, chapter, scene
            ) m
              ON s.series_id = m.series_id
             AND s.book      = m.book
             AND s.chapter   = m.chapter
             AND s.scene     = m.scene
             AND s.revision  = m.max_rev
            WHERE s.book = ?
            ORDER BY s.chapter, s.scene
        """, (book, book))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def split_train_val(scenes: Sequence[dict], val_chapter: int) -> tuple[list[dict], list[dict]]:
    train = [s for s in scenes if s["chapter"] != val_chapter]
    val   = [s for s in scenes if s["chapter"] == val_chapter]
    return train, val


def write_jsonl(scenes: Sequence[dict], path: str) -> None:
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in scenes:
            text = format_scene(
                book=int(s["book"]), chapter=int(s["chapter"]), scene=int(s["scene"]),
                pov=s.get("pov_character"), location=s.get("location"),
                text=s["text"],
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def main(args: argparse.Namespace) -> int:
    canon = str(pathlib.Path(args.canon_db).expanduser())
    out_dir = pathlib.Path(args.out).expanduser()
    scenes = load_scenes_from_canon(canon, book=args.book)
    train, val = split_train_val(scenes, val_chapter=args.val_chapter)
    write_jsonl(train, str(out_dir / "train.jsonl"))
    write_jsonl(val,   str(out_dir / "val.jsonl"))
    print(f"[prepare] book={args.book} canon={canon}")
    print(f"[prepare] train: {len(train)} scenes -> {out_dir/'train.jsonl'}")
    print(f"[prepare] val:   {len(val)} scenes -> {out_dir/'val.jsonl'}")
    return 0
