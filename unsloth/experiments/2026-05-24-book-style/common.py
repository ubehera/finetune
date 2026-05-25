"""Shared formatting helpers for the book-style LoRA experiment.

Keep this small: header/scene formatting only. Anything touching tokenizers,
datasets, models, or the canon DB belongs in prepare/train/evaluate.
"""
from __future__ import annotations


def format_scene_header(
    *, book: int, chapter: int, scene: int,
    pov: str | None, location: str | None,
) -> str:
    pov_s = pov if pov else "unknown"
    loc_s = location if location else "unknown"
    return f"[Book {book}, Chapter {chapter}, Scene {scene} — POV: {pov_s}, Location: {loc_s}]"


def format_scene(
    *, book: int, chapter: int, scene: int,
    pov: str | None, location: str | None,
    text: str,
) -> str:
    header = format_scene_header(book=book, chapter=chapter, scene=scene, pov=pov, location=location)
    return f"{header}\n\n{text}"
