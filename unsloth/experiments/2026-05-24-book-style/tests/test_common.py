"""Plain-assert tests for common.py. Run with:

    cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style
    uv run --project ../.. python tests/test_common.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import format_scene_header, format_scene


def test_header_format_minimal():
    h = format_scene_header(book=1, chapter=5, scene=3, pov="Mira Cho", location="Hangar 4-B")
    assert h == "[Book 1, Chapter 5, Scene 3 — POV: Mira Cho, Location: Hangar 4-B]"


def test_header_handles_missing_pov_or_location():
    h = format_scene_header(book=1, chapter=1, scene=1, pov=None, location=None)
    assert h == "[Book 1, Chapter 1, Scene 1 — POV: unknown, Location: unknown]"


def test_scene_format_full():
    s = format_scene(
        book=1, chapter=2, scene=4,
        pov="Lt. Voss",
        location="CIC, USS Tartarus",
        text="The klaxons cut off at zero-three-twelve.\n\nVoss didn't move.",
    )
    expected = (
        "[Book 1, Chapter 2, Scene 4 — POV: Lt. Voss, Location: CIC, USS Tartarus]\n"
        "\n"
        "The klaxons cut off at zero-three-twelve.\n"
        "\n"
        "Voss didn't move."
    )
    assert s == expected


if __name__ == "__main__":
    test_header_format_minimal()
    test_header_handles_missing_pov_or_location()
    test_scene_format_full()
    print("OK: all common.py tests passed")
