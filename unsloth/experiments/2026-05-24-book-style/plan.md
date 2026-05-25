# Book-Style LoRA Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a self-contained LoRA fine-tune of `Qwen3-4B` on Book 1 of `fiction_corpus`, verify training had effect via held-out perplexity + 3 generation samples, all on the existing `~/projects/finetune/unsloth/` native install (single GB10 node).

**Architecture:** New subdir `experiments/2026-05-24-book-style/` under the existing `finetune/unsloth/` install. Multi-file Python project: thin CLI dispatcher (`lora_test.py`) → focused modules (`common.py`, `prepare.py`, `train.py`, `evaluate.py`). Reuses parent's venv via `uv run --project ../..`, inheriting all five GB10/sm_121 hotfixes. Continued pretraining (raw CLM, not SFT/chat), packed 1024-token sequences, 5 epochs, LoRA r=16.

**Tech Stack:** Python 3.12, Unsloth (existing install), transformers/peft/trl/datasets, sqlite3 (stdlib), plain assert-based test scripts (no pytest dep added to parent).

---

## File Structure

| File | Purpose |
|------|---------|
| `README.md` | One-screen "what is this and how to run" |
| `design.md` | (already written) the validated spec |
| `plan.md` | (this file) the implementation plan |
| `run.sh` | uv-run wrapper, mirrors `../../train.sh` pattern |
| `lora_test.py` | Thin CLI dispatcher (argparse → subcommand main) |
| `common.py` | Header formatting + scene-formatting helpers |
| `prepare.py` | sqlite canon → train/val JSONL |
| `train.py` | Model load + LoRA setup + training loop + stats |
| `evaluate.py` | PPL computation + greedy sample generation |
| `tests/test_common.py` | Header/scene format assertions |
| `tests/test_prepare.py` | In-memory sqlite → JSONL correctness |
| `tests/test_evaluate.py` | PPL math + sample-prompt construction |
| `tests/verify_ppl.py` | Standalone script that reads `results/ppl_table.json` and prints PASS/FAIL vs the success bar |
| `data/`, `lora-output/`, `results/` | Generated at runtime, gitignored |

---

## Task 1: Scaffold + git init + run.sh + README

**Files:**
- Create: `~/projects/finetune/.gitignore`
- Create: `~/projects/finetune/unsloth/experiments/2026-05-24-book-style/README.md`
- Create: `~/projects/finetune/unsloth/experiments/2026-05-24-book-style/run.sh`
- Create: `tests/` directory
- Init: `~/projects/finetune/.git`

- [ ] **Step 1.1: Create subdirs**

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style
mkdir -p tests data lora-output results/samples
```

- [ ] **Step 1.2: Write `.gitignore`**

Path: `~/projects/finetune/.gitignore`

```gitignore
# venvs (one per framework subdir)
*/.venv/
**/.venv/

# build artifacts
flash-attn-src/build/
flash-attn-src/*.egg-info/

# experiment outputs (preserve dirs via .gitkeep, ignore contents)
unsloth/experiments/*/data/*.jsonl
unsloth/experiments/*/lora-output/*/
unsloth/experiments/*/results/ppl_table.json
unsloth/experiments/*/results/samples/*.txt

# HF caches
**/hf_cache/

# Python
__pycache__/
*.pyc

# OS
.DS_Store
```

- [ ] **Step 1.3: Add `.gitkeep` files to preserve empty experiment dirs**

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style
touch data/.gitkeep lora-output/.gitkeep results/.gitkeep results/samples/.gitkeep
```

- [ ] **Step 1.4: Write `run.sh`**

Path: `~/projects/finetune/unsloth/experiments/2026-05-24-book-style/run.sh`

```bash
#!/usr/bin/env bash
# Thin wrapper: sets PYTHONPATH for the torchvision stub, then runs lora_test.py
# via uv inside the parent project's venv.
#
# Usage:
#   ./run.sh prepare [--canon-db PATH]
#   ./run.sh train   [--epochs N] [--lr 1e-4]
#   ./run.sh eval    [--epoch best]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
export PYTHONPATH="$ROOT/stubs:${PYTHONPATH:-}"
exec uv run --project "$ROOT" python "$DIR/lora_test.py" "$@"
```

```bash
chmod +x ~/projects/finetune/unsloth/experiments/2026-05-24-book-style/run.sh
```

- [ ] **Step 1.5: Write `README.md`**

Path: `~/projects/finetune/unsloth/experiments/2026-05-24-book-style/README.md`

```markdown
# Book-Style LoRA Test (2026-05-24)

Continued-pretraining LoRA fine-tune of Qwen3-4B on `fiction_corpus` Book 1,
to validate the GB10 native Unsloth install end-to-end.

**See `design.md` for the full spec and `plan.md` for the build sequence.**

## Quick run

    ./run.sh prepare
    ./run.sh train
    ./run.sh eval

Inspect `results/ppl_table.json` and `results/samples/sample_{1..3}_{base,lora}.txt`.

## Tests

    uv run --project ../.. python tests/test_common.py
    uv run --project ../.. python tests/test_prepare.py
    uv run --project ../.. python tests/test_evaluate.py

## Requires

- Parent `../../` venv populated by `uv sync` and patched by `../../apply-hotfixes.sh`.
- Read access to `~/projects/scriptorium/canon/fiction_corpus.sqlite`.
- One GB10 node (target: node-2).
```

- [ ] **Step 1.6: git init + initial commit**

```bash
cd ~/projects/finetune
git init -b main
git add .gitignore unsloth/experiments/2026-05-24-book-style/
git status
git commit -m "Scaffold book-style LoRA experiment

- Add .gitignore for finetune/ tree (venvs, experiment outputs, caches)
- Create experiments/2026-05-24-book-style/ with design, plan, README, run.sh
- Empty data/, lora-output/, results/ kept via .gitkeep"
```

Expected: clean tree after commit, one entry in `git log --oneline`.

---

## Task 2: `common.py` header formatter (TDD)

**Files:**
- Create: `tests/test_common.py`
- Create: `common.py`

- [ ] **Step 2.1: Write the failing test**

Path: `tests/test_common.py`

```python
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
```

- [ ] **Step 2.2: Run the test, confirm it fails**

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style
uv run --project ../.. python tests/test_common.py
```

Expected: `ModuleNotFoundError: No module named 'common'`

- [ ] **Step 2.3: Implement `common.py`**

Path: `common.py`

```python
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
```

- [ ] **Step 2.4: Run the test, confirm it passes**

```bash
uv run --project ../.. python tests/test_common.py
```

Expected: `OK: all common.py tests passed`

- [ ] **Step 2.5: Commit**

```bash
cd ~/projects/finetune
git add unsloth/experiments/2026-05-24-book-style/common.py \
        unsloth/experiments/2026-05-24-book-style/tests/test_common.py
git commit -m "Add common.py: scene header + scene formatter

format_scene_header() emits the canonical
'[Book {b}, Chapter {c}, Scene {s} — POV: ..., Location: ...]' prefix.
format_scene() composes header + blank line + text. Falls back to
'unknown' for missing pov/location so the header shape stays uniform."
```

---

## Task 3: `prepare.py` (TDD with in-memory sqlite)

**Files:**
- Create: `tests/test_prepare.py`
- Create: `prepare.py`

- [ ] **Step 3.1: Write the failing test**

Path: `tests/test_prepare.py`

```python
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
        ("b1:c1:s1", "ab", 1, 1, 1, "Mira", "Bridge", "b1.1.1.1", "ch1 scene1 rev1", 100, 0.7, 1, "t"),
        ("b1:c1:s1", "ab", 1, 1, 1, "Mira", "Bridge", "b1.1.1.1", "ch1 scene1 rev2 LATEST", 110, 0.8, 2, "t"),
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
```

- [ ] **Step 3.2: Run the test, confirm it fails**

```bash
uv run --project ../.. python tests/test_prepare.py
```

Expected: `ModuleNotFoundError: No module named 'prepare'`

- [ ] **Step 3.3: Implement `prepare.py`**

Path: `prepare.py`

```python
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
```

- [ ] **Step 3.4: Run the test, confirm it passes**

```bash
uv run --project ../.. python tests/test_prepare.py
```

Expected: `OK: all prepare.py tests passed`

- [ ] **Step 3.5: Commit**

```bash
cd ~/projects/finetune
git add unsloth/experiments/2026-05-24-book-style/prepare.py \
        unsloth/experiments/2026-05-24-book-style/tests/test_prepare.py
git commit -m "Add prepare.py: canon sqlite -> train/val JSONL

load_scenes_from_canon picks latest revision per (book,chapter,scene)
via JOIN on MAX(revision). split_train_val holds out one chapter as
validation. write_jsonl applies common.format_scene. CLI main wires it
together for the 'prepare' subcommand."
```

---

## Task 4: `lora_test.py` CLI + `prepare` subcommand integration

**Files:**
- Create: `lora_test.py`

- [ ] **Step 4.1: Implement `lora_test.py` with all three subparsers (train/eval routes fail until later tasks)**

Path: `lora_test.py`

```python
"""Single-entry CLI for the book-style LoRA experiment.

Subcommands:
    prepare   sqlite canon -> data/train.jsonl, data/val.jsonl
    train     LoRA fine-tune Qwen3-4B on data/train.jsonl
    eval      PPL on val + 3 generation samples (base vs LoRA)

Invoked via ./run.sh which sets PYTHONPATH for the torchvision stub.
"""
from __future__ import annotations

import argparse
import pathlib
import sys


DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CANON = pathlib.Path.home() / "projects" / "scriptorium" / "canon" / "fiction_corpus.sqlite"


def _add_prepare(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("prepare", help="Build train.jsonl / val.jsonl from the canon sqlite")
    p.add_argument("--canon-db", default=str(DEFAULT_CANON), help="Path to canon sqlite")
    p.add_argument("--book", type=int, default=1)
    p.add_argument("--val-chapter", type=int, default=10)
    p.add_argument("--out", default=str(DIR / "data"))


def _add_train(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="LoRA fine-tune (see design.md for hyperparams)")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--data", default=str(DIR / "data"))
    p.add_argument("--out", default=str(DIR / "lora-output"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)


def _add_eval(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("eval", help="PPL table + 3 generation samples")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--data", default=str(DIR / "data"))
    p.add_argument("--adapters", default=str(DIR / "lora-output"))
    p.add_argument("--out", default=str(DIR / "results"))
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--gen-tokens", type=int, default=500)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_prepare(sub)
    _add_train(sub)
    _add_eval(sub)
    args = parser.parse_args()

    if args.cmd == "prepare":
        import prepare
        return prepare.main(args)
    if args.cmd == "train":
        import train
        return train.main(args)
    if args.cmd == "eval":
        import evaluate
        return evaluate.main(args)
    parser.error(f"unknown subcommand: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4.2: Manual smoke test of `prepare` on the real canon**

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style
./run.sh prepare
```

Expected:
```
[prepare] book=1 canon=/home/umankb/projects/scriptorium/canon/fiction_corpus.sqlite
[prepare] train: 43 scenes -> data/train.jsonl
[prepare] val:   3 scenes -> data/val.jsonl
```

Inspect:
```bash
wc -l data/*.jsonl
head -1 data/val.jsonl | python -c "import sys, json; print(json.loads(sys.stdin.read())['text'][:200])"
```

Expected: `43 data/train.jsonl`, `3 data/val.jsonl`. Val first row starts with `[Book 1, Chapter 10, Scene 1 — POV: ...`.

- [ ] **Step 4.3: Commit**

```bash
cd ~/projects/finetune
git add unsloth/experiments/2026-05-24-book-style/lora_test.py
git commit -m "Add lora_test.py CLI; wire 'prepare' subcommand

Lazy module imports keep the CLI usable for individual subcommands
without pulling heavy deps until needed. train/eval routes will work
once their modules land in later tasks."
```

---

## Task 5: `train.py` — model load, LoRA, packed CLM training loop

**Files:**
- Create: `train.py`

- [ ] **Step 5.1: Implement `train.py`**

Path: `train.py`

```python
"""LoRA fine-tune Qwen3-4B on packed continued-pretraining data.

Differs from ../../train.py (SFT chat smoke test) in that we:
  - read pre-built JSONL rather than building an in-memory chat dataset,
  - pack tokens into fixed-length windows for raw CLM (no chat template),
  - use transformers.Trainer + DataCollatorForLanguageModeling, not SFTTrainer.

Model loading, LoRA config, and stats logging are cribbed from ../../train.py.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

# Must be imported before transformers so kernel patches apply
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import torch
from datasets import Dataset, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def _pack_text_to_chunks(tokenizer, jsonl_path: str, max_seq_len: int) -> Dataset:
    """Tokenize every row, concatenate with EOS separators, chunk into max_seq_len windows."""
    raw = load_dataset("json", data_files=jsonl_path, split="train")
    eos = tokenizer.eos_token_id

    def tok(batch):
        return tokenizer(batch["text"], add_special_tokens=False, truncation=False)

    tokenized = raw.map(tok, batched=True, remove_columns=raw.column_names)
    all_ids: list[int] = []
    for ids in tokenized["input_ids"]:
        all_ids.extend(ids)
        all_ids.append(eos)
    n_full = (len(all_ids) // max_seq_len) * max_seq_len
    chunks = [all_ids[i : i + max_seq_len] for i in range(0, n_full, max_seq_len)]
    return Dataset.from_dict({"input_ids": chunks, "labels": [list(c) for c in chunks]})


def main(args: argparse.Namespace) -> int:
    out_dir = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = pathlib.Path(args.data).expanduser()

    print(f"[train] model:      {args.model}")
    print(f"[train] out:        {out_dir}")
    print(f"[train] epochs/lr:  {args.epochs} / {args.lr}")
    print(f"[train] lora r/a:   {args.lora_r} / {args.lora_alpha}")
    print(f"[train] device:     {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})")

    torch.cuda.reset_peak_memory_stats()
    t_load_start = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    t_load = time.perf_counter() - t_load_start

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    train_ds = _pack_text_to_chunks(tokenizer, str(data_dir / "train.jsonl"), args.max_seq_len)
    val_ds   = _pack_text_to_chunks(tokenizer, str(data_dir / "val.jsonl"),   args.max_seq_len)
    print(f"[train] train chunks: {len(train_ds)}  val chunks: {len(val_ds)}  seq_len={args.max_seq_len}")

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=3407,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("[train] starting training...")
    t_train_start = time.perf_counter()
    result = trainer.train()
    t_train = time.perf_counter() - t_train_start

    best_dir = out_dir / "best"
    model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    stats = {
        "framework": "unsloth+peft+transformers",
        "model": args.model,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_len": args.max_seq_len,
        "train_chunks": len(train_ds),
        "val_chunks": len(val_ds),
        "load_seconds": round(t_load, 2),
        "train_seconds": round(t_train, 2),
        "peak_gpu_memory_gib": round(peak_mem_gb, 2),
        "final_train_loss": round(result.training_loss, 4),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0
```

- [ ] **Step 5.2: Commit (no run yet — Task 6 does the smoke run)**

```bash
cd ~/projects/finetune
git add unsloth/experiments/2026-05-24-book-style/train.py
git commit -m "Add train.py: continued-pretraining LoRA loop

Cribs model/LoRA/stats from ../../train.py but swaps the SFT chat
dataset for packed CLM on the prepared JSONL. _pack_text_to_chunks
tokenizes each row, concatenates with EOS separators, chunks into
fixed max_seq_len windows. Uses transformers.Trainer +
DataCollatorForLanguageModeling rather than SFTTrainer."
```

---

## Task 6: Train smoke run (1 epoch)

**Files:** none modified — this is a verification task.

- [ ] **Step 6.1: Run a fast smoke pass on node-2**

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style
./run.sh train --epochs 1 --batch-size 2 --grad-accum 1
```

Expected:
- First-time download of `Qwen/Qwen3-4B` (~8 GB) unless already cached.
- `[train] train chunks: ~170  val chunks: ~10  seq_len=1024`
- ~5 min wall-clock after download.
- Printed `stats.json` shows `peak_gpu_memory_gib < 100` and `final_train_loss > 0`.
- `lora-output/best/adapter_config.json` and `adapter_model.safetensors` present.

If OOM: drop `--batch-size 1`.
If a known Unsloth patch error: `cd ../.. && ./apply-hotfixes.sh` then retry.

- [ ] **Step 6.2: Capture smoke stats**

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style
cat lora-output/stats.json
cp lora-output/stats.json results/smoke-stats.json
```

```bash
cd ~/projects/finetune
git add unsloth/experiments/2026-05-24-book-style/results/smoke-stats.json
git commit -m "Record train.py smoke-run stats (1 epoch, batch 2)

Validates the continued-pretraining pipeline runs end-to-end on
Qwen3-4B + GB10 + Unsloth hotfixes. Captures peak memory, train
seconds, final loss. Full 5-epoch run lands in Task 9."
```

---

## Task 7: `evaluate.py` — PPL computation (TDD)

**Files:**
- Create: `tests/test_evaluate.py`
- Create: `evaluate.py` (PPL portion only; sample-gen added in Task 8)

- [ ] **Step 7.1: Write the failing test**

Path: `tests/test_evaluate.py`

```python
"""Plain-assert tests for evaluate.py PPL math + prompt builder."""
from __future__ import annotations

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evaluate import _ppl_from_logits_and_labels


def test_ppl_zero_loss_is_one():
    vocab = 5
    seq_len = 4
    logits = torch.full((1, seq_len, vocab), -100.0)
    labels = torch.tensor([[1, 2, 3, 4]])
    for t in range(seq_len):
        logits[0, t, labels[0, t]] = 100.0
    ppl, ce = _ppl_from_logits_and_labels(logits, labels)
    assert ce < 1e-3, f"expected near-zero ce, got {ce}"
    assert abs(ppl - 1.0) < 1e-2, f"expected ppl ~1.0, got {ppl}"


def test_ppl_uniform_logits_matches_vocab_size():
    vocab = 7
    seq_len = 8
    logits = torch.zeros((1, seq_len, vocab))
    labels = torch.randint(0, vocab, (1, seq_len))
    ppl, ce = _ppl_from_logits_and_labels(logits, labels)
    assert abs(ce - math.log(vocab)) < 1e-4, f"ce {ce} != log({vocab})={math.log(vocab)}"
    assert abs(ppl - vocab) < 1e-3, f"ppl {ppl} != {vocab}"


def test_ppl_ignores_minus_100_labels():
    vocab = 5
    seq_len = 4
    logits = torch.zeros((1, seq_len, vocab))
    labels = torch.tensor([[-100, -100, 0, 1]])
    ppl, ce = _ppl_from_logits_and_labels(logits, labels)
    assert abs(ce - math.log(vocab)) < 1e-4


if __name__ == "__main__":
    test_ppl_zero_loss_is_one()
    test_ppl_uniform_logits_matches_vocab_size()
    test_ppl_ignores_minus_100_labels()
    print("OK: all evaluate.py PPL tests passed")
```

- [ ] **Step 7.2: Run the test, confirm it fails**

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style
uv run --project ../.. python tests/test_evaluate.py
```

Expected: `ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 7.3: Implement `evaluate.py` (PPL portion)**

Path: `evaluate.py`

```python
"""Run PPL across base + each LoRA checkpoint, then 3 generation samples.

Outputs:
  results/ppl_table.json     - {"base": {ce, ppl}, "epoch_1": {...}, ...}
  results/samples/...        - 3 generation pairs (base vs best adapter)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Optional

# Unsloth before transformers
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from peft import PeftModel


def _ppl_from_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Cross-entropy + perplexity over non-ignored tokens (HF -100 convention).

    Callers do the next-token shift themselves before calling.
    """
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError(f"shape mismatch: logits {logits.shape}, labels {labels.shape}")
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_labels = labels.reshape(-1)
    loss = torch.nn.functional.cross_entropy(
        flat_logits, flat_labels, ignore_index=-100, reduction="mean",
    )
    ce = float(loss.item())
    ppl = math.exp(ce)
    return ppl, ce


def _pack_text_to_chunks(tokenizer, jsonl_path: str, max_seq_len: int) -> list[list[int]]:
    """Same packing as train.py; returns list[list[int]] of input_ids chunks."""
    raw = load_dataset("json", data_files=jsonl_path, split="train")
    eos = tokenizer.eos_token_id

    def tok(batch):
        return tokenizer(batch["text"], add_special_tokens=False, truncation=False)

    tokenized = raw.map(tok, batched=True, remove_columns=raw.column_names)
    all_ids: list[int] = []
    for ids in tokenized["input_ids"]:
        all_ids.extend(ids)
        all_ids.append(eos)
    n_full = (len(all_ids) // max_seq_len) * max_seq_len
    return [all_ids[i : i + max_seq_len] for i in range(0, n_full, max_seq_len)]


@torch.no_grad()
def _model_ppl_on_chunks(model, tokenizer, chunks: list[list[int]]) -> tuple[float, float]:
    device = next(model.parameters()).device
    total_ce_sum = 0.0
    total_tokens = 0
    for ids in chunks:
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        _, ce = _ppl_from_logits_and_labels(logits, labels)
        n_tok = labels.numel()
        total_ce_sum += ce * n_tok
        total_tokens += n_tok
    mean_ce = total_ce_sum / max(total_tokens, 1)
    return math.exp(mean_ce), mean_ce


def _load_base_or_adapter(model_name: str, max_seq_len: int, adapter_dir: Optional[pathlib.Path]):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer


def _enumerate_adapters(adapters_root: pathlib.Path) -> list[tuple[str, Optional[pathlib.Path]]]:
    out: list[tuple[str, Optional[pathlib.Path]]] = [("base", None)]
    ckpt_root = adapters_root / "checkpoints"
    if ckpt_root.exists():
        ckpts = sorted(p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-"))
        for i, p in enumerate(ckpts, start=1):
            out.append((f"epoch_{i}", p))
    best = adapters_root / "best"
    if best.exists():
        out.append(("best", best))
    return out


def main(args: argparse.Namespace) -> int:
    data_dir = pathlib.Path(args.data).expanduser()
    adapters_root = pathlib.Path(args.adapters).expanduser()
    out_dir = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    ppl_table: dict[str, dict[str, float]] = {}
    for label, adapter_path in _enumerate_adapters(adapters_root):
        print(f"[eval] PPL for {label} ({adapter_path or 'base'})")
        model, tokenizer = _load_base_or_adapter(args.model, args.max_seq_len, adapter_path)
        chunks = _pack_text_to_chunks(tokenizer, str(data_dir / "val.jsonl"), args.max_seq_len)
        ppl, ce = _model_ppl_on_chunks(model, tokenizer, chunks)
        ppl_table[label] = {"ce": round(ce, 4), "ppl": round(ppl, 4)}
        del model
        torch.cuda.empty_cache()

    (out_dir / "ppl_table.json").write_text(json.dumps(ppl_table, indent=2))
    print(json.dumps(ppl_table, indent=2))

    # Sample generation is added in Task 8.
    return 0
```

- [ ] **Step 7.4: Run the test, confirm it passes**

```bash
uv run --project ../.. python tests/test_evaluate.py
```

Expected: `OK: all evaluate.py PPL tests passed`

- [ ] **Step 7.5: Commit**

```bash
cd ~/projects/finetune
git add unsloth/experiments/2026-05-24-book-style/evaluate.py \
        unsloth/experiments/2026-05-24-book-style/tests/test_evaluate.py
git commit -m "Add evaluate.py PPL portion + tests

_ppl_from_logits_and_labels does the shifted next-token CE (ignoring
-100 labels). _model_ppl_on_chunks runs the model on each 1024-token
window and returns token-weighted mean PPL. _enumerate_adapters walks
lora-output/ to score base + each epoch checkpoint + best.

Tests cover zero-loss, uniform-logits-matches-vocab-size, and -100-label
ignored corner cases."
```

---

## Task 8: `evaluate.py` — 3-sample greedy generation

**Files:**
- Modify: `evaluate.py` (add sample-gen functions, extend `main`)
- Modify: `tests/test_evaluate.py` (add prompt-builder test)

- [ ] **Step 8.1: Append sample-generation code to `evaluate.py`**

Add these functions **immediately above** the existing `main()` in `evaluate.py`:

```python
# --- sample generation ------------------------------------------------

def _read_val_scenes(jsonl_path: str) -> list[str]:
    rows: list[str] = []
    for line in pathlib.Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line)["text"])
    return rows


def _build_sample_prompts(val_scenes: list[str]) -> list[tuple[str, str]]:
    """Return three (label, prompt) pairs from the val scenes.

    Sample 1: header + first paragraph of scene 1 (text up to but not
              including the first blank line after the header).
    Samples 2 and 3: header alone for scenes 2 and 3.
    """
    if len(val_scenes) < 3:
        raise ValueError(f"need 3 val scenes for sampling, got {len(val_scenes)}")
    s1 = val_scenes[0]
    head_end = s1.index("\n\n")
    body = s1[head_end + 2 :]
    first_para_end = body.find("\n\n")
    first_para = body if first_para_end == -1 else body[:first_para_end]
    p1 = s1[: head_end + 2] + first_para
    s2 = val_scenes[1]
    p2 = s2[: s2.index("\n\n") + 2]
    s3 = val_scenes[2]
    p3 = s3[: s3.index("\n\n") + 2]
    return [("sample_1", p1), ("sample_2", p2), ("sample_3", p3)]


@torch.no_grad()
def _generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def _generate_samples_and_write(
    model_name: str, adapters_root: pathlib.Path, val_jsonl: pathlib.Path,
    out_dir: pathlib.Path, max_seq_len: int, max_new_tokens: int,
) -> None:
    val_scenes = _read_val_scenes(str(val_jsonl))
    prompts = _build_sample_prompts(val_scenes)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    base_model, base_tok = _load_base_or_adapter(model_name, max_seq_len, None)
    for label, prompt in prompts:
        text = _generate_greedy(base_model, base_tok, prompt, max_new_tokens)
        (samples_dir / f"{label}_base.txt").write_text(text, encoding="utf-8")
    del base_model
    torch.cuda.empty_cache()

    best = adapters_root / "best"
    if not best.exists():
        print(f"[eval] WARN: {best} missing; skipping LoRA samples")
        return
    lora_model, lora_tok = _load_base_or_adapter(model_name, max_seq_len, best)
    for label, prompt in prompts:
        text = _generate_greedy(lora_model, lora_tok, prompt, max_new_tokens)
        (samples_dir / f"{label}_lora.txt").write_text(text, encoding="utf-8")
    del lora_model
    torch.cuda.empty_cache()
```

- [ ] **Step 8.2: Replace the tail of `main()` to call sample generation**

In `evaluate.py`, locate the last block of `main()`:

```python
    # Sample generation is added in Task 8.
    return 0
```

Replace it with:

```python
    _generate_samples_and_write(
        model_name=args.model,
        adapters_root=adapters_root,
        val_jsonl=data_dir / "val.jsonl",
        out_dir=out_dir,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.gen_tokens,
    )
    print(f"[eval] wrote samples to {out_dir/'samples'}/")
    return 0
```

- [ ] **Step 8.3: Add the prompt-builder test**

Add this test to `tests/test_evaluate.py` (before the `if __name__ == "__main__":` block):

```python
from evaluate import _build_sample_prompts


def test_build_sample_prompts_grounded_and_header_only():
    scenes = [
        "[Book 1, Chapter 10, Scene 1 — POV: Mira, Location: Bridge]\n\nFirst para.\n\nSecond para.",
        "[Book 1, Chapter 10, Scene 2 — POV: Voss, Location: CIC]\n\nScene 2 body.",
        "[Book 1, Chapter 10, Scene 3 — POV: Mira, Location: Hangar]\n\nScene 3 body.",
    ]
    prompts = _build_sample_prompts(scenes)
    assert [p[0] for p in prompts] == ["sample_1", "sample_2", "sample_3"]
    assert prompts[0][1].endswith("First para.")
    assert "Second para." not in prompts[0][1]
    assert prompts[1][1].endswith("Location: CIC]\n\n")
    assert "Scene 2 body" not in prompts[1][1]
    assert prompts[2][1].endswith("Location: Hangar]\n\n")
```

Then update the runner block at the bottom of the file:

```python
if __name__ == "__main__":
    test_ppl_zero_loss_is_one()
    test_ppl_uniform_logits_matches_vocab_size()
    test_ppl_ignores_minus_100_labels()
    test_build_sample_prompts_grounded_and_header_only()
    print("OK: all evaluate.py tests passed")
```

- [ ] **Step 8.4: Run the tests, confirm all pass**

```bash
uv run --project ../.. python tests/test_evaluate.py
```

Expected: `OK: all evaluate.py tests passed`

- [ ] **Step 8.5: Commit**

```bash
cd ~/projects/finetune
git add unsloth/experiments/2026-05-24-book-style/evaluate.py \
        unsloth/experiments/2026-05-24-book-style/tests/test_evaluate.py
git commit -m "evaluate.py: add 3-sample greedy generation (base vs best LoRA)

_build_sample_prompts produces: sample 1 = header + first paragraph
(text up to first blank line after the header), samples 2 and 3 =
header-only prompts from val chapter scenes 2 and 3. Greedy decode
(do_sample=False, num_beams=1) for deterministic repro. Saves six
text files to results/samples/.

Added prompt-builder test confirming header-only samples don't leak
scene body, grounded sample includes first paragraph only."
```

---

## Task 9: Full end-to-end run + verify success criteria

**Files:**
- Create: `tests/verify_ppl.py` (standalone PPL-pass-check)
- Modify: `README.md` (results table)

- [ ] **Step 9.1: Add the `verify_ppl.py` standalone checker**

Path: `tests/verify_ppl.py`

```python
"""Read results/ppl_table.json and print PASS/FAIL vs the success bar.

Success bar (from design.md): best PPL across non-base entries must be
<= 0.70 * base PPL.
"""
from __future__ import annotations

import json
import pathlib
import sys


def main() -> int:
    table_path = pathlib.Path(__file__).resolve().parent.parent / "results" / "ppl_table.json"
    if not table_path.exists():
        print(f"missing {table_path}; run ./run.sh eval first", file=sys.stderr)
        return 2
    table = json.loads(table_path.read_text())
    if "base" not in table:
        print("ppl_table.json has no 'base' entry", file=sys.stderr)
        return 2
    base_ppl = table["base"]["ppl"]
    others = {k: v for k, v in table.items() if k != "base"}
    if not others:
        print("no non-base PPL entries to compare", file=sys.stderr)
        return 2
    best_label = min(others, key=lambda k: others[k]["ce"])
    best_ppl = others[best_label]["ppl"]
    ratio = best_ppl / base_ppl
    print(f"base ppl       = {base_ppl:.4f}")
    print(f"best ({best_label}) = {best_ppl:.4f}")
    print(f"ratio          = {ratio:.4f}  (success bar: <= 0.70)")
    if ratio <= 0.70:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 9.2: Run the full pipeline on node-2**

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style

# Clean smoke-run leftovers so we start from a known state
rm -rf lora-output/best lora-output/checkpoints lora-output/stats.json \
       results/ppl_table.json results/samples/*.txt 2>/dev/null

./run.sh prepare
./run.sh train
./run.sh eval
```

Expected wall-clock total: ~25 min on first run (includes 4B download), ~15 min on re-runs (model cached).

- [ ] **Step 9.3: Verify the quantitative success criterion**

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style
uv run --project ../.. python tests/verify_ppl.py
```

Expected: prints `PASS` and exits 0.

If FAIL: do **not** record this as success. Capture `results/ppl_table.json` plus `lora-output/stats.json`, append a "Diagnostic: failure path" note to `design.md`'s "Diagnostic outcomes" section, and re-evaluate hyperparameters before re-running.

- [ ] **Step 9.4: Eyeball qualitative samples**

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style/results/samples
for n in 1 2 3; do
  echo "=== sample $n BASE ===" ; cat sample_${n}_base.txt
  echo
  echo "=== sample $n LORA ===" ; cat sample_${n}_lora.txt
  echo
done | less
```

Check:
- Samples 2 and 3 (header-only): does the LoRA output sound like your source corpus's register and voice (genre-appropriate vocabulary, named POV character appearing)? Or does it produce header-noise / generic continuation?
- Sample 1 (grounded): does LoRA stay on tone better than base?

Record one observation per sample in a new `results/notes.md`.

- [ ] **Step 9.5: Commit results**

```bash
cd ~/projects/finetune
# Override .gitignore for the specific result files we want under version control
git add -f unsloth/experiments/2026-05-24-book-style/results/ppl_table.json \
           unsloth/experiments/2026-05-24-book-style/results/samples/*.txt \
           unsloth/experiments/2026-05-24-book-style/results/notes.md \
           unsloth/experiments/2026-05-24-book-style/tests/verify_ppl.py
# Move run stats out of lora-output (gitignored) into results
mv unsloth/experiments/2026-05-24-book-style/lora-output/stats.json \
   unsloth/experiments/2026-05-24-book-style/results/run-stats.json
git add unsloth/experiments/2026-05-24-book-style/results/run-stats.json
git commit -m "Record full 5-epoch run results

ppl_table.json: base vs each epoch checkpoint + best
samples/sample_{1..3}_{base,lora}.txt: side-by-side greedy generations
notes.md: qualitative observations on each sample pair
run-stats.json: timing/memory for the real run
verify_ppl.py: standalone pass/fail checker against the design success bar"
```

- [ ] **Step 9.6: Update README with actual numbers**

Append a "Results (2026-05-24)" section to `README.md`:

```markdown
## Results (2026-05-24)

| Metric | Value |
|--------|------:|
| Base PPL on val | (from ppl_table.json) |
| Best LoRA PPL on val | (from ppl_table.json) |
| Ratio (best/base) | (computed) |
| Train seconds | (from run-stats.json) |
| Peak GPU memory (GiB) | (from run-stats.json) |

Pass/fail vs success criteria: see `design.md` § "Success criteria".
Qualitative notes: `results/notes.md`.
```

Fill in each `(from ...)` placeholder with the actual numbers from the run, then:

```bash
cd ~/projects/finetune
git add unsloth/experiments/2026-05-24-book-style/README.md
git commit -m "README: record real-run results table"
```

---

## Self-Review

**1. Spec coverage** (each section of `design.md` → task):

| Spec item | Task(s) |
|-----------|---------|
| Project layout | 1 (scaffolding, .gitignore, .gitkeep, run.sh, README) |
| `lora_test.py prepare` (header format, sqlite query, train/val split) | 2, 3, 4 |
| `lora_test.py train` (model load, LoRA, packed CLM, Trainer) | 5, 6 |
| `lora_test.py eval` (PPL table per epoch, 3 greedy samples) | 7, 8 |
| Run flow (`./run.sh prepare && train && eval`) | 4 (prepare), 6 (smoke train), 9 (full run) |
| Quantitative success criterion (PPL ≤ 0.70 × base) | 9.3 (via `verify_ppl.py`) |
| Qualitative success criterion (samples sound like the book) | 9.4 (eyeball + notes.md) |
| Risks: HF download speed | 6.1 (note) |
| Risks: Unsloth API drift after `uv sync` | 6.1 (note re: `apply-hotfixes.sh`) |
| Risks: header overfitting | 9.4 (eyeball check) |
| Note on `git init` for finetune/ | 1.6 |

**2. Placeholder scan:** no "TBD"/"TODO"/"to be implemented" anywhere. The `(from ppl_table.json)` markers in Step 9.6 are deliberate slots for run-time values, not unspecified design.

**3. Type/name consistency:** `_pack_text_to_chunks` appears in both `train.py` (Task 5) and `evaluate.py` (Task 7) with the same signature — intentional duplication (~10 lines) to keep `common.py` dependency-free. Acceptable for this experiment; revisit if a third caller appears.

**Test + evaluation suite coverage** (per the explicit user request):

- `tests/test_common.py` — 3 assertions on header/scene formatting (Task 2)
- `tests/test_prepare.py` — 3 assertions on sqlite query, split, JSONL writeback, with in-memory fixture (Task 3)
- `tests/test_evaluate.py` — 4 assertions on PPL math (3) and prompt construction (1), with synthetic tensors (Tasks 7, 8)
- `tests/verify_ppl.py` — standalone PASS/FAIL gate against the design's quantitative success bar (Task 9)
- `evaluate.py` — the evaluation suite itself: PPL table across base + each epoch checkpoint + best, plus 3 side-by-side greedy generations
- Smoke run (Task 6) and full run (Task 9) are the integration tests that exercise the whole pipeline against real data and real hardware
