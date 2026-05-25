# Book-Style LoRA Test — 2026-05-24

## Purpose

First end-to-end LoRA training experiment on the existing GB10 native Unsloth
install. Validates that the training stack works on real prose data, and that
we can measurably detect "training worked" through both quantitative
(perplexity drop on held-out chapters) and qualitative (side-by-side
generation samples) signals.

This is infrastructure validation and experimentation, not a production
fine-tune. There is no expectation that the resulting adapter is useful for
downstream work.

## Goals

- Confirm end-to-end LoRA training works on the existing `finetune/unsloth/`
  venv with its five GB10/sm_121 hotfixes.
- Establish a reusable PPL-based + sample-based verification pattern for
  future LoRA experiments.
- Produce baseline timing and peak-memory numbers for `Qwen3-4B` + LoRA on
  one GB10.
- Surface any Unsloth + aarch64 + sm_121 issues not caught by the
  Q&A-style smoke test in `../../train.py`.

## Non-goals

- Multi-node training. Defer until a validated multi-node Unsloth path
  exists; this run is single-node only.
- Production-quality book-style LoRA. The corpus (~170K tokens) and model
  (4B) are too small for that.
- Cross-framework comparison (TRL, axolotl, llama-factory, NeMo). Those
  warrant their own experiments under sibling subdirs of
  `~/projects/finetune/`.

## Inputs

### Source data

Fiction-corpus Book 1, sourced from
`~/projects/scriptorium/canon/fiction_corpus.sqlite`:

- 46 scenes across 10 chapters
- 138,820 words (~180K tokens at ~1.3 tok/word)
- 7 revisions max; we use the latest revision per (book, chapter, scene)

The `scenes` table also exposes `pov_character` and `location` per scene —
used in the metadata header below.

### Base model

`Qwen/Qwen3-4B` (base, not Instruct).

- Base is cleaner for continued-pretraining on raw prose; no chat template
  to fight.
- 4B fits comfortably in 121 GiB unified memory with room for LoRA,
  activations, and optimizer state.
- Realistic enough that PPL/sample signals plausibly translate to larger
  models for future runs.
- Not yet cached locally or on NAS → ~8 GB initial download from HF.

## Project layout

```
~/projects/finetune/unsloth/                            # existing, untouched
├── pyproject.toml, uv.lock, stubs/                     # shared venv + torchvision stub
├── train.py, train.sh, apply-hotfixes.sh               # existing smoke test
└── experiments/
    └── 2026-05-24-book-style/                          # this experiment
        ├── README.md
        ├── design.md                                   # this file
        ├── lora_test.py                                # subcommands: prepare, train, eval
        ├── run.sh                                      # uv-run wrapper, mirrors ../../train.sh
        ├── data/                                       # populated by `prepare`
        │   ├── train.jsonl
        │   └── val.jsonl
        ├── lora-output/                                # populated by `train`
        │   └── epoch_{1..5}/
        └── results/                                    # populated by `eval`
            ├── ppl_table.json
            └── samples/
                ├── sample_{1..3}_base.txt
                └── sample_{1..3}_lora.txt
```

The experiment reuses the parent venv by invoking
`uv run --project ../..` inside `run.sh`. Inherits all five hotfixes via
the patched site-packages. Adds zero new dependencies — `peft`, `trl`,
`transformers`, `datasets` are already present.

## Components

### `lora_test.py prepare`

CLI: `prepare --canon-db PATH --out data/`

- Default `--canon-db`:
  `~/projects/scriptorium/canon/fiction_corpus.sqlite`
- Query latest revision per (book, chapter, scene) where `book = 1`,
  ordered by chapter, scene.
- Per scene, format text as:

  ```
  [Book 1, Chapter {c}, Scene {s} — POV: {pov_character}, Location: {location}]

  {text}
  ```

- Train split: chapters 1-9 (43 scenes, ~131K words ≈ 170K tokens).
- Val split: chapter 10 (3 scenes, ~7.5K words ≈ 10K tokens).
- Output `data/train.jsonl` and `data/val.jsonl`, one JSON object per
  scene of shape `{"text": "<formatted scene>"}`.

### `lora_test.py train`

Cribs model loading, LoRA setup, and stats logging from `../../train.py`.
Differs in:

- Uses raw `transformers.Trainer`, not `SFTTrainer` — we want CLM on packed
  prose, not chat formatting.
- Dataset: tokenize each `data/*.jsonl` row, concatenate all token streams
  per split, chunk into fixed `max_seq_length` windows.
- Collator: `DataCollatorForLanguageModeling(mlm=False)`.

Hyperparameters:

| Field | Value |
|------|------|
| Model | `Qwen/Qwen3-4B` |
| dtype | `torch.bfloat16` |
| `max_seq_length` | 1024 |
| LoRA `r` / `alpha` | 16 / 32 |
| LoRA target_modules | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| LoRA dropout / bias | 0 / `none` |
| `use_gradient_checkpointing` | `"unsloth"` |
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 2 (effective batch 8) |
| `learning_rate` | 1e-4 |
| `lr_scheduler_type` | `cosine` |
| `warmup_ratio` | 0.03 |
| `num_train_epochs` | 5 |
| `optim` | `adamw_torch` |
| `weight_decay` | 0.01 |
| `eval_strategy` | `epoch` |
| `save_strategy` | `epoch` |
| `load_best_model_at_end` | `True` (on `eval_loss`) |
| `bf16` | `True`, `fp16` `False` |
| `report_to` | `"none"` |

Why `max_seq_length=1024`, not 4096: corpus is small. At 4K windows we'd
get ~40 sequences total — ~5 gradient steps per epoch, ~25 across the
whole run, not enough to see clean signal. At 1K we get ~170 sequences →
~21 steps/epoch → ~105 total updates, which is in the standard band for
small-corpus LoRA.

Save adapters to `lora-output/epoch_{N}/`. After training, also write a
`stats.json` next to them (same fields as the existing smoke test:
peak_gpu_memory_gib, train_seconds, final_train_loss, etc.).

### `lora_test.py eval`

For each of `[base, epoch_1, epoch_2, ..., epoch_5]`:

1. Load `Qwen/Qwen3-4B` via `FastLanguageModel.from_pretrained(... )`.
2. If not the base, wrap with `PeftModel.from_pretrained(model, adapter_path)`.
3. Tokenize `data/val.jsonl`, pack into 1024-token windows (same as
   training).
4. Compute mean next-token cross-entropy with `torch.no_grad()` in bf16.
5. `ppl = math.exp(ce)`.

Write `results/ppl_table.json`:

```json
{
  "base":    {"ce": 2.65, "ppl": 14.20},
  "epoch_1": {"ce": 2.31, "ppl": 10.10},
  ...,
  "epoch_5": {"ce": 1.93, "ppl":  6.90}
}
```

Then generate three samples using the **best** adapter (by `eval_loss`)
and the base, side by side, with greedy decoding (deterministic for
repro), `max_new_tokens=500`:

- **Sample 1 — grounded continuation.** Scene 1 of chapter 10:
  header + first paragraph (defined as the scene text up to but not
  including the first blank line after the header) → continue. Tests
  whether the model can extend in-style from a prose anchor.
- **Sample 2 — header-only generation.** Scene 2 of chapter 10:
  metadata header alone → produce 500 tokens. Tests whether the model
  learned to condition on the header.
- **Sample 3 — header-only generation.** Scene 3 of chapter 10:
  metadata header alone → produce 500 tokens. Second header-conditioned
  data point.

Save each prompt + continuation pair to `results/samples/sample_{N}_{base,lora}.txt`.
The eyeball comparison is the qualitative half of "did training work"; no
LLM-judge is used in this experiment.

## Run flow

```bash
cd ~/projects/finetune/unsloth/experiments/2026-05-24-book-style

./run.sh prepare
./run.sh train
./run.sh eval
```

Where `run.sh` is roughly:

```bash
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
export PYTHONPATH="$ROOT/stubs:${PYTHONPATH:-}"
exec uv run --project "$ROOT" python "$DIR/lora_test.py" "$@"
```

Estimated wall-clock on one GB10 (node-2):

| Step    | Time          |
|---------|---------------|
| prepare | < 1 min       |
| train   | ~15-20 min (incl. ~5 min first-time Qwen3-4B download) |
| eval    | ~5 min        |
| **total** | **~25 min first run, ~15 min on re-runs (model cached)** |

Target node: **node-2**. node-1 is running the scriptorium books daemon
plus `claude-tmux-books`; isolating training to node-2 avoids contention.

## Success criteria

**Quantitative:**

- Final val PPL ≤ 0.70 × base PPL (i.e. ≥ 30% relative drop).
- No CUDA OOM, no Unsloth/transformers exception, no hang.
- `peak_gpu_memory_gib` recorded in stats.json (used for future capacity
  planning).

**Qualitative:**

- LoRA samples — especially the two header-only prompts (sample 2 and 3) —
  read recognizably more like Hard MilSF in the book's voice than the base
  outputs: technical register, characteristic vocabulary, named POV
  character or location appearing somewhere in the generation.

**Diagnostic outcomes:**

- PPL pass + samples pass → training stack and verification pattern both
  validated. Move on to the next experiment.
- PPL pass + samples fail → training memorized the corpus distribution
  but didn't generalize to header-conditioning. Still useful infra signal;
  consider whether to retry with raw concat (no header) or different
  header format.
- PPL fail → real pipeline issue (data prep, optimizer config, hotfix
  drift). Debug before drawing any conclusions.

## Risks and open questions

- **HF download speed.** Qwen3-4B is ~8 GB. NAS cache check first
  (`/mnt/nas/hf-cache/models--Qwen--Qwen3-4B*`); if absent there too, just
  pull from HF and accept ~5 min on first run.
- **Tiny corpus.** 170K train tokens is small for a 4B LoRA at r=16. May
  overfit fast; mitigated by `load_best_model_at_end` on val loss. If val
  loss bottoms out at epoch 1-2, future runs should drop to 2-3 epochs.
- **Header overfitting.** The model might learn to emit garbled
  `[Book 1, Chapter ...]` headers in continuations. If samples 2 and 3
  produce header-noise rather than prose, drop the metadata header for
  the next run and use raw scene-text concat.
- **xformers SDPA fallback.** Hotfix #5 disables xformers on sm_121 →
  PyTorch SDPA fallback. Performance hit not yet measured. Captured in
  `peak_gpu_memory_gib` and `train_seconds` for this run.
- **Unsloth API drift.** Hotfixes target a specific Unsloth version. If
  `uv sync` updated Unsloth recently, re-run `apply-hotfixes.sh` first
  before training.
- **Git.** `~/projects/finetune/` is not currently a git repo. We are
  *not* initializing one as part of this experiment — defer that decision
  to the user. The design doc and code can be committed if/when the
  repo is initialized.

## Future work (out of scope here)

- Repeat with `Qwen3.6-27B-bf16` once the 4B path is validated, to check
  whether the PPL/sample signal translates to a production-size base.
- Multi-node DDP smoke test with TRL+PEFT+Accelerate as the contribution
  opportunity discussed (issue: Unsloth multi-node isn't first-class yet).
- Beat-to-scene instruction-tuning variant — directly useful for
  scriptorium Phase 3 (plotter sketches → scene drafts).
- Comparison run with vanilla TRL+PEFT (no Unsloth) on the same corpus,
  same model, same hyperparams, to measure Unsloth's per-GPU speedup on
  GB10 specifically.
