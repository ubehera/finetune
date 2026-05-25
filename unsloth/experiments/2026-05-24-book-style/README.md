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

## Results (2026-05-24)

| Metric | Value |
|--------|------:|
| Base PPL on val | 30.3406 |
| Best LoRA PPL on val | 14.2714 |
| Ratio (best/base) | 0.4704 |
| Train seconds | 486.16 |
| Peak GPU memory (GiB) | 13.15 |

Pass/fail vs success criteria: see `design.md` § "Success criteria".
Qualitative notes: `results/notes.md`.
