# Book-Style LoRA Test (2026-05-24)

Continued-pretraining LoRA fine-tune of Qwen3-4B on a `fiction_corpus` Book 1,
to validate the GB10 native Unsloth install end-to-end. The corpus is a private
WIP novel — committed sample outputs and qualitative notes are excluded from
this public release; only the aggregate metrics in `results/` are public.

**See `design.md` for the full spec and `plan.md` for the build sequence.**

## Quick run

    ./run.sh prepare
    ./run.sh train
    ./run.sh eval

Inspect `results/ppl_table.json` for the PPL trajectory. Sample text outputs
are gitignored — they're regenerated locally by `./run.sh eval` from your own
corpus.

## Tests

    uv run --project ../.. python tests/test_common.py
    uv run --project ../.. python tests/test_prepare.py
    uv run --project ../.. python tests/test_evaluate.py

## Requires

- Parent `../../` venv populated by `uv sync` and patched by `../../apply-hotfixes.sh`.
- Read access to your corpus sqlite (default path: `~/projects/scriptorium/canon/fiction_corpus.sqlite`; override with `--canon-db`).
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

Qualitative sample inspection was performed locally but the side-by-side
generations contain WIP novel text and have been excluded from this repo. The
quantitative pass (ratio 0.4704 ≤ 0.70) is reproducible by anyone with the
codebase and a similar corpus.
