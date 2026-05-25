# LoRA Fine-Tuning Test Results — 2026-05-24

A portable summary of what this experiment did and what it found. Written for
readers without prior machine-learning background.

---

## What This Test Was

We took an off-the-shelf AI text generator and spent eight minutes teaching
it to write in the specific style of one unpublished novel. Two independent
checks — a numerical measurement and a side-by-side sample comparison —
both confirmed the training had a real, measurable effect. The whole thing
ran on a single desktop workstation.

This was a deliberately small, controlled test: an infrastructure check,
not an attempt to produce a polished writing tool.

---

## A Few Terms, Defined Once

Skim or skip if these are familiar.

- **Model.** A large mathematical function that turns a string of input text
  into a probability distribution over possible next words. The model used
  here is `Qwen3-4B`, a freely available text generator from Alibaba. The
  "4B" means 4 **billion parameters** — that is, the model has 4 billion
  internal numerical settings, learned previously from a large slice of the
  open internet. By 2026 standards, 4B is mid-small: capable but light.
- **GPU memory.** The dedicated working memory the model occupies while it
  runs. Models that don't fit can't run. The workstation here has 121 GB of
  it; the 4B model used 13 GB.
- **Training corpus.** The collection of text we want the model to learn
  from. In this experiment: one unpublished novel, around 139,000 words.
- **Fine-tuning.** Adjusting a pre-trained model so it does better on a
  specific kind of text, without retraining from scratch.
- **LoRA (Low-Rank Adaptation).** A specific, cheap form of fine-tuning.
  Think of the original model as a vast machine with millions of dials
  already set during years of training. LoRA clips a thin overlay of new,
  smaller dials on top, adjusts only those, and never touches the
  underlying machine. About 0.7% of the model's parameters get added and
  trained; the original 99.3% are preserved exactly. The benefit: training
  is fast, cheap, reversible, and easy to swap out.

---

## What "Perplexity" Means

Perplexity is the headline measurement here, so it's worth a paragraph.

When an AI text model reads a passage, at each word it assigns probabilities
to all the possible next words. Perplexity is a single number summarizing
how concentrated those probabilities are on average. A perplexity of 1 would
mean the model perfectly predicts each next word; higher numbers mean more
spread-out uncertainty.

Two useful rules of thumb, both rough:

- Perplexity scales like a "branching factor": a score of 30 corresponds to
  the average uncertainty a model would feel if it were choosing the next
  word from about 30 equally-likely options at each step.
- What matters in practice is not the absolute number but the **ratio**
  between two models on the same text. If model B's perplexity is half of
  model A's on the same passage, model B is roughly half as uncertain.

Lower is better. Ratios are what we compare.

---

## The Setup

- **Training corpus:** One unpublished novel, Book 1 (~139,000 words across
  10 chapters). Chapters 1–9 were used as the study material; chapter 10
  was held back as an exam the model never saw during training.
- **Starting model:** `Qwen3-4B`. No knowledge of this specific novel.
- **Method:** LoRA fine-tuning. Five passes through the training material
  (one pass = the model reads the whole training set once and slightly
  adjusts its added dials). After each pass, a snapshot of the model's
  adjusted state is saved. At the end, the snapshot that performed best
  on the exam chapter is kept.
- **Hardware:** One NVIDIA DGX Spark workstation ("GB10"), 121 GB of unified
  memory.

---

## The Headline Numbers

| What we measured | Value | What it means |
|---|---|---|
| Training time | **8 minutes** | The learning loop itself, after a one-time model download. |
| GPU memory used | **13 GB out of 121 GB** (~11%) | The 4B model fits comfortably; the box is well under-utilized. |
| Perplexity on the exam chapter, original model | **30.3** | The base model's surprise score reading the held-out chapter cold. |
| Perplexity on the exam chapter, trained model | **14.3** | The trained model's surprise score on the same chapter. |
| Ratio (trained ÷ original) | **0.47** | The trained model is **53% less perplexed** by the held-out prose. |

The test was set to pass at a ratio of 0.70 or lower (a 30% relative drop in
perplexity). The actual result of 0.47 cleared that bar by a wide margin.

A note: the 30% pass bar was chosen by the experiment's author in advance,
not externally validated. It was set high enough that a no-op or broken
training run would not pass.

---

## What the Generated Samples Looked Like

The numerical perplexity test is paired with a qualitative check. Both
models (original and trained) were given the same three writing prompts
derived from the held-out chapter:

- **Prompt 1:** A scene header plus the opening paragraph of an actual
  chapter, asking the model to continue. *Outcome:* Both models started on
  tone, then degraded into the same pattern of repetitive phrases. This
  failure is a known consequence of one specific generation setting
  (the model was instructed to pick the single most likely next word at
  every step, with no penalty for repeating itself; this setting was held
  fixed across both models so the comparison would be apples-to-apples).
  It is independent of whether the training worked; both base and trained
  models exhibit it, and it is fixed by changing the generation setting,
  not by more training.
- **Prompt 2 and Prompt 3:** Just a scene header — character name and
  location — asking the model to invent the scene from scratch. *Outcome:*
  The original model invented a generic fantasy or sci-fi setting,
  unrelated to the source novel; in one case it wrote stage directions for
  a screenplay rather than novel prose. The trained model wrote novel-style
  prose, used the correct point-of-view character, named the right
  locations, and referenced procedural terms specific to the source
  novel's world. None of that vocabulary appeared in the input prompt —
  the model learned it from the training corpus.

Sample text is not included in this public repository. The qualitative
comparison above was verified locally by the experiment's author; a reader
of this document is taking it on trust. The quantitative perplexity result
is independently reproducible from the code with any comparable corpus.

---

## What This Test Does Not Show

- **Whether the trained model still works well on text outside the novel.**
  This was not measured. LoRA fine-tuning is generally believed to leave
  the original model's behavior on out-of-domain text mostly intact, but
  that was not directly tested here.
- **Whether five passes was the right number.** Three or four passes might
  have produced similar results in less time; more than five might have
  caused the model to memorize rather than generalize. The perplexity drop
  flattened sharply after the third pass, hinting at diminishing returns,
  but this wasn't explored further.
- **Whether a different base model, different size, or different LoRA
  settings would have been better.** Only one configuration was tested.
- **Anything about writing quality beyond surface style.** Perplexity and
  vocabulary uptake are measurable; "good prose" is not. The trained model
  imitates the source's voice; whether that voice is *good* (compelling,
  coherent, dramatic) is a different question this test did not address.

---

## What This Test Does Show

- The training pipeline (data preparation, training, evaluation) runs
  cleanly end-to-end on this specific hardware-and-software combination,
  which had previously been an open question. Several known compatibility
  problems between the GPU's chip architecture, the operating system, and
  the AI libraries had to be worked around to get the pipeline running;
  the successful run is evidence those workarounds are correct.
- The objective measurement (perplexity ratio) and the qualitative
  inspection (sample comparison) detect the effect of training clearly,
  in the same direction, on a corpus this small (around 170,000 tokens).
  That gives a reusable template for future fine-tuning experiments to
  measure success against.
- A 4B model uses roughly 11% of the workstation's memory while training,
  meaning larger models (e.g. 27B parameters, about seven times larger)
  should also fit. This is suggestive rather than proven — the 35B case
  in particular would be tighter — but it is encouraging.

---

## How the Run Was Structured

The whole pipeline is three commands:

```
./run.sh prepare    # extract the training corpus, split into training/exam portions
./run.sh train      # five passes over the training material, save the best snapshot
./run.sh eval       # measure perplexity on the exam chapter, generate sample text
```

End-to-end wall-clock the first time is around 25 minutes (most of that is
a one-time download of the base model). Subsequent runs are under 15
minutes total.

---

## Reproducibility

Anyone with:

1. An NVIDIA workstation with at least 16 GB of GPU memory.
2. The codebase in this repository.
3. Their own text corpus, plus a willingness to adapt the data-loading
   step. The pipeline currently expects scenes stored as rows in a
   small SQLite database with specific metadata columns; the data loader
   (`prepare.py`) would need a small modification for any other format.

…can run a comparable experiment. The exact numbers reported here will
differ depending on the corpus used. The pass/fail criterion (ratio ≤ 0.70)
is reusable as-is.

---

## For Researchers Digging Deeper

These files in the repository contain the raw artifacts behind the numbers
above:

- `design.md` — the experiment's technical specification (written before
  the run).
- `plan.md` — the step-by-step build plan executed to produce the code.
- `results/ppl_table.json` — perplexity for every snapshot (one per pass),
  the raw data behind the headline result.
- `results/run-stats.json` — timing, memory, loss, and hardware details
  for the full run.
- `results/smoke-stats.json` — equivalents for a faster 1-pass sanity-check
  run that preceded the full experiment.
- `tests/verify_ppl.py` — the script that produces the PASS/FAIL verdict
  from `ppl_table.json` against the 0.70 threshold.

---

## Bottom Line

The training stack works. A small, well-defined experiment with a clear
pass/fail criterion produced a clear pass on this hardware. Future
fine-tuning experiments on this workstation can now proceed with a
validated baseline and a verification pattern to compare against.
