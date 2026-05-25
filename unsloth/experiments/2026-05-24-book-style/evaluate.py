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
    if not chunks:
        raise ValueError("chunks is empty — check val.jsonl path and packing")
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
    mean_ce = total_ce_sum / total_tokens
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
        ckpts = sorted(
            (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        for i, p in enumerate(ckpts, start=1):
            out.append((f"epoch_{i}", p))
    best = adapters_root / "best"
    if best.exists():
        out.append(("best", best))
    return out


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
