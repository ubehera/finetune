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
