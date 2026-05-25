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
