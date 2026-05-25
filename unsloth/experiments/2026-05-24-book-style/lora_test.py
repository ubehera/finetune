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
