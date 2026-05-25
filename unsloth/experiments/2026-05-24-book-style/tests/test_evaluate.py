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


import pathlib as _pl
import tempfile as _tf

from evaluate import _enumerate_adapters


def test_enumerate_adapters_sorts_by_integer_step_not_lex():
    """Five 5-epoch run at ~21 steps/epoch produces checkpoint-105 > checkpoint-21
    by integer, but checkpoint-105 < checkpoint-21 lexicographically. The function
    must return ascending integer-step order.
    """
    with _tf.TemporaryDirectory() as tmp:
        root = _pl.Path(tmp) / "lora-output"
        ckpts = root / "checkpoints"
        ckpts.mkdir(parents=True)
        # Intentionally create them out of "lex" order to ensure sort logic runs
        for step in (105, 21, 42, 84, 63):
            (ckpts / f"checkpoint-{step}").mkdir()
        (root / "best").mkdir()

        result = _enumerate_adapters(root)
        # Expected: base, then 5 epoch entries by step, then best
        labels = [label for label, _ in result]
        assert labels == ["base", "epoch_1", "epoch_2", "epoch_3", "epoch_4", "epoch_5", "best"], labels
        # The first (epoch_1) checkpoint must be step 21 (not step 105, the lex-sort winner)
        epoch_1_path = result[1][1]
        assert epoch_1_path.name == "checkpoint-21", f"epoch_1 got {epoch_1_path.name}; sort is still lex"
        # Last epoch must be step 105
        epoch_5_path = result[5][1]
        assert epoch_5_path.name == "checkpoint-105", f"epoch_5 got {epoch_5_path.name}"


def test_enumerate_adapters_handles_no_checkpoints_no_best():
    """Empty lora-output -> only ('base', None)."""
    with _tf.TemporaryDirectory() as tmp:
        root = _pl.Path(tmp) / "lora-output"
        root.mkdir()
        result = _enumerate_adapters(root)
        assert result == [("base", None)]


if __name__ == "__main__":
    test_ppl_zero_loss_is_one()
    test_ppl_uniform_logits_matches_vocab_size()
    test_ppl_ignores_minus_100_labels()
    test_enumerate_adapters_sorts_by_integer_step_not_lex()
    test_enumerate_adapters_handles_no_checkpoints_no_best()
    print("OK: all evaluate.py PPL tests passed")
