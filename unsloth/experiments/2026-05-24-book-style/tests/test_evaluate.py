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
