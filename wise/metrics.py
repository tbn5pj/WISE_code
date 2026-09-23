"""Canonical open-answer scoring and paired bootstrap."""

from __future__ import annotations

import collections
import re
import string

import numpy as np


def normalize_answer(text: str) -> str:
    text = text.casefold()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1(prediction: str, gold: str) -> float:
    predicted, reference = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = collections.Counter(predicted) & collections.Counter(reference)
    overlap = sum(common.values())
    if not predicted or not reference:
        return float(predicted == reference)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def twowiki_normalize_answer(text: str) -> str:
    """Official 2Wiki evaluator normalization (lower, not casefold)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def twowiki_answer_f1(prediction: str, gold: str) -> float:
    """Official 2Wiki yes/no/noanswer guard and token-overlap rule."""
    p, g = twowiki_normalize_answer(prediction), twowiki_normalize_answer(gold)
    specials = {"yes", "no", "noanswer"}
    if p != g and (p in specials or g in specials):
        return 0.0
    pt, gt = p.split(), g.split()
    overlap = sum((collections.Counter(pt) & collections.Counter(gt)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pt), overlap / len(gt)
    return 2 * precision * recall / (precision + recall)


def benchmark_scores(benchmark: str, prediction: str, gold: str) -> tuple[float, float]:
    if benchmark in {"2WikiMultihopQA", "2WikiMultiHopQA"}:
        return twowiki_answer_f1(prediction, gold), float(
            twowiki_normalize_answer(prediction) == twowiki_normalize_answer(gold))
    if benchmark == "HotpotQA":
        return answer_f1(prediction, gold), answer_em(prediction, gold)
    raise ValueError(f"unsupported behavioral benchmark: {benchmark}")


def paired_bootstrap(differences, seed: int, draws: int = 10_000) -> tuple[float, float, float]:
    """Mean difference and 2.5/97.5% quantiles of paired resampled means."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("expected nonempty paired differences")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)
