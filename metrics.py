import re

from typing import Iterable
from dataclasses import dataclass, field
from rouge_score.rouge_scorer import RougeScorer
from logger_config import setup_logger
from utils import preprocess

logger = setup_logger(__name__)


@dataclass(frozen=True)
class RougeMetrics:
    """
    Data for each metric type.
    0: min
    1: avg
    2: max
    """
    precision: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    recall: list    = field(default_factory=lambda: [0.0, 0.0, 0.0])
    f1: list        = field(default_factory=lambda: [0.0, 0.0, 0.0])

def compute_rouge(
    predictions: Iterable[str],
    references: Iterable[str]
) -> dict[str, RougeMetrics]:
    """Compute mean ROUGE-1/2/L over dataset with preprocessing.

    :param predictions: predicted abstracts
    :param references: reference abstracts
    :return: dictionary with ROUGE metrics
    """

    predictions_tokens = [preprocess(p) for p in predictions]
    references_tokens  = [preprocess(r) for r in references]

    if len(predictions_tokens) != len(references_tokens):
        raise ValueError(f"predictions and references must have same length: {len(predictions_tokens)} != {len(references_tokens)}")

    rouges = ["rouge1", "rouge2", "rougeL"]
    scorer = RougeScorer(rouges, use_stemmer=False)

    n = len(predictions_tokens)
    if n == 0:
        logger.info("Iterable of predictions and references is empty")
        return {k: RougeMetrics() for k in rouges}

    _inf = float("inf")
    stats: dict[str, dict[str, float]] = {
        k: {
            "p_min": _inf,
            "p_sum": 0.0,
            "p_max": float("-inf"),
            "r_min": _inf,
            "r_sum": 0.0,
            "r_max": float("-inf"),
            "f_min": _inf,
            "f_sum": 0.0,
            "f_max": float("-inf"),
        }
        for k in rouges
    }

    for prediction, reference in zip(predictions_tokens, references_tokens, strict=True):
        scores = scorer.score(reference, prediction)
        for k in rouges:
            s = scores[k]
            p = float(s.precision)
            r = float(s.recall)
            f = float(s.fmeasure)
            st = stats[k]
            st["p_min"] = min(st["p_min"], p)
            st["p_sum"] += p
            st["p_max"] = max(st["p_max"], p)
            st["r_min"] = min(st["r_min"], r)
            st["r_sum"] += r
            st["r_max"] = max(st["r_max"], r)
            st["f_min"] = min(st["f_min"], f)
            st["f_sum"] += f
            st["f_max"] = max(st["f_max"], f)

    return {
        k: RougeMetrics(
            precision=[st["p_min"], st["p_sum"] / n, st["p_max"]],
            recall=[st["r_min"], st["r_sum"] / n, st["r_max"]],
            f1=[st["f_min"], st["f_sum"] / n, st["f_max"]],
        )
        for k, st in stats.items()
    }
