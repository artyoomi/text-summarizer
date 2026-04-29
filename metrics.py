import nltk
import re
from typing import Iterable

from dataclasses import dataclass
from razdel import tokenize as razdel_tokenize
from rouge_score import rouge_scorer


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# _RU_STOPWORDS = {
#     "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так",
#     "его","но","да","ты","к","у","же","вы","за","бы","по","ее","мне","было","вот",
#     "от","меня","еще","нет","о","из","ему","теперь","когда","даже","ну","вдруг",
#     "ли","если","уже","или","ни","быть","был","него","до","вас","нибудь","опять",
#     "уж","вам","ведь","там","потом","себя","ничего","ей","может","они","тут",
#     "где","есть","надо","ней","для","мы","тебя","их","чем","была","сам","чтоб",
#     "без","будто","чего","раз","тоже","себе","под","будет","ж","тогда","кто",
#     "этот","того","потому","этого","какой","совсем","ним","здесь","этом","один",
#     "почти","мой","тем","чтобы","нее","сейчас","были","куда","зачем","всех",
#     "никогда","можно","при","наконец","два","об","другой","хоть","после","над",
#     "больше","тот","через","эти","нас","про","всего","них","какая","много",
#     "разве","три","эту","моя","впрочем","хорошо","свою","этой","перед","иногда",
#     "лучше","чуть","том","нельзя","такой","им","более","всегда","конечно",
#     "всю","между"
# }
# nltk.download('stopwords')
# _RU_STOPWORDS = nltk.corpus.stopwords.words('russian')


def _normalize_whitespace(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


def _trim_pred_for_eval(s: str, limit: int = 300) -> str:
    s = _normalize_whitespace(s)
    if len(s) <= limit:
        return s
    return s[:limit].rstrip()


def _prepare_text_for_rouge(text: str, *, is_pred: bool, limit: int = 300) -> str:
    """Normalize, (optionally) trim prediction, drop punctuation, remove stopwords, tokenize with razdel."""
    if is_pred:
        text = _trim_pred_for_eval(text, limit=limit)
    else:
        text = _normalize_whitespace(text)

    text = _PUNCT_RE.sub(" ", text.lower())
    tokens = [t.text for t in razdel_tokenize(text)]
    # tokens = [t for t in tokens if t and t not in _RU_STOPWORDS]
    return " ".join(tokens)


@dataclass(frozen=True)
class RougeTriple:
    precision: float
    recall: float
    f1: float

def compute_rouge(
    predictions: Iterable[str],
    references: Iterable[str],
    *,
    limit: int = 300,
) -> dict[str, RougeTriple]:
    """
    Compute mean ROUGE-1/2/L over dataset with RU-specific preprocessing:
    - predictions trimmed to first `limit` chars
    - punctuation removed
    - razdel tokenization
    - Russian stopwords removed
    """
    preds = [_prepare_text_for_rouge(p, is_pred=True, limit=limit) for p in predictions]
    refs = [_prepare_text_for_rouge(r, is_pred=False, limit=limit) for r in references]

    if len(preds) != len(refs):
        raise ValueError(f"predictions and references must have same length: {len(preds)} != {len(refs)}")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    acc = {
        "rouge1": [0.0, 0.0, 0.0],
        "rouge2": [0.0, 0.0, 0.0],
        "rougeL": [0.0, 0.0, 0.0],
    }

    n = len(preds)
    if n == 0:
        return {k: RougeTriple(0.0, 0.0, 0.0) for k in acc}

    for pred, ref in zip(preds, refs, strict=True):
        scores = scorer.score(ref, pred)
        for k in acc:
            s = scores[k]
            acc[k][0] += float(s.precision)
            acc[k][1] += float(s.recall)
            acc[k][2] += float(s.fmeasure)

    return {
        k: RougeTriple(acc[k][0] / n, acc[k][1] / n, acc[k][2] / n)
        for k in acc
    }
