import re
import nltk

from razdel import sentenize, tokenize
from logger_config import setup_logger
from typing import Union

logger = setup_logger(__name__)


class StopwordsManager:
    RU_STOPWORDS = None

    @classmethod
    def get_ru_stopwords(cls):
        PATH = 'corpora/stopwords'
        if cls.RU_STOPWORDS is None:
            try:
                nltk.data.find(PATH)
                logger.info("Stopwords found in local cache!")
                cls.RU_STOPWORDS = nltk.corpus.stopwords.words('russian')
            except Exception:
                logger.info("Downloading stopwords...")
                nltk.download('stopwords', quiet=True)
                logger.info("Stopwords downloaded!")

            cls.RU_STOPWORDS = nltk.corpus.stopwords.words('russian')
        return cls.RU_STOPWORDS

class MorphManager:
    MORPH = None

    @classmethod
    def get_morth(cls):
        if cls.MORPH is None:
            from pymorphy3 import MorphAnalyzer
            cls.MORPH = MorphAnalyzer()
        return cls.MORPH

def to_sentences(text: str):
    return [s.text for s in sentenize(text)]

def to_tokens(text: str):
    return [t.text for t in tokenize(text)]

def _preprocess(
    text: str,
    *,
    remove_punct=True,
    lower=True,
    remove_stopwords=True,
    lemmatize=False) -> str:

    # Lower text
    if lower:
        text = text.lower()

    # Remove punctuation
    if remove_punct:
        text = re.compile(r"[^\w\s]", re.UNICODE).sub(" ", text)

    # Tokenize
    tokens = [t.text for t in tokenize(text)]

    # Delete stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t and t not in StopwordsManager.get_ru_stopwords()]

    if lemmatize:
        morph = MorphManager.get_morth()
        tokens = [morph.parse(t)[0].normal_form for t in tokens]

    return " ".join(tokens)

def preprocess(
    text: str,
    *,
    remove_punct=True,
    lower=True,
    remove_stopwords=True,
    lemmatize=False,
    keep_sentences=False) -> Union[str, list[str]]:

    preprocessed_kwargs = {
        'remove_punct': remove_punct,
        'lower': lower,
        'remove_stopwords': remove_stopwords,
        'lemmatize': lemmatize
    }

    if not keep_sentences:
        return _preprocess(text, **preprocessed_kwargs)

    output: list[str] = []
    sentences = to_sentences(text)
    for sentence in sentences:
        output.append(_preprocess(sentence, **preprocessed_kwargs))
    return output
