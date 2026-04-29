import nltk

import networkx as nx
import numpy as np

from razdel import sentenize, tokenize
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
_RU_STOPWORDS = nltk.corpus.stopwords.words('russian')

_morph = MorphAnalyzer()

class TextRankSummarizer:
    @staticmethod
    def _to_sentences(text: str) -> list[str]:
        return [s.text for s in sentenize(text)]

    @staticmethod
    def _to_tokens(sentence: str) -> list[str]:
        return [t.text for t in tokenize(sentence)]

    @staticmethod
    def _preprocess(text: str) -> list[str]:
        sentences = []
        for sentence in TextRankSummarizer._to_sentences(text):
            lemmas = []
            for token in TextRankSummarizer._to_tokens(sentence):
                word = token.lower()
                if word in _RU_STOPWORDS:
                    continue
                lemmas.append(_morph.parse(word)[0].normal_form)
            sentences.append(" ".join(lemmas))
        return sentences

    def summarize(self, texts: list[str], limit: int = 300):
        output: list[str] = []

        for text in texts:
            initial_sentences = TextRankSummarizer._to_sentences(text)
            sentences = TextRankSummarizer._preprocess(text)

            if len(initial_sentences) == 0:
                output.append("")
                continue

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            np.fill_diagonal(similarity_matrix, 0.0)

            graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(graph)

            ranked = sorted(
                [(scores[i], i) for i, s in enumerate(sentences)],
                reverse=True
            )

            # Add sentences while length is less than limit
            summary_len = 0
            summary_sentences_indexes = []
            for _, i in ranked:
                if summary_len + len(initial_sentences[i]) <= limit:
                    summary_sentences_indexes.append(i)
                    summary_len += len(initial_sentences[i])

            summary = " ".join([initial_sentences[i] for i in summary_sentences_indexes])
            summary = summary.strip()

            if len(summary) == 0:
                # Fallback: take best-ranked sentence and trim to limit.
                best_idx = ranked[0][1]
                summary = initial_sentences[best_idx].strip()

            if len(summary) > limit:
                summary = summary[:limit].rstrip()

            output.append(summary)

        return output
