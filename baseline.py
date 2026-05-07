import nltk

import networkx as nx
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import to_tokens, to_sentences, preprocess


class TextRankSummarizer:
    def summarize(self, text: str, limit: int = 300) -> str:
        """Create abstract of given text.

        :param text: initial blob
        :param limit: limit of abstract length in characters
        :return: abstract
        """

        # Need to reconstruct abstract at the end
        initial_sentences = to_sentences(text)

        sentences = preprocess(text, lemmatize=True, keep_sentences=True)
        if len(sentences) == 0:
            return ""

        # Calculate scores
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

        # Fallback: take best-ranked sentence and trim to limit.
        if len(summary) == 0:
            best_idx = ranked[0][1]
            summary = initial_sentences[best_idx].strip()

        return summary
