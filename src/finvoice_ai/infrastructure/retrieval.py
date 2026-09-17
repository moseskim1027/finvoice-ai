import math
import re
from collections import Counter

from finvoice_ai.application.ports import RetrievedDocument


class BM25Retriever:
    """Small deterministic BM25 implementation for the approved knowledge base."""

    def __init__(
        self,
        documents: list[RetrievedDocument],
        *,
        top_k: int = 3,
        minimum_score: float = 0.1,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25Retriever requires at least one document")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        self._documents = documents
        self._top_k = top_k
        self._minimum_score = minimum_score
        self._k1 = k1
        self._b = b
        self._tokens = [self.tokenize(f"{doc.title} {doc.content}") for doc in documents]
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        self._average_length = sum(map(len, self._tokens)) / len(self._tokens)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.casefold())

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        query_terms = set(self.tokenize(query))
        if not query_terms:
            return []

        scored = [
            (self._score(query_terms, index), document)
            for index, document in enumerate(self._documents)
        ]
        ranked = sorted(scored, key=lambda item: (-item[0], item[1].document_id))
        return [
            RetrievedDocument(
                document_id=document.document_id,
                title=document.title,
                content=document.content,
                source=document.source,
                score=score,
            )
            for score, document in ranked[: self._top_k]
            if score >= self._minimum_score
        ]

    def _score(self, query_terms: set[str], document_index: int) -> float:
        frequencies = self._frequencies[document_index]
        document_length = len(self._tokens[document_index])
        score = 0.0

        for term in query_terms:
            term_frequency = frequencies[term]
            if term_frequency == 0:
                continue

            document_frequency = sum(term in tokens for tokens in self._tokens)
            inverse_document_frequency = math.log(
                1 + (len(self._documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            length_normalization = 1 - self._b + self._b * (document_length / self._average_length)
            score += (
                inverse_document_frequency
                * (term_frequency * (self._k1 + 1))
                / (term_frequency + self._k1 * length_normalization)
            )

        return score
