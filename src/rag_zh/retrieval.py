from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Iterable

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

from .types import Passage, RetrievedPassage


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    if jieba is not None:
        return [token.strip() for token in jieba.cut(text) if token.strip()]
    return TOKEN_PATTERN.findall(text)


class BM25Retriever:
    def __init__(self, passages: Iterable[Passage]):
        self.passages = list(passages)
        self.tokenized_corpus = [tokenize(passage.prompt_text()) for passage in self.passages]
        if BM25Okapi is not None:
            self.backend = BM25Okapi(self.tokenized_corpus)
        else:
            self.backend = None
            self._doc_freq = Counter(token for doc in self.tokenized_corpus for token in set(doc))
            self._avgdl = sum(len(doc) for doc in self.tokenized_corpus) / max(len(self.tokenized_corpus), 1)

    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        query_tokens = tokenize(query)
        if self.backend is not None:
            scores = self.backend.get_scores(query_tokens)
        else:
            scores = [self._fallback_score(query_tokens, doc) for doc in self.tokenized_corpus]
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[:top_k]
        return [
            RetrievedPassage(passage=self.passages[index], score=float(score), rank=rank)
            for rank, (index, score) in enumerate(ranked, start=1)
        ]

    def _fallback_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        counts = Counter(doc_tokens)
        score = 0.0
        n_docs = max(len(self.tokenized_corpus), 1)
        doc_len = max(len(doc_tokens), 1)
        for token in query_tokens:
            if token not in counts:
                continue
            df = self._doc_freq.get(token, 0)
            idf = log((n_docs - df + 0.5) / (df + 0.5) + 1)
            tf = counts[token]
            score += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * doc_len / self._avgdl))
        return score
