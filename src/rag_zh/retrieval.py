from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Any, Iterable, Protocol

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


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        ...


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
            RetrievedPassage(
                passage=self.passages[index],
                score=float(score),
                rank=rank,
                metadata={"bm25_rank": rank, "bm25_score": float(score)},
            )
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


class BGEReranker:
    def __init__(self, model_name: str, use_fp16: bool = True):
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install FlagEmbedding before using retrieval.pipeline=bm25_rerank."
            ) from exc
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def score(self, query: str, passages: list[Passage]) -> list[float]:
        pairs = [[query, passage.prompt_text()] for passage in passages]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            return [scores]
        return [float(score) for score in scores]


class BM25RerankRetriever:
    def __init__(self, bm25: BM25Retriever, reranker: BGEReranker, candidate_k: int):
        self.bm25 = bm25
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        if self.candidate_k < top_k:
            raise ValueError(
                f"retrieval.candidate_k ({self.candidate_k}) must be >= retrieval.top_k ({top_k})."
            )
        candidates = self.bm25.search(query, top_k=self.candidate_k)
        reranker_scores = self.reranker.score(query, [item.passage for item in candidates])
        rescored = []
        for item, reranker_score in zip(candidates, reranker_scores):
            metadata = {
                **item.metadata,
                "reranker_score": float(reranker_score),
            }
            rescored.append(
                RetrievedPassage(
                    passage=item.passage,
                    score=float(reranker_score),
                    rank=item.rank,
                    metadata=metadata,
                )
            )
        ranked = sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]
        return [
            RetrievedPassage(
                passage=item.passage,
                score=item.score,
                rank=rank,
                metadata=item.metadata,
            )
            for rank, item in enumerate(ranked, start=1)
        ]


def build_retriever(passages: Iterable[Passage], config: dict[str, Any]) -> Retriever:
    retrieval_config = config.get("retrieval", {})
    pipeline = retrieval_config.get("pipeline", "bm25")
    bm25 = BM25Retriever(passages)
    if pipeline == "bm25":
        return bm25
    if pipeline == "bm25_rerank":
        candidate_k = int(retrieval_config.get("candidate_k", 25))
        reranker_config = config.get("reranker", {})
        reranker = BGEReranker(
            model_name=reranker_config.get("model_name", "BAAI/bge-reranker-v2-m3"),
            use_fp16=bool(reranker_config.get("use_fp16", True)),
        )
        return BM25RerankRetriever(bm25=bm25, reranker=reranker, candidate_k=candidate_k)
    raise ValueError(f"Unknown retrieval.pipeline: {pipeline}")
