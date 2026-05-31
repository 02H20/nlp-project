from pathlib import Path

from rag_zh.data import load_dureader
import pytest

from rag_zh.retrieval import BM25Retriever, BM25RerankRetriever, build_retriever


FIXTURE = Path(__file__).parent / "fixtures" / "dureader_sample.jsonl"


def test_bm25_retrieves_relevant_chinese_passage():
    examples, passages = load_dureader(FIXTURE)
    retriever = BM25Retriever(passages)

    results = retriever.search(examples[1].question, top_k=2)

    assert len(results) == 2
    assert results[0].passage.id == "p3"
    assert results[0].rank == 1
    assert results[0].metadata["bm25_rank"] == 1
    assert isinstance(results[0].metadata["bm25_score"], float)


class FakeReranker:
    def __init__(self, scores_by_id):
        self.scores_by_id = scores_by_id

    def score(self, query, passages):
        return [self.scores_by_id[passage.id] for passage in passages]


def test_bm25_rerank_reorders_candidates():
    examples, passages = load_dureader(FIXTURE)
    bm25 = BM25Retriever(passages)
    reranker = FakeReranker({"p3": 0.1, "p4": 0.9})
    retriever = BM25RerankRetriever(bm25=bm25, reranker=reranker, candidate_k=2)

    results = retriever.search(examples[1].question, top_k=2)

    assert [item.passage.id for item in results] == ["p4", "p3"]
    assert results[0].rank == 1
    assert results[0].score == 0.9
    assert results[0].metadata["bm25_rank"] == 2
    assert results[0].metadata["reranker_score"] == 0.9


def test_bm25_rerank_requires_candidate_k_at_least_top_k():
    _, passages = load_dureader(FIXTURE)
    bm25 = BM25Retriever(passages)
    retriever = BM25RerankRetriever(bm25=bm25, reranker=FakeReranker({}), candidate_k=1)

    with pytest.raises(ValueError, match="candidate_k"):
        retriever.search("问题", top_k=2)


def test_build_retriever_rejects_unknown_pipeline():
    _, passages = load_dureader(FIXTURE)

    with pytest.raises(ValueError, match="Unknown retrieval.pipeline"):
        build_retriever(passages, {"retrieval": {"pipeline": "unknown"}})
