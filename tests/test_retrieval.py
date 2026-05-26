from pathlib import Path

from rag_zh.data import load_dureader
from rag_zh.retrieval import BM25Retriever


FIXTURE = Path(__file__).parent / "fixtures" / "dureader_sample.jsonl"


def test_bm25_retrieves_relevant_chinese_passage():
    examples, passages = load_dureader(FIXTURE)
    retriever = BM25Retriever(passages)

    results = retriever.search(examples[1].question, top_k=2)

    assert len(results) == 2
    assert results[0].passage.id == "p3"
    assert results[0].rank == 1
