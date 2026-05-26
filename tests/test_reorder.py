from rag_zh.reorder import reorder_passages
from rag_zh.types import Passage, RetrievedPassage


def items(n=5):
    return [
        RetrievedPassage(Passage(id=str(index), text=f"text {index}"), score=1.0 / index, rank=index)
        for index in range(1, n + 1)
    ]


def ids(values):
    return [item.passage.id for item in values]


def test_reorder_sequential_and_inverse():
    retrieved = items()

    assert ids(reorder_passages(retrieved, "sequential")) == ["1", "2", "3", "4", "5"]
    assert ids(reorder_passages(retrieved, "inverse")) == ["5", "4", "3", "2", "1"]


def test_reorder_shuffle_is_seeded():
    retrieved = items()

    assert ids(reorder_passages(retrieved, "shuffle", seed=3)) == ids(
        reorder_passages(retrieved, "shuffle", seed=3)
    )


def test_reorder_max_relevance_and_min_distraction():
    retrieved = items()

    assert ids(reorder_passages(retrieved, "max_relevance")) == ["2", "5", "4", "3", "1"]
    assert ids(reorder_passages(retrieved, "min_distraction")) == ["4", "2", "1", "3", "5"]
