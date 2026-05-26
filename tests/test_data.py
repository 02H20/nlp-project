from pathlib import Path

from rag_zh.data import load_dureader, sample_dataset


FIXTURE = Path(__file__).parent / "fixtures" / "dureader_sample.jsonl"


def test_load_dureader_fixture():
    examples, passages = load_dureader(FIXTURE)

    assert len(examples) == 3
    assert len(passages) == 6
    assert examples[0].question == "杭州亚运会在哪里举办？"
    assert examples[0].answers == ["杭州"]
    assert "p1" in examples[0].positive_passage_ids


def test_sample_dataset_keeps_positive_passages():
    examples, passages = load_dureader(FIXTURE)
    sampled_examples, sampled_passages = sample_dataset(examples, passages, 2, 3, seed=7)

    passage_ids = {passage.id for passage in sampled_passages}
    for example in sampled_examples:
        assert set(example.positive_passage_ids).issubset(passage_ids)
