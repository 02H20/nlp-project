from pathlib import Path

from rag_zh.data import answer_overlap_score, load_dureader, sample_dataset


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


def test_answer_overlap_score_detects_noisy_answer():
    score = answer_overlap_score("如何查看jenkins版本", ["系统设置页面配置权限"], ["linux 下查看 Jenkins 版本号"])

    assert score == 0.0


def test_sample_dataset_can_filter_noisy_examples():
    examples, passages = load_dureader(FIXTURE)
    sampled_examples, _ = sample_dataset(
        examples,
        passages,
        sample_size=3,
        corpus_size=6,
        seed=7,
        min_answer_overlap=0.5,
    )

    assert 0 < len(sampled_examples) < len(examples)
