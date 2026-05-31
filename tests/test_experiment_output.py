import json

from rag_zh.experiment import run_experiment
from rag_zh.types import GenerationResult, JudgeResult, Passage, RetrievedPassage


class FakeRetriever:
    def search(self, query, top_k):
        return [
            RetrievedPassage(
                passage=Passage(id="p1", title="标题", text="正文"),
                score=0.8,
                rank=1,
                metadata={"bm25_rank": 2, "bm25_score": 1.5, "reranker_score": 0.8},
            )
        ]


class FakeGenerator:
    def __init__(self, **kwargs):
        pass

    def generate(self, question, passages):
        return GenerationResult(answer="杭州", prompt="prompt")


class FakeJudge:
    def __init__(self, **kwargs):
        pass

    def judge(self, question, reference_answers, prediction):
        return JudgeResult(correct=True, rationale="ok", raw_response="{}")


def test_experiment_writes_rerank_metadata(monkeypatch, tmp_path):
    prepared = tmp_path / "prepared.json"
    prepared.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "id": "q1",
                        "question": "问题",
                        "answers": ["杭州"],
                        "positive_passage_ids": ["p1"],
                        "metadata": {},
                    }
                ],
                "passages": [
                    {
                        "id": "p1",
                        "text": "正文",
                        "title": "标题",
                        "source": "fixture",
                        "metadata": {},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("rag_zh.experiment.build_retriever", lambda passages, config: FakeRetriever())
    monkeypatch.setattr("rag_zh.experiment.HFGenerator", FakeGenerator)
    monkeypatch.setattr("rag_zh.experiment.DeepSeekJudge", FakeJudge)

    output_dir = tmp_path / "results"
    run_experiment(
        {
            "data": {"prepared_path": str(prepared), "seed": 42},
            "retrieval": {"pipeline": "bm25_rerank", "top_k": 1, "candidate_k": 25},
            "generator": {},
            "judge": {},
            "output": {"dir": str(output_dir)},
        }
    )

    first = json.loads((output_dir / "details.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert first["retrieval_pipeline"] == "bm25_rerank"
    assert first["candidate_k"] == 25
    retrieved = first["retrieved"][0]
    assert retrieved["bm25_rank"] == 2
    assert retrieved["bm25_score"] == 1.5
    assert retrieved["reranker_score"] == 0.8
    assert retrieved["text"] == "正文"
