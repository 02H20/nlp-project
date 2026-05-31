from rag_zh.generation import HFGenerator, build_rag_prompt
from rag_zh.types import Passage, RetrievedPassage


def test_prompt_prefers_best_effort_before_unknown():
    passages = [RetrievedPassage(Passage(id="p1", text="杭州亚运会在杭州举办。"), 1.0, 1)]

    prompt = build_rag_prompt("杭州亚运会在哪里举办？", passages)

    assert "优先给出基于文档的最佳答案" in prompt
    assert "只有在所有文档都完全没有相关信息时" in prompt
    assert "不要复述问题" in prompt
    assert "不要生成新的问答样例" in prompt


def test_hf_generator_accepts_repetition_defaults_without_loading_model(monkeypatch):
    monkeypatch.setattr("rag_zh.generation._validate_model_path", lambda model_path: None)

    class FakeTokenizer:
        eos_token_id = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeModel:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    import sys
    import types

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForCausalLM=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    generator = HFGenerator(
        model_path="fake-model",
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
    )

    assert generator.repetition_penalty == 1.2
    assert generator.no_repeat_ngram_size == 4


def test_hf_generator_can_use_chat_template_without_changing_rag_prompt(monkeypatch):
    monkeypatch.setattr("rag_zh.generation._validate_model_path", lambda model_path: None)

    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        eos_token_id = 0
        rendered_prompt = ""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            self.rendered_prompt = messages[0]["content"]
            assert kwargs["enable_thinking"] is False
            return FakeBatch(input_ids=[[1, 2]])

        def decode(self, *args, **kwargs):
            return "杭州"

    class FakeModel:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    import sys
    import types

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForCausalLM=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    generator = HFGenerator(model_path="fake-model", use_chat_template=True)
    passage = RetrievedPassage(Passage(id="p1", text="杭州亚运会在杭州举办。"), 1.0, 1)
    result = generator.generate("杭州亚运会在哪里举办？", [passage])

    assert result.answer == "杭州"
    assert "文档[1]" in generator.tokenizer.rendered_prompt
    assert "杭州亚运会在哪里举办？" in generator.tokenizer.rendered_prompt
