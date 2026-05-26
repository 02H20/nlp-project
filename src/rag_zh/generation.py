from __future__ import annotations

from .types import GenerationResult, RetrievedPassage


def build_rag_prompt(question: str, passages: list[RetrievedPassage]) -> str:
    docs = []
    for index, item in enumerate(passages, start=1):
        docs.append(f"文档[{index}]\n{item.passage.prompt_text()}")
    documents = "\n\n".join(docs)
    return (
        "你将看到一个问题和若干中文文档。请只根据文档内容直接回答问题，"
        "不要输出推理过程；如果文档无法回答，请回答“无法确定”。\n\n"
        f"{documents}\n\n问题：{question}\n答案："
    )


class HFGenerator:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        device: str = "auto",
    ):
        if not model_path:
            raise ValueError("generator.model_path is required. Set GENERATOR_MODEL_PATH or config value.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install torch and transformers before running generation.") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if device == "auto":
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, question: str, passages: list[RetrievedPassage]) -> GenerationResult:
        prompt = build_rag_prompt(question, passages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return GenerationResult(answer=answer, prompt=prompt)
