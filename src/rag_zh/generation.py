from __future__ import annotations

from pathlib import Path

from .types import GenerationResult, RetrievedPassage


def _looks_like_local_path(model_path: str) -> bool:
    return (
        Path(model_path).is_absolute()
        or model_path.startswith(("./", "../", "~/"))
        or model_path.startswith("models/")
    )


def _validate_model_path(model_path: str) -> None:
    if not model_path:
        raise ValueError("generator.model_path is required. Set GENERATOR_MODEL_PATH or config value.")
    if _looks_like_local_path(model_path) and not Path(model_path).expanduser().exists():
        raise FileNotFoundError(
            "generator.model_path points to a local path that does not exist: "
            f"{model_path!r}. Download the model there or set GENERATOR_MODEL_PATH to an existing "
            "local checkpoint directory."
        )


def build_rag_prompt(question: str, passages: list[RetrievedPassage]) -> str:
    docs = []
    for index, item in enumerate(passages, start=1):
        docs.append(f"文档[{index}]\n{item.passage.prompt_text()}")
    documents = "\n\n".join(docs)
    return (
        "你将看到一个问题和若干中文文档。请只根据文档内容直接回答问题。"
        "如果文档中有部分证据可用于回答，请优先给出基于文档的最佳答案；"
        "只有在所有文档都完全没有相关信息时，才回答“无法确定”。"
        "不要输出推理过程，不要复述问题，不要生成新的问答样例，不要重复输出，"
        "不要编造文档外信息。\n\n"
        f"{documents}\n\n问题：{question}\n答案："
    )


class HFGenerator:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        device: str = "auto",
        repetition_penalty: float = 1.08,
        no_repeat_ngram_size: int = 6,
        use_chat_template: bool = False,
        enable_thinking: bool = False,
    ):
        _validate_model_path(model_path)
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
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.use_chat_template = use_chat_template
        self.enable_thinking = enable_thinking

    def generate(self, question: str, passages: list[RetrievedPassage]) -> GenerationResult:
        prompt = build_rag_prompt(question, passages)
        if self.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=self.enable_thinking,
                ).to(self.model.device)
            except TypeError:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.temperature
        outputs = self.model.generate(**generate_kwargs)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return GenerationResult(answer=answer, prompt=prompt)
