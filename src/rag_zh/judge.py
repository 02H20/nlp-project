from __future__ import annotations

import json
import re

from .types import JudgeResult


JUDGE_SYSTEM_PROMPT = (
    "你是中文问答评估裁判。请判断模型回答是否与任一参考答案语义一致。"
    "只输出 JSON，字段为 correct(boolean) 和 rationale(string)。"
)


def build_judge_prompt(question: str, reference_answers: list[str], prediction: str) -> str:
    refs = "\n".join(f"- {answer}" for answer in reference_answers)
    return (
        f"问题：{question}\n"
        f"参考答案：\n{refs}\n"
        f"模型回答：{prediction}\n\n"
        "如果模型回答表达了参考答案中的核心事实，则 correct 为 true；"
        "如果答非所问、无法确定、缺少核心事实或与参考答案矛盾，则 correct 为 false。"
    )


def parse_judge_response(text: str) -> JudgeResult:
    raw = text.strip()
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return JudgeResult(
                correct=bool(data.get("correct")),
                rationale=str(data.get("rationale", "")),
                raw_response=raw,
            )
        except json.JSONDecodeError:
            pass
    lowered = raw.lower()
    negative_tokens = ("false", "不正确", "错误", "否", "no", "incorrect")
    positive_tokens = ("true", "正确", "是", "yes", "correct")
    correct = False if any(token in lowered for token in negative_tokens) else any(
        token in lowered for token in positive_tokens
    )
    return JudgeResult(correct=correct, rationale=raw, raw_response=raw)


class DeepSeekJudge:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("judge.api_key is required. Set DEEPSEEK_API_KEY or config value.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install openai before running DeepSeek judge.") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def judge(self, question: str, reference_answers: list[str], prediction: str) -> JudgeResult:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_prompt(question, reference_answers, prediction)},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return parse_judge_response(content)
