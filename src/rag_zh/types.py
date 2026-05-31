from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt_text(self) -> str:
        title = f"标题：{self.title}\n" if self.title else ""
        return f"{title}{self.text}".strip()


@dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    answers: list[str]
    positive_passage_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedPassage:
    passage: Passage
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    prompt: str


@dataclass(frozen=True)
class JudgeResult:
    correct: bool
    rationale: str
    raw_response: str
