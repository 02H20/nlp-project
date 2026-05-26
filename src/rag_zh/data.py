from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from .types import Passage, QAExample


def iter_json_records(path: str | Path) -> Iterable[dict[str, Any]]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.rglob("*.json")) + sorted(root.rglob("*.jsonl"))
    for file_path in files:
        if file_path.name.startswith("."):
            continue
        with file_path.open("r", encoding="utf-8") as file:
            first = file.read(1)
            file.seek(0)
            if first == "[":
                data = json.load(file)
                if isinstance(data, list):
                    yield from (item for item in data if isinstance(item, dict))
                elif isinstance(data, dict):
                    yield data
                continue
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    yield item


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_as_text(item) for item in value if _as_text(item)).strip()
    return ""


def _extract_answers(record: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("answers", "answer", "gold_answers", "reference_answers"):
        value = record.get(key)
        if isinstance(value, list):
            candidates.extend(_as_text(item) for item in value)
        elif isinstance(value, str):
            candidates.append(value.strip())
    return [item for item in dict.fromkeys(candidates) if item]


def _extract_documents(record: dict[str, Any]) -> list[dict[str, Any]]:
    docs = record.get("documents") or record.get("docs") or record.get("passages") or []
    return docs if isinstance(docs, list) else []


def _document_text(document: dict[str, Any]) -> str:
    for key in ("paragraphs", "segmented_paragraphs", "sentences", "text", "content", "passage"):
        text = _as_text(document.get(key))
        if text:
            return text
    return ""


def _document_title(document: dict[str, Any]) -> str:
    return _as_text(document.get("title") or document.get("doc_title") or document.get("source"))


def load_dureader(path: str | Path) -> tuple[list[QAExample], list[Passage]]:
    examples: list[QAExample] = []
    passages: dict[str, Passage] = {}

    for index, record in enumerate(iter_json_records(path)):
        question = _as_text(record.get("question") or record.get("query"))
        answers = _extract_answers(record)
        if not question or not answers:
            continue

        example_id = str(record.get("question_id") or record.get("id") or f"q{index}")
        positive_ids: list[str] = []
        for doc_index, document in enumerate(_extract_documents(record)):
            if not isinstance(document, dict):
                continue
            text = _document_text(document)
            if not text:
                continue
            passage_id = str(document.get("id") or document.get("doc_id") or f"{example_id}_d{doc_index}")
            title = _document_title(document)
            passages.setdefault(
                passage_id,
                Passage(
                    id=passage_id,
                    text=text,
                    title=title,
                    source="dureader",
                    metadata={"question_id": example_id},
                ),
            )
            if document.get("is_selected") is True or any(answer in text for answer in answers):
                positive_ids.append(passage_id)

        examples.append(
            QAExample(
                id=example_id,
                question=question,
                answers=answers,
                positive_passage_ids=positive_ids,
                metadata={"source": "dureader"},
            )
        )

    if not examples:
        raise ValueError(f"No DuReader-style QA examples found under {path}")
    if not passages:
        raise ValueError(f"No passages found under {path}")
    return examples, list(passages.values())


def sample_dataset(
    examples: list[QAExample],
    passages: list[Passage],
    sample_size: int,
    corpus_size: int,
    seed: int,
) -> tuple[list[QAExample], list[Passage]]:
    rng = random.Random(seed)
    selected_examples = examples[:]
    rng.shuffle(selected_examples)
    selected_examples = selected_examples[: min(sample_size, len(selected_examples))]

    positive_ids = {pid for example in selected_examples for pid in example.positive_passage_ids}
    selected_passages: dict[str, Passage] = {
        passage.id: passage for passage in passages if passage.id in positive_ids
    }

    remaining = [passage for passage in passages if passage.id not in selected_passages]
    rng.shuffle(remaining)
    for passage in remaining:
        if len(selected_passages) >= corpus_size:
            break
        selected_passages[passage.id] = passage

    return selected_examples, list(selected_passages.values())


def save_prepared(path: str | Path, examples: list[QAExample], passages: list[Passage]) -> None:
    output = {
        "examples": [example.__dict__ for example in examples],
        "passages": [passage.__dict__ for passage in passages],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def load_prepared(path: str | Path) -> tuple[list[QAExample], list[Passage]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {source}. Run `rag-zh prepare-data` first."
        )
    data = json.loads(source.read_text(encoding="utf-8"))
    examples = [QAExample(**item) for item in data["examples"]]
    passages = [Passage(**item) for item in data["passages"]]
    return examples, passages
