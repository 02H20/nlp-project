from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .data import load_prepared
from .generation import HFGenerator
from .judge import DeepSeekJudge
from .reorder import available_strategies, reorder_passages
from .retrieval import BM25Retriever


def run_experiment(config: dict[str, Any]) -> dict[str, float]:
    examples, passages = load_prepared(config["data"]["prepared_path"])
    retriever = BM25Retriever(passages)
    generator = HFGenerator(**config["generator"])
    judge = DeepSeekJudge(**config["judge"])

    top_k = int(config["retrieval"]["top_k"])
    seed = int(config["data"].get("seed", 42))
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.jsonl"
    summary_csv = output_dir / "summary.csv"
    summary_md = output_dir / "summary.md"

    totals = {strategy: 0 for strategy in available_strategies()}
    corrects = {strategy: 0 for strategy in available_strategies()}

    with details_path.open("w", encoding="utf-8") as details_file:
        for example_index, example in enumerate(examples):
            retrieved = retriever.search(example.question, top_k=top_k)
            for strategy in available_strategies():
                ordered = reorder_passages(retrieved, strategy, seed=seed + example_index)
                generation = generator.generate(example.question, ordered)
                result = judge.judge(example.question, example.answers, generation.answer)
                totals[strategy] += 1
                corrects[strategy] += int(result.correct)
                details_file.write(
                    json.dumps(
                        {
                            "question_id": example.id,
                            "question": example.question,
                            "answers": example.answers,
                            "strategy": strategy,
                            "prediction": generation.answer,
                            "correct": result.correct,
                            "judge_rationale": result.rationale,
                            "retrieved": [
                                {
                                    "rank": item.rank,
                                    "score": item.score,
                                    "passage_id": item.passage.id,
                                    "title": item.passage.title,
                                }
                                for item in ordered
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        strategy: corrects[strategy] / totals[strategy] if totals[strategy] else 0.0
        for strategy in available_strategies()
    }
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["strategy", "accuracy", "correct", "total"])
        writer.writeheader()
        for strategy in available_strategies():
            writer.writerow(
                {
                    "strategy": strategy,
                    "accuracy": f"{summary[strategy]:.4f}",
                    "correct": corrects[strategy],
                    "total": totals[strategy],
                }
            )

    lines = ["| strategy | accuracy | correct | total |", "|---|---:|---:|---:|"]
    for strategy in available_strategies():
        lines.append(
            f"| {strategy} | {summary[strategy]:.4f} | {corrects[strategy]} | {totals[strategy]} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
