from __future__ import annotations

import argparse
from pathlib import Path

from .config import deep_update, load_config
from .data import load_dureader, sample_dataset, save_prepared
from .experiment import run_experiment
from .retrieval import build_retriever


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")


def prepare_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.dureader_path:
        config = deep_update(config, {"data": {"dureader_path": args.dureader_path}})
    if args.output:
        config = deep_update(config, {"data": {"prepared_path": args.output}})
    examples, passages = load_dureader(config["data"]["dureader_path"])
    sampled_examples, sampled_passages = sample_dataset(
        examples=examples,
        passages=passages,
        sample_size=int(args.sample_size or config["data"]["sample_size"]),
        corpus_size=int(args.corpus_size or config["data"]["corpus_size"]),
        seed=int(config["data"].get("seed", 42)),
        min_answer_overlap=float(
            args.min_answer_overlap
            if args.min_answer_overlap is not None
            else config["data"].get("min_answer_overlap", 0.0)
        ),
    )
    save_prepared(config["data"]["prepared_path"], sampled_examples, sampled_passages)
    print(
        f"prepared {len(sampled_examples)} examples and {len(sampled_passages)} passages "
        f"-> {config['data']['prepared_path']}"
    )


def retrieve(args: argparse.Namespace) -> None:
    from .data import load_prepared

    config = load_config(args.config)
    examples, passages = load_prepared(config["data"]["prepared_path"])
    retriever = build_retriever(passages, config)
    top_k = int(args.top_k or config["retrieval"]["top_k"])
    for example in examples[: int(args.limit)]:
        print(f"\nQ: {example.question}")
        for item in retriever.search(example.question, top_k):
            extra = ""
            if item.metadata.get("reranker_score") is not None:
                extra = (
                    f" bm25_rank={item.metadata.get('bm25_rank')}"
                    f" bm25_score={item.metadata.get('bm25_score'):.4f}"
                    f" reranker_score={item.metadata.get('reranker_score'):.4f}"
                )
            print(f"{item.rank}. {item.passage.id} score={item.score:.4f}{extra} title={item.passage.title}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-zh")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    add_common_args(prepare)
    prepare.add_argument("--dureader-path")
    prepare.add_argument("--output")
    prepare.add_argument("--sample-size", type=int)
    prepare.add_argument("--corpus-size", type=int)
    prepare.add_argument(
        "--min-answer-overlap",
        type=float,
        help="Keep examples whose answer-token overlap with positive passages is at least this value.",
    )
    prepare.set_defaults(func=prepare_data)

    retrieve_parser = subparsers.add_parser("retrieve")
    add_common_args(retrieve_parser)
    retrieve_parser.add_argument("--top-k", type=int)
    retrieve_parser.add_argument("--limit", type=int, default=3)
    retrieve_parser.set_defaults(func=retrieve)

    run_parser = subparsers.add_parser("run-experiment")
    add_common_args(run_parser)
    run_parser.set_defaults(func=lambda args: print(run_experiment(load_config(args.config))))

    args = parser.parse_args()
    Path("results").mkdir(exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
