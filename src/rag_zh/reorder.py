from __future__ import annotations

import random

from .types import RetrievedPassage


MAX_RELEVANCE_ORDER = [5, 1, 4, 3, 2]
MIN_DISTRACTION_ORDER = [3, 2, 4, 1, 5]


def _place_by_position_order(items: list[RetrievedPassage], position_order: list[int]) -> list[RetrievedPassage]:
    result: list[RetrievedPassage | None] = [None] * len(items)
    for item, one_based_position in zip(items, position_order):
        if one_based_position <= len(items):
            result[one_based_position - 1] = item
    leftovers = [item for item in items if item not in result]
    for index, current in enumerate(result):
        if current is None:
            result[index] = leftovers.pop(0)
    return [item for item in result if item is not None]


def reorder_passages(
    retrieved: list[RetrievedPassage],
    strategy: str,
    seed: int = 42,
) -> list[RetrievedPassage]:
    items = list(retrieved)
    if strategy == "sequential":
        return items
    if strategy == "inverse":
        return list(reversed(items))
    if strategy == "shuffle":
        rng = random.Random(seed)
        rng.shuffle(items)
        return items
    if strategy == "max_relevance":
        return _place_by_position_order(items, MAX_RELEVANCE_ORDER)
    if strategy == "min_distraction":
        return _place_by_position_order(items, MIN_DISTRACTION_ORDER)
    raise ValueError(f"Unknown ordering strategy: {strategy}")


def available_strategies() -> list[str]:
    return ["sequential", "inverse", "shuffle", "max_relevance", "min_distraction"]
