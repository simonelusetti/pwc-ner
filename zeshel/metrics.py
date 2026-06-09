from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_10: float
    recall_at_64: float
    recall_at_100: float


@dataclass
class LinkingMetrics:
    recall_at_64: float
    normalised_accuracy: float    # accuracy on mentions where gold was in top-64
    unnormalised_accuracy: float  # accuracy on all mentions


@dataclass
class CategoryMetrics:
    high_overlap: LinkingMetrics | None
    ambiguous: LinkingMetrics | None
    multiple: LinkingMetrics | None
    low_overlap: LinkingMetrics | None
    overall: LinkingMetrics


def compute_recall_at_k(
    gold_ids: list[str],
    candidates: list[list[str]],
    k: int,
) -> float:
    hits = sum(g in set(c[:k]) for g, c in zip(gold_ids, candidates))
    return hits / max(1, len(gold_ids))


def compute_retrieval_metrics(
    gold_ids: list[str],
    candidates: list[list[str]],  # top-100 candidates per mention
) -> RetrievalMetrics:
    return RetrievalMetrics(
        recall_at_1=compute_recall_at_k(gold_ids, candidates, 1),
        recall_at_10=compute_recall_at_k(gold_ids, candidates, 10),
        recall_at_64=compute_recall_at_k(gold_ids, candidates, 64),
        recall_at_100=compute_recall_at_k(gold_ids, candidates, 100),
    )


def compute_linking_metrics(
    gold_ids: list[str],
    retrieved: list[list[str]],   # top-64 candidates per mention
    predictions: list[str],       # model's top-1 prediction per mention
) -> LinkingMetrics:
    recall_64 = compute_recall_at_k(gold_ids, retrieved, 64)

    in_top64 = [g in set(c[:64]) for g, c in zip(gold_ids, retrieved)]
    norm_correct = sum(
        p == g for p, g, in64 in zip(predictions, gold_ids, in_top64) if in64
    )
    normalised_acc = norm_correct / max(1, sum(in_top64))

    unnorm_correct = sum(p == g for p, g in zip(predictions, gold_ids))
    unnormalised_acc = unnorm_correct / max(1, len(gold_ids))

    return LinkingMetrics(
        recall_at_64=recall_64,
        normalised_accuracy=normalised_acc,
        unnormalised_accuracy=unnormalised_acc,
    )


def compute_category_metrics(
    gold_ids: list[str],
    retrieved: list[list[str]],
    predictions: list[str],
    categories: list[str],
) -> CategoryMetrics:
    subsets = {"HIGH_OVERLAP": [], "AMBIGUOUS": [], "MULTIPLE": [], "LOW_OVERLAP": []}
    for i, cat in enumerate(categories):
        if cat in subsets:
            subsets[cat].append(i)

    def _subset(idxs: list[int]) -> LinkingMetrics | None:
        if not idxs:
            return None
        return compute_linking_metrics(
            [gold_ids[i] for i in idxs],
            [retrieved[i] for i in idxs],
            [predictions[i] for i in idxs],
        )

    return CategoryMetrics(
        high_overlap=_subset(subsets["HIGH_OVERLAP"]),
        ambiguous=_subset(subsets["AMBIGUOUS"]),
        multiple=_subset(subsets["MULTIPLE"]),
        low_overlap=_subset(subsets["LOW_OVERLAP"]),
        overall=compute_linking_metrics(gold_ids, retrieved, predictions),
    )
