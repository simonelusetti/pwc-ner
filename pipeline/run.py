"""Backend protocol and evaluation loop for the entity linking pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from zeshel.metrics import compute_category_metrics, compute_retrieval_metrics
from zeshel.types import Entity, Mention

from .types import ELExample, ELResult, RunResult


@runtime_checkable
class Backend(Protocol):
    name: str

    def link(
        self,
        mentions: list[Mention],
        entity_dict: dict[str, Entity],
    ) -> tuple[list[list[str]], list[str]]:
        """Return (retrieved_top64, predictions) for each mention.

        retrieved_top64: top-64 candidate entity_ids per mention
        predictions:     top-1 prediction per mention
        """
        ...


def run_backend(
    examples: list[ELExample],
    backend: Backend,
) -> RunResult:
    mentions    = [ex.mention for ex in examples]
    entity_dict = examples[0].entity_dict if examples else {}

    retrieved_all, predictions = backend.link(mentions, entity_dict)

    results = [
        ELResult(example=ex, retrieved=ret, prediction=pred)
        for ex, ret, pred in zip(examples, retrieved_all, predictions)
    ]

    gold_ids   = [m.gold_entity_id for m in mentions]
    categories = [m.category for m in mentions]

    cat_metrics = compute_category_metrics(gold_ids, retrieved_all, predictions, categories)
    retr_metrics = compute_retrieval_metrics(gold_ids, retrieved_all)

    metrics = {
        "recall_at_1":          retr_metrics.recall_at_1,
        "recall_at_10":         retr_metrics.recall_at_10,
        "recall_at_64":         cat_metrics.overall.recall_at_64,
        "recall_at_100":        retr_metrics.recall_at_100,
        "normalised_accuracy":  cat_metrics.overall.normalised_accuracy,
        "unnormalised_accuracy": cat_metrics.overall.unnormalised_accuracy,
    }
    for cat_name, cm in [
        ("high_overlap", cat_metrics.high_overlap),
        ("ambiguous",    cat_metrics.ambiguous),
        ("multiple",     cat_metrics.multiple),
        ("low_overlap",  cat_metrics.low_overlap),
    ]:
        if cm is not None:
            metrics[f"{cat_name}_norm_acc"]   = cm.normalised_accuracy
            metrics[f"{cat_name}_unnorm_acc"] = cm.unnormalised_accuracy
            metrics[f"{cat_name}_recall_64"]  = cm.recall_at_64

    return RunResult(
        backend=backend.name,
        dataset=examples[0].mention.corpus if examples else "",
        results=results,
        metrics=metrics,
    )


def load_examples(
    data_dir: Path,
    dataset: str,
    splits: list[str],
    subset: int | None = None,
) -> list[ELExample]:
    """Load EL examples from a prepared data directory.

    Supports datasets: ``zeshel``, ``tackbp``.
    """
    from zeshel.data import load_tackbp, load_zeshel

    if dataset == "zeshel":
        entities, mentions_by_split = load_zeshel(data_dir, splits, subset=subset)
    elif dataset == "tackbp":
        entities, mentions_by_split = load_tackbp(data_dir, splits, subset=subset)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    examples: list[ELExample] = []
    for split in splits:
        for m in mentions_by_split[split]:
            examples.append(ELExample(mention=m, entity_dict=entities))
    return examples
