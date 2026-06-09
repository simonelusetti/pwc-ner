from __future__ import annotations

from dataclasses import dataclass, field

from zeshel.types import Entity, Mention


@dataclass(frozen=True)
class ELExample:
    mention: Mention
    entity_dict: dict[str, Entity]  # entities available at test time


@dataclass(frozen=True)
class ELResult:
    example: ELExample
    retrieved: list[str]    # top-64 candidate entity_ids
    prediction: str         # top-1 predicted entity_id


@dataclass(frozen=True)
class RunResult:
    backend: str
    dataset: str
    results: list[ELResult]
    metrics: dict[str, float] = field(default_factory=dict)
