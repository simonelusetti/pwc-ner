from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    title: str
    description: str
    corpus: str


@dataclass(frozen=True, slots=True)
class Mention:
    mention_id: str
    mention_text: str
    context_left: str
    context_right: str
    gold_entity_id: str
    corpus: str
    split: str
    category: str = ""  # HIGH_OVERLAP | AMBIGUOUS | MULTIPLE | LOW_OVERLAP
