"""Data loading for ZeSHEL and TACKBP-2010.

ZeSHEL directory layout (Logeswaran et al. 2019):
    <data_dir>/
        documents/  ← one JSONL file per Wikia domain
        mentions/   ← train.json, valid.json, testa.json, testb.json

Each document line: {"document_id", "corpus", "title", "text"}
Each mention: {"mention_id", "corpus", "text", "context_document_id",
               "start_index", "end_index", "label_document_id", "category"}

TACKBP-2010 expects a normalised format:
    <data_dir>/
        entities.jsonl  ← {"entity_id", "title", "description", "corpus"}
        mentions.json   ← [{"mention_id", "mention_text", "context_left",
                             "context_right", "gold_entity_id", "corpus",
                             "split", "category"}, ...]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .types import Entity, Mention

log = logging.getLogger(__name__)

_CONTEXT_WORDS = 128  # words to keep on each side of the mention
_MENTION_START = "[unused0]"
_MENTION_END   = "[unused1]"

# ZeSHEL split files (real filenames from Logeswaran et al. 2019)
_ZESHEL_SPLITS: dict[str, str] = {
    "train":               "train.json",
    "heldout_train_seen":  "heldout_train_seen.json",
    "heldout_train_unseen":"heldout_train_unseen.json",
    "val":                 "val.json",
    "test":                "test.json",
}


# ---------------------------------------------------------------------------
# Raw data loaders
# ---------------------------------------------------------------------------

def _load_documents(documents_dir: Path) -> dict[str, Entity]:
    entities: dict[str, Entity] = {}
    for fpath in sorted(documents_dir.glob("*.json")):
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # corpus not stored in documents — derive from filename
                corpus = fpath.stem
                entities[d["document_id"]] = Entity(
                    entity_id=d["document_id"],
                    title=d["title"],
                    description=d["text"],
                    corpus=corpus,
                )
    return entities


def _extract_context(
    doc_text: str, start: int, end: int, window: int = _CONTEXT_WORDS
) -> tuple[str, str, str]:
    """Extract mention and surrounding context.

    start/end are whitespace-token indices (0-based, end inclusive),
    as used in the ZeSHEL dataset.
    """
    words = doc_text.split()
    end_excl = end + 1  # make exclusive
    ctx_left  = " ".join(words[max(0, start - window): start])
    mention   = " ".join(words[start: end_excl])
    ctx_right = " ".join(words[end_excl: end_excl + window])
    return ctx_left, mention, ctx_right


def load_zeshel(
    data_dir: Path,
    splits: list[str] | None = None,
    subset: int | float | None = None,
    full_entities: bool = True,
) -> tuple[dict[str, Entity], dict[str, list[Mention]]]:
    """Load ZeSHEL entities and mentions.

    Args:
        full_entities: when True (default) the full entity dict is always returned,
            keeping eval metrics meaningful even under subset. Set False only for
            true smoke runs where speed matters more than correct metrics.

    Returns:
        entities: entity_id → Entity
        mentions_by_split: split → list[Mention]
    """
    documents_dir = data_dir / "documents"
    mentions_dir  = data_dir / "mentions"

    if not documents_dir.exists():
        raise FileNotFoundError(f"ZeSHEL documents dir not found: {documents_dir}")
    if not mentions_dir.exists():
        raise FileNotFoundError(f"ZeSHEL mentions dir not found: {mentions_dir}")

    entities = _load_documents(documents_dir)
    log.info("Loaded %d entities", len(entities))

    splits = splits or list(_ZESHEL_SPLITS.keys())
    mentions_by_split: dict[str, list[Mention]] = {}

    for split in splits:
        fname = _ZESHEL_SPLITS.get(split)
        if fname is None:
            raise ValueError(f"Unknown ZeSHEL split: {split!r}")
        fpath = mentions_dir / fname
        if not fpath.exists():
            log.warning("Mentions file not found, skipping: %s", fpath)
            continue

        raw = [json.loads(line) for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()]
        mentions: list[Mention] = []
        for m in raw:
            doc = entities.get(m["context_document_id"])
            if doc is None:
                continue
            ctx_left, mention_text, ctx_right = _extract_context(
                doc.description, m["start_index"], m["end_index"]
            )
            mentions.append(Mention(
                mention_id=m["mention_id"],
                mention_text=m.get("text", mention_text),  # "text" field in ZeSHEL
                context_left=ctx_left,
                context_right=ctx_right,
                gold_entity_id=m["label_document_id"],
                corpus=m["corpus"],
                split=split,
                category=m.get("category", ""),
            ))
        cap = _resolve_subset(subset, len(mentions))
        if cap is not None:
            mentions = mentions[:cap]
        mentions_by_split[split] = mentions
        log.info("Loaded %d mentions for split=%s", len(mentions), split)

    if subset is not None and not full_entities:
        kept_ids: set[str] = {m.gold_entity_id for ms in mentions_by_split.values() for m in ms}
        entities = {eid: e for eid, e in entities.items() if eid in kept_ids}
        log.info("Subset active: restricted entity index to %d entities", len(entities))

    return entities, mentions_by_split


def load_tackbp(
    data_dir: Path,
    splits: list[str] | None = None,
    subset: int | float | None = None,
) -> tuple[dict[str, Entity], dict[str, list[Mention]]]:
    """Load TACKBP-2010 from the normalised format."""
    entities_path = data_dir / "entities.jsonl"
    mentions_path = data_dir / "mentions.json"

    if not entities_path.exists():
        raise FileNotFoundError(f"TACKBP entities file not found: {entities_path}")
    if not mentions_path.exists():
        raise FileNotFoundError(f"TACKBP mentions file not found: {mentions_path}")

    entities: dict[str, Entity] = {}
    with entities_path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            entities[d["entity_id"]] = Entity(
                entity_id=d["entity_id"],
                title=d["title"],
                description=d.get("description", ""),
                corpus=d.get("corpus", "tackbp"),
            )

    raw_mentions: list[dict] = json.loads(mentions_path.read_text(encoding="utf-8"))
    mentions_by_split: dict[str, list[Mention]] = {}
    for m in raw_mentions:
        s = m.get("split", "test")
        if splits and s not in splits:
            continue
        mentions_by_split.setdefault(s, []).append(Mention(
            mention_id=m["mention_id"],
            mention_text=m["mention_text"],
            context_left=m.get("context_left", ""),
            context_right=m.get("context_right", ""),
            gold_entity_id=m["gold_entity_id"],
            corpus=m.get("corpus", "tackbp"),
            split=s,
            category=m.get("category", ""),
        ))

    mentions_by_split = {
        s: ms[:_resolve_subset(subset, len(ms))] if subset is not None else ms
        for s, ms in mentions_by_split.items()
    }
    return entities, mentions_by_split


# ---------------------------------------------------------------------------
# PyTorch datasets
# ---------------------------------------------------------------------------

def _resolve_subset(subset: int | float | None, total: int) -> int | None:
    """Convert subset to an absolute count. Fractions in (0, 1) are treated as percentages."""
    if subset is None:
        return None
    if 0 < subset < 1:
        return max(1, round(subset * total))
    return int(subset)


def _tok_kw(max_len: int) -> dict:
    """Tokenizer kwargs for length limiting. Values <= 0 mean no user cap (model max applies)."""
    if max_len > 0:
        return {"max_length": max_len, "truncation": True}
    return {"truncation": True}


def _mention_context(m: Mention) -> str:
    return (
        m.context_left + " " + _MENTION_START + " " +
        m.mention_text + " " + _MENTION_END + " " + m.context_right
    )


def _entity_text(e: Entity) -> str:
    return e.title + " : " + e.description


class MentionEntityPairDataset(Dataset):
    """One item per mention: (mention context tokens, gold entity tokens).

    Used for bi-encoder in-batch negative training.
    """

    def __init__(
        self,
        mentions: list[Mention],
        entities: dict[str, Entity],
        tokenizer: AutoTokenizer,
        max_mention_len: int = 128,
        max_entity_len: int = 128,
    ) -> None:
        self.mentions = [m for m in mentions if m.gold_entity_id in entities]
        self.entities = entities
        self.tokenizer = tokenizer
        self.max_mention_len = max_mention_len
        self.max_entity_len = max_entity_len

    def __len__(self) -> int:
        return len(self.mentions)

    def __getitem__(self, idx: int) -> dict:
        m = self.mentions[idx]
        e = self.entities[m.gold_entity_id]
        m_enc = self.tokenizer(
            _mention_context(m),
            **_tok_kw(self.max_mention_len),
            padding=False,
        )
        e_enc = self.tokenizer(
            _entity_text(e),
            **_tok_kw(self.max_entity_len),
            padding=False,
        )
        return {
            "mention_ids":   m_enc["input_ids"],
            "mention_mask":  m_enc["attention_mask"],
            "entity_ids":    e_enc["input_ids"],
            "entity_mask":   e_enc["attention_mask"],
            "gold_entity_id": m.gold_entity_id,
            "mention_id":    m.mention_id,
        }


class MentionDataset(Dataset):
    """One item per mention: tokenised mention context.  Used for evaluation."""

    def __init__(
        self,
        mentions: list[Mention],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.mentions = mentions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.mentions)

    def __getitem__(self, idx: int) -> dict:
        m = self.mentions[idx]
        enc = self.tokenizer(
            _mention_context(m),
            **_tok_kw(self.max_length),
            padding=False,
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "gold_entity_id": m.gold_entity_id,
            "mention_id":     m.mention_id,
            "category":       m.category,
        }


class EntityDataset(Dataset):
    """One item per entity: tokenised title + description.  Used for index building."""

    def __init__(
        self,
        entities: list[Entity],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.entities = entities
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.entities)

    def __getitem__(self, idx: int) -> dict:
        e = self.entities[idx]
        enc = self.tokenizer(
            _entity_text(e),
            **_tok_kw(self.max_length),
            padding=False,
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "entity_id":      e.entity_id,
        }


class CrossEncoderDataset(Dataset):
    """One item per (mention, candidate) pair: concatenated tokens for re-ranking."""

    def __init__(
        self,
        mentions: list[Mention],
        entities: dict[str, Entity],
        candidates: list[list[str]],  # top-k candidate entity_ids per mention
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ) -> None:
        self.items: list[tuple[Mention, Entity, int]] = []
        for m, cands in zip(mentions, candidates):
            gold_pos = next((i for i, c in enumerate(cands) if c == m.gold_entity_id), None)
            if gold_pos is None or m.gold_entity_id not in entities:
                continue
            for cid in cands:
                if cid in entities:
                    self.items.append((m, entities[cid], int(cid == m.gold_entity_id)))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        m, e, label = self.items[idx]
        enc = self.tokenizer(
            _mention_context(m),
            _entity_text(e),
            **_tok_kw(self.max_length),
            padding=False,
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label":          label,
        }


class Seq2SeqDataset(Dataset):
    """One item per mention: (context, entity_title) for autoregressive training."""

    def __init__(
        self,
        mentions: list[Mention],
        entities: dict[str, Entity],
        tokenizer: AutoTokenizer,
        max_input_len: int = 128,
        max_target_len: int = 32,
    ) -> None:
        self.mentions = [m for m in mentions if m.gold_entity_id in entities]
        self.entities = entities
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.mentions)

    def __getitem__(self, idx: int) -> dict:
        m = self.mentions[idx]
        e = self.entities[m.gold_entity_id]
        src_enc = self.tokenizer(
            _mention_context(m),
            **_tok_kw(self.max_input_len),
            padding=False,
        )
        tgt_enc = self.tokenizer(
            e.title,
            **_tok_kw(self.max_target_len),
            padding=False,
        )
        # Shift labels: pad with -100 for ignored positions
        labels = tgt_enc["input_ids"] + [-100]  # append -100 as sentinel
        return {
            "input_ids":      src_enc["input_ids"],
            "attention_mask": src_enc["attention_mask"],
            "labels":         labels,
        }


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def _pad(seqs: list[list[int]], pad_value: int = 0) -> torch.Tensor:
    return pad_sequence(
        [torch.tensor(s, dtype=torch.long) for s in seqs],
        batch_first=True,
        padding_value=pad_value,
    )


def collate_pairs(batch: list[dict]) -> dict:
    return {
        "mention_ids":    _pad([x["mention_ids"]  for x in batch]),
        "mention_mask":   _pad([x["mention_mask"] for x in batch]),
        "entity_ids":     _pad([x["entity_ids"]   for x in batch]),
        "entity_mask":    _pad([x["entity_mask"]  for x in batch]),
        "gold_entity_ids": [x["gold_entity_id"] for x in batch],
        "mention_ids_str": [x["mention_id"]     for x in batch],
    }


def collate_mentions(batch: list[dict]) -> dict:
    return {
        "input_ids":       _pad([x["input_ids"]      for x in batch]),
        "attention_mask":  _pad([x["attention_mask"] for x in batch]),
        "gold_entity_ids": [x["gold_entity_id"] for x in batch],
        "mention_ids":     [x["mention_id"]     for x in batch],
        "categories":      [x["category"]       for x in batch],
    }


def collate_entities(batch: list[dict]) -> dict:
    return {
        "input_ids":      _pad([x["input_ids"]      for x in batch]),
        "attention_mask": _pad([x["attention_mask"] for x in batch]),
        "entity_ids":     [x["entity_id"] for x in batch],
    }


def collate_cross(batch: list[dict]) -> dict:
    return {
        "input_ids":      _pad([x["input_ids"]      for x in batch]),
        "attention_mask": _pad([x["attention_mask"] for x in batch]),
        "labels":         torch.tensor([x["label"] for x in batch], dtype=torch.long),
    }


def collate_seq2seq(batch: list[dict]) -> dict:
    return {
        "input_ids":      _pad([x["input_ids"]      for x in batch]),
        "attention_mask": _pad([x["attention_mask"] for x in batch]),
        "labels":         _pad([x["labels"]         for x in batch], pad_value=-100),
    }


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def build_biencoder_loaders(
    mentions_train: list[Mention],
    mentions_eval: list[Mention],
    entities: dict[str, Entity],
    tokenizer: AutoTokenizer,
    data_cfg,
    runtime_cfg,
    *,
    device: str = "cpu",
) -> tuple[DataLoader, DataLoader]:
    """Build (train_dl, eval_mention_dl) for bi-encoder training."""
    max_m = int(data_cfg.get("max_mention_len", 128))
    max_e = int(data_cfg.get("max_entity_len", 128))
    bs     = max(1, int(runtime_cfg.batch_size))
    nw     = int(runtime_cfg.get("num_workers", 0))

    train_ds = MentionEntityPairDataset(mentions_train, entities, tokenizer, max_m, max_e)
    eval_ds  = MentionDataset(mentions_eval, tokenizer, max_m)

    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, collate_fn=collate_pairs,
        pin_memory=(device == "cuda"),
        persistent_workers=(nw > 0),
    )
    eval_dl = DataLoader(
        eval_ds, batch_size=bs, shuffle=False,
        num_workers=nw, collate_fn=collate_mentions,
        pin_memory=(device == "cuda"),
        persistent_workers=(nw > 0),
    )
    return train_dl, eval_dl


def build_entity_loader(
    entities: list[Entity],
    tokenizer: AutoTokenizer,
    max_entity_len: int,
    batch_size: int,
    *,
    device: str = "cpu",
    num_workers: int = 0,
) -> DataLoader:
    ds = EntityDataset(entities, tokenizer, max_entity_len)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_entities,
        pin_memory=(device == "cuda"),
    )


def build_seq2seq_loader(
    mentions: list[Mention],
    entities: dict[str, Entity],
    tokenizer: AutoTokenizer,
    data_cfg,
    runtime_cfg,
    *,
    device: str = "cpu",
    shuffle: bool = True,
) -> DataLoader:
    max_in  = int(data_cfg.get("max_input_len",  128))
    max_out = int(data_cfg.get("max_target_len", 32))
    bs = max(1, int(runtime_cfg.batch_size))
    nw = int(runtime_cfg.get("num_workers", 0))

    ds = Seq2SeqDataset(mentions, entities, tokenizer, max_in, max_out)
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=nw, collate_fn=collate_seq2seq,
        pin_memory=(device == "cuda"),
        persistent_workers=(nw > 0),
    )
