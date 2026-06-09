"""Bi-encoder backend: encode mentions and entities independently, retrieve via NN."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from zeshel.biencoder import BiEncoder
from zeshel.data import EntityDataset, MentionDataset, collate_entities, collate_mentions
from zeshel.encoder import build_tokenizer
from zeshel.index import EntityIndex
from zeshel.types import Entity, Mention


class BiEncoderBackend:
    name = "biencoder"
    _CACHE_VERSION = 1

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "bert-base-uncased",
        device: str = "cpu",
        batch_size: int = 32,
        max_mention_len: int = 128,
        max_entity_len: int = 128,
        retrieval_k: int = 64,
        cache_dir: str | None = None,
        tqdm_disabled: bool = False,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.max_mention_len = max_mention_len
        self.max_entity_len = max_entity_len
        self.retrieval_k = retrieval_k
        self.tqdm_disabled = tqdm_disabled
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else self.checkpoint_path.parent / ".entity_index_cache"
        )

        self.tokenizer = build_tokenizer(model_name)
        self.model = BiEncoder(model_name).to(device)

        ckpt = torch.load(self.checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _entity_signature(self, entity_dict: dict[str, Entity]) -> str:
        digest = hashlib.sha256()
        for entity_id in sorted(entity_dict):
            entity = entity_dict[entity_id]
            digest.update(entity.entity_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(entity.title.encode("utf-8"))
            digest.update(b"\0")
            digest.update(entity.description.encode("utf-8"))
            digest.update(b"\0")
            digest.update(entity.corpus.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _cache_path(self, entity_dict: dict[str, Entity]) -> Path:
        checkpoint_stat = self.checkpoint_path.stat()
        payload = (
            f"{self._CACHE_VERSION}|{self.checkpoint_path}|"
            f"{checkpoint_stat.st_mtime_ns}|{checkpoint_stat.st_size}|"
            f"{self.model_name}|{self.max_entity_len}|{self._entity_signature(entity_dict)}"
        )
        cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{cache_key}.pt"

    def _load_cached_index(self, cache_path: Path) -> EntityIndex | None:
        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception as exc:
            print(f"Failed to load cached entity index {cache_path}: {exc}", file=sys.stderr)
            return None

        if not isinstance(payload, dict):
            return None
        if payload.get("cache_version") != self._CACHE_VERSION:
            return None

        entity_ids = payload.get("entity_ids")
        embeddings = payload.get("embeddings")
        if not isinstance(entity_ids, list) or embeddings is None:
            return None

        index = EntityIndex()
        index.build(entity_ids, embeddings)
        return index

    @torch.no_grad()
    def _embed_entities(self, entity_dict: dict[str, Entity]) -> EntityIndex:
        cache_path = self._cache_path(entity_dict)
        if cache_path.exists():
            cached_index = self._load_cached_index(cache_path)
            if cached_index is not None:
                print(f"Loaded cached entity index from {cache_path}", file=sys.stderr)
                return cached_index

        entities = list(entity_dict.values())
        if not entities:
            index = EntityIndex()
            index.build([], torch.empty((0, self.model.hidden_size)))
            return index

        ds = EntityDataset(entities, self.tokenizer, self.max_entity_len)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_entities)

        embeddings = torch.empty(len(entities), self.model.hidden_size)
        all_ids: list[str] = []
        offset = 0
        for batch in tqdm(dl, desc="Encoding entities", disable=self.tqdm_disabled, file=sys.stderr):
            emb = self.model.encode_entities(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            ).cpu()
            b = emb.size(0)
            embeddings[offset:offset + b] = emb
            offset += b
            all_ids.extend(batch["entity_ids"])

        index = EntityIndex()
        index.build(all_ids, embeddings)

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "cache_version": self._CACHE_VERSION,
                    "entity_ids": all_ids,
                    "embeddings": embeddings,
                },
                cache_path,
            )
            print(f"Saved entity index cache to {cache_path}", file=sys.stderr)
        except Exception as exc:
            print(f"Failed to save entity index cache {cache_path}: {exc}", file=sys.stderr)

        return index

    @torch.no_grad()
    def link(
        self,
        mentions: list[Mention],
        entity_dict: dict[str, Entity],
    ) -> tuple[list[list[str]], list[str]]:
        index = self._embed_entities(entity_dict)

        ds = MentionDataset(mentions, self.tokenizer, self.max_mention_len)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_mentions)

        all_retrieved: list[list[str]] = []
        all_predictions: list[str] = []

        for batch in tqdm(dl, desc="Linking mentions", disable=self.tqdm_disabled, file=sys.stderr):
            mention_emb = self.model.encode_mentions(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
            top_ids, _ = index.search(mention_emb, k=max(100, self.retrieval_k))
            for row in top_ids:
                all_retrieved.append(row[:self.retrieval_k])
                all_predictions.append(row[0] if row else "")

        return all_retrieved, all_predictions
