"""Autoregressive (GENRE-style) backend: generate entity names with prefix-trie decoding."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BartTokenizer

from zeshel.autoregressive import AutoregressiveEL, PrefixTrie
from zeshel.data import MentionDataset, collate_mentions
from zeshel.types import Entity, Mention


class AutoregressiveBackend:
    """Entity linker that generates entity names token-by-token.

    A prefix trie constrains decoding to valid entity titles; no entity index
    is required.  Memory cost scales with total title token length.
    """

    name = "autoregressive"

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "facebook/bart-base",
        device: str = "cpu",
        batch_size: int = 8,
        max_input_len: int = 128,
        max_title_len: int = 32,
        num_beams: int = 5,
        tqdm_disabled: bool = False,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.max_title_len = max_title_len
        self.num_beams = num_beams
        self.tqdm_disabled = tqdm_disabled

        if Path(checkpoint_path).exists():
            # Local .pth: {"model": state_dict} produced by train.py
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = AutoregressiveEL(model_name).to(device)
            ckpt = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(ckpt["model"])
        else:
            # HuggingFace model ID (e.g. "facebook/genre-linking-blink")
            self.tokenizer = BartTokenizer.from_pretrained(checkpoint_path)
            self.model = AutoregressiveEL(checkpoint_path).to(device)

        self.model.eval()

    def _build_trie(self, entity_dict: dict[str, Entity]) -> tuple[PrefixTrie, dict[str, str]]:
        """Build prefix trie from all entity titles; return (trie, title→entity_id map)."""
        trie = PrefixTrie()
        title_to_id: dict[str, str] = {}
        for e in entity_dict.values():
            token_ids = self.tokenizer.encode(e.title, add_special_tokens=False)
            trie.insert(token_ids)
            title_to_id[e.title] = e.entity_id
        return trie, title_to_id

    @torch.no_grad()
    def link(
        self,
        mentions: list[Mention],
        entity_dict: dict[str, Entity],
    ) -> tuple[list[list[str]], list[str]]:
        trie, title_to_id = self._build_trie(entity_dict)

        ds = MentionDataset(mentions, self.tokenizer, self.max_input_len)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_mentions)

        all_retrieved: list[list[str]] = []
        all_predictions: list[str] = []

        for batch in tqdm(dl, desc="Autoregressive linking", disable=self.tqdm_disabled, file=sys.stderr):
            outputs = self.model.generate_constrained(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                trie=trie,
                num_beams=self.num_beams,
                max_new_tokens=self.max_title_len if self.max_title_len > 0 else None,
            )
            for seq in outputs:
                title = self.tokenizer.decode(seq, skip_special_tokens=True)
                entity_id = title_to_id.get(title, "")
                # Autoregressive produces one prediction; wrap in list for compatibility
                all_retrieved.append([entity_id] if entity_id else [])
                all_predictions.append(entity_id)

        return all_retrieved, all_predictions
