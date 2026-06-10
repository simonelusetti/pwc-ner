"""Cross-encoder backend: re-rank bi-encoder candidates with full cross-attention."""
from __future__ import annotations

import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from zeshel.crossencoder import CrossEncoder
from zeshel.data import CrossEncoderDataset, collate_cross
from zeshel.encoder import build_tokenizer
from zeshel.types import Entity, Mention

from .biencoder import BiEncoderBackend


class CrossEncoderBackend:
    """Two-stage linker: bi-encoder retrieval → cross-encoder re-ranking."""

    name = "crossencoder"

    def __init__(
        self,
        biencoder_checkpoint: str,
        crossencoder_checkpoint: str,
        model_name: str = "bert-base-uncased",
        device: str = "cpu",
        batch_size: int = 32,
        max_mention_len: int = 128,
        max_entity_len: int = 128,
        max_cross_len: int = 256,
        retrieval_k: int = 64,
        tqdm_disabled: bool = False,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.max_cross_len = max_cross_len
        self.retrieval_k = retrieval_k
        self.tqdm_disabled = tqdm_disabled

        self.tokenizer = build_tokenizer(model_name)

        self.retriever = BiEncoderBackend(
            checkpoint_path=biencoder_checkpoint,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_mention_len=max_mention_len,
            max_entity_len=max_entity_len,
            retrieval_k=retrieval_k,
            tqdm_disabled=tqdm_disabled,
        )

        self.reranker = CrossEncoder(model_name).to(device)
        ckpt = torch.load(crossencoder_checkpoint, map_location=device)
        # strict=False: BLINK-converted checkpoints omit the scorer head (random init is fine)
        self.reranker.load_state_dict(ckpt["model"], strict=False)
        self.reranker.eval()

    @torch.no_grad()
    def link(
        self,
        mentions: list[Mention],
        entity_dict: dict[str, Entity],
    ) -> tuple[list[list[str]], list[str]]:
        # Stage 1: retrieve top-64 candidates with bi-encoder
        retrieved_all, _ = self.retriever.link(mentions, entity_dict)

        # Stage 2: re-rank each candidate list with cross-encoder
        all_retrieved: list[list[str]] = []
        all_predictions: list[str] = []

        for mention, cands in tqdm(
            zip(mentions, retrieved_all),
            total=len(mentions),
            desc="Cross-encoding",
            disable=self.tqdm_disabled,
            file=sys.stderr,
        ):
            cand_entities = [entity_dict[c] for c in cands if c in entity_dict]
            if not cand_entities:
                all_retrieved.append(cands)
                all_predictions.append(cands[0] if cands else "")
                continue

            ds = CrossEncoderDataset(
                [mention] * len(cand_entities),
                {e.entity_id: e for e in cand_entities},
                [[e.entity_id for e in cand_entities]],
                self.tokenizer,
                self.max_cross_len,
            )
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_cross)

            scores: list[float] = []
            for batch in dl:
                logits = self.reranker(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                scores.extend(logits.cpu().tolist())

            ranked = sorted(zip(scores, [e.entity_id for e in cand_entities]), reverse=True)
            ranked_ids = [eid for _, eid in ranked]
            all_retrieved.append(ranked_ids[:self.retrieval_k])
            all_predictions.append(ranked_ids[0] if ranked_ids else "")

        return all_retrieved, all_predictions
