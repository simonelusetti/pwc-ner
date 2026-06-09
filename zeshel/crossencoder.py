from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import BertEncoder


class CrossEncoder(nn.Module):
    """Cross-encoder re-ranker for entity linking.

    Concatenates mention context and entity description into a single BERT input
    with full bidirectional cross-attention. Produces a scalar relevance score.
    Used as a re-ranker over the top-k candidates from the bi-encoder.
    """

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        super().__init__()
        self.encoder = BertEncoder(model_name)
        self.scorer = nn.Linear(self.encoder.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, L]  mention_tokens [SEP] entity_tokens
        attention_mask: torch.Tensor, # [B, L]
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns scalar logits [B]."""
        cls = self.encoder(input_ids, attention_mask)
        return self.scorer(cls).squeeze(-1)


def reranking_loss(
    logits: torch.Tensor,   # [B, n_cands]  scores for each candidate
    gold_idx: torch.Tensor, # [B]            which candidate is gold (usually 0)
) -> torch.Tensor:
    return F.cross_entropy(logits, gold_idx)
