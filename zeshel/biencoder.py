from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import BertEncoder


class BiEncoder(nn.Module):
    """Dual-encoder for entity linking.

    Two independent BERT encoders produce fixed-length mention and entity vectors.
    Score = dot_product(mention_emb, entity_emb). Entity embeddings can be
    pre-computed offline; retrieval is a nearest-neighbour search.
    """

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        super().__init__()
        self.mention_encoder = BertEncoder(model_name)
        self.entity_encoder = BertEncoder(model_name)
        self.hidden_size = self.mention_encoder.hidden_size

    def encode_mentions(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.mention_encoder(input_ids, attention_mask)

    def encode_entities(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.entity_encoder(input_ids, attention_mask)

    def forward(
        self,
        mention_ids: torch.Tensor,    # [B, Lm]
        mention_mask: torch.Tensor,   # [B, Lm]
        entity_ids: torch.Tensor,     # [B*(1+K), Le]  gold + K negatives per mention
        entity_mask: torch.Tensor,    # [B*(1+K), Le]
        n_cands: int,                 # 1 + K
    ) -> torch.Tensor:
        """In-batch negative forward pass.  Returns logits [B, B*n_cands] or [B, n_cands]."""
        B = mention_ids.size(0)
        mention_emb = self.encode_mentions(mention_ids, mention_mask)      # [B, D]
        entity_emb  = self.encode_entities(entity_ids, entity_mask)       # [B*n_cands, D]
        # In the pure in-batch case (n_cands == 1) the score matrix is [B, B]:
        # each mention scores against every gold entity in the batch.
        return mention_emb @ entity_emb.T                                  # [B, B*n_cands]


def in_batch_loss(logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over in-batch negatives; labels are the diagonal."""
    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)
    return F.cross_entropy(logits, labels)
