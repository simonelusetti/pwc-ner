from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


_DEFAULT_MODEL = "bert-base-uncased"


class BertEncoder(nn.Module):
    """BERT backbone that returns the [CLS] representation."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size: int = self.bert.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B, D] CLS vectors."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]


def build_tokenizer(model_name: str = _DEFAULT_MODEL) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
