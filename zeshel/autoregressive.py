from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration


class PrefixTrie:
    """Token-ID trie built from all valid entity titles.

    Used to constrain autoregressive decoding so every beam hypothesis
    terminates at a known entity name.
    """

    def __init__(self) -> None:
        self._children: dict[int, "PrefixTrie"] = {}
        self.is_end: bool = False

    def insert(self, token_ids: list[int]) -> None:
        node = self
        for tid in token_ids:
            if tid not in node._children:
                node._children[tid] = PrefixTrie()
            node = node._children[tid]
        node.is_end = True

    def allowed_next(self, prefix: list[int]) -> list[int]:
        node = self
        for tid in prefix:
            if tid not in node._children:
                return []
            node = node._children[tid]
        return list(node._children.keys())


class AutoregressiveEL(nn.Module):
    """BART-based entity linker with prefix-trie-constrained decoding.

    The encoder reads the mention-in-context; the decoder generates the entity
    title token-by-token. A prefix trie over all valid entity names constrains
    each decoding step so every beam terminates at a known entity.
    Memory cost scales with total title token count, not embedding dimensionality.
    """

    def __init__(self, model_name: str = "facebook/bart-base") -> None:
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forcing seq2seq loss."""
        return self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss

    def generate_constrained(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trie: PrefixTrie,
        eos_token_id: int | None = None,
        num_beams: int = 5,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """Generate entity titles constrained by the prefix trie."""
        eos = eos_token_id or self.bart.config.eos_token_id

        def _prefix_allowed(batch_id: int, sent: torch.Tensor) -> list[int]:
            prefix = sent[1:].tolist()  # skip decoder BOS
            allowed = trie.allowed_next(prefix)
            return allowed if allowed else [eos]

        kwargs: dict = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens

        return self.bart.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            prefix_allowed_tokens_fn=_prefix_allowed,
            early_stopping=True,
            **kwargs,
        )
