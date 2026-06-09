from __future__ import annotations

import torch


class EntityIndex:
    """Brute-force nearest-neighbour index over pre-computed entity embeddings.

    Scales to ~hundreds of thousands of entities on CPU. For millions of entities,
    swap the matmul for faiss (drop-in replacement: build with faiss.IndexFlatIP).
    """

    def __init__(self) -> None:
        self._entity_ids: list[str] = []
        self._embeddings: torch.Tensor | None = None

    def build(self, entity_ids: list[str], embeddings: torch.Tensor) -> None:
        self._entity_ids = list(entity_ids)
        self._embeddings = embeddings.detach().float().cpu()

    def search(
        self, query: torch.Tensor, k: int
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Return top-k entity_ids and scores for each query vector."""
        if self._embeddings is None:
            raise RuntimeError("Index not built — call build() first.")
        scores = query.detach().float().cpu() @ self._embeddings.T  # [B, N]
        top_k = torch.topk(scores, k=min(k, scores.size(1)), dim=1)
        top_k_ids = [[self._entity_ids[i] for i in row] for row in top_k.indices.tolist()]
        top_k_scores = top_k.values.tolist()
        return top_k_ids, top_k_scores

    def __len__(self) -> int:
        return len(self._entity_ids)
