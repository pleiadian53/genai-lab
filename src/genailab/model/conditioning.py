"""Conditioning embeddings for categorical covariates."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ConditionSpec:
    """Specification for condition variables.

    Attributes:
        n_cats: Mapping from condition name to number of categories
        emb_dim: Embedding dimension per condition
        out_dim: Output dimension after projection
    """

    n_cats: dict[str, int] = field(default_factory=dict)
    emb_dim: int = 32
    out_dim: int = 128


class ConditionEncoder(nn.Module):
    """
    Turns a dict of categorical condition tensors into a single condition vector.

    Example conditions: tissue_id, disease_id, batch_id.

    Args:
        spec: ConditionSpec defining the conditions and dimensions
    """

    def __init__(self, spec: ConditionSpec):
        super().__init__()
        self.spec = spec
        self.embedders = nn.ModuleDict(
            {name: nn.Embedding(n, spec.emb_dim) for name, n in spec.n_cats.items()}
        )
        in_dim = len(spec.n_cats) * spec.emb_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, spec.out_dim),
            nn.SiLU(),
            nn.Linear(spec.out_dim, spec.out_dim),
        )

    def forward(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode conditions to a single vector.

        Args:
            cond: Dict mapping condition name to tensor of shape (B,) with int64 indices

        Returns:
            Condition vector of shape (B, out_dim)
        """
        embs = [self.embedders[name](cond[name]) for name in self.spec.n_cats.keys()]
        x = torch.cat(embs, dim=-1)
        return self.proj(x)

    @property
    def output_dim(self) -> int:
        """Output dimension of the condition vector."""
        return self.spec.out_dim
