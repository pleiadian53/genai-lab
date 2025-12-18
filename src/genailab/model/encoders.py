"""Encoder architectures for gene expression data."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Simple MLP encoder for expression data.

    Args:
        input_dim: Input dimension (number of genes)
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (latent dim * 2 for VAE: mu + logvar)
        dropout: Dropout probability
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[activation]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder treating genes as tokens.

    This is a simplified version for learning purposes.
    For production, consider using pre-trained models like Geneformer.

    Args:
        n_genes: Number of genes (vocabulary size)
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model

        # Gene embedding (each gene gets a learned embedding)
        self.gene_embedding = nn.Embedding(n_genes, d_model)

        # Positional encoding (optional, genes don't have natural order)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_genes, d_model) * 0.02)

        # Expression value projection
        self.expr_proj = nn.Linear(1, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Expression values of shape (B, n_genes)

        Returns:
            Encoded representation of shape (B, d_model)
        """
        B, G = x.shape

        # Gene indices
        gene_idx = torch.arange(G, device=x.device).unsqueeze(0).expand(B, -1)

        # Combine gene embedding + expression value + position
        gene_emb = self.gene_embedding(gene_idx)  # (B, G, d_model)
        expr_emb = self.expr_proj(x.unsqueeze(-1))  # (B, G, d_model)
        h = gene_emb + expr_emb + self.pos_encoding

        # Transformer encoding
        h = self.transformer(h)  # (B, G, d_model)

        # Pool over genes (mean pooling)
        h = h.mean(dim=1)  # (B, d_model)

        return self.output_proj(h)
