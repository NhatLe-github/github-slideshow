"""Latent trait severity model definition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Simple MLP encoder that projects features into latent mean and log-variance."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_logvar = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class OrdinalDecoder(nn.Module):
    """Decoder that maps latent trait to ordinal severity probabilities."""

    def __init__(self, thresholds: list[float]) -> None:
        super().__init__()
        self.thresholds = nn.Parameter(torch.tensor(thresholds, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = []
        thresholds = torch.sort(self.thresholds)[0]
        extended = torch.cat([
            torch.tensor([-torch.inf], device=z.device),
            thresholds,
            torch.tensor([torch.inf], device=z.device),
        ])
        for low, high in zip(extended[:-1], extended[1:]):
            logits.append(-F.relu(low - z) - F.relu(z - high))
        return torch.stack(logits, dim=-1)


@dataclass
class LatentTraitSeverityModel(nn.Module):
    """VAE-style latent trait model with ordinal decoder."""

    input_dim: int
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        super().__init__()
        self.encoder = Encoder(self.input_dim, self.hidden_dim)
        self.decoder = OrdinalDecoder([ -1.0, 0.0, 1.0 ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        logits = self.decoder(z)
        return logits, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


__all__ = ["LatentTraitSeverityModel"]
