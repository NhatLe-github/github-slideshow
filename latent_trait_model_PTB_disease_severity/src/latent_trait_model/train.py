"""Training loop for the latent trait severity model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import SyntheticPTBDataset
from .model import LatentTraitSeverityModel


@dataclass
class TrainingConfig:
    """Configuration for the training routine."""

    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    device: str = "cpu"


def train_model(
    dataset: Iterable | None = None,
    config: TrainingConfig | None = None,
) -> LatentTraitSeverityModel:
    """Train the latent trait model on the provided dataset."""

    config = config or TrainingConfig()
    dataset = dataset or SyntheticPTBDataset()

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    device = torch.device(config.device)

    model = LatentTraitSeverityModel(input_dim=dataset.n_features)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            logits, mu, logvar = model(features)
            reconstruction_loss = F.cross_entropy(logits.squeeze(1), targets)
            kl_loss = LatentTraitSeverityModel.kl_divergence(mu, logvar) / len(features)
            loss = reconstruction_loss + 0.1 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(features)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {epoch_loss:.4f}")

    return model


__all__ = ["TrainingConfig", "train_model"]
