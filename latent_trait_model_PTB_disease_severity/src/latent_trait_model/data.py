"""Data utilities for the PTB disease severity latent trait model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticPTBDataset(Dataset):
    """Synthetic dataset mimicking PTB severity assessments.

    Each sample contains:
    - a vector of clinical features (continuous and binary)
    - an ordinal severity score derived from a latent trait
    """

    n_samples: int = 512
    n_features: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.features = rng.normal(size=(self.n_samples, self.n_features))
        latent_trait = rng.normal(size=self.n_samples)

        # Simulate feature influence on the latent trait
        weights = rng.normal(size=self.n_features)
        severity_score = self.features @ weights + latent_trait

        # Convert latent scores to ordinal categories (0-3)
        thresholds = np.quantile(severity_score, [0.25, 0.5, 0.75])
        self.targets = np.digitize(severity_score, thresholds)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(self.features[index], dtype=torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return features, target


__all__ = ["SyntheticPTBDataset"]
