"""Smoke tests for the latent trait model pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from latent_trait_model import SyntheticPTBDataset, TrainingConfig, train_model


def test_training_runs_for_one_epoch():
    dataset = SyntheticPTBDataset(n_samples=64, n_features=4, seed=0)
    config = TrainingConfig(epochs=1, batch_size=16)
    model = train_model(dataset=dataset, config=config)
    assert model is not None
    assert model.encoder.fc1.in_features == dataset.n_features
