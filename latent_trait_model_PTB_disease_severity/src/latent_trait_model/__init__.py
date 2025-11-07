"""Latent trait modelling toolkit for PTB disease severity."""

from .data import SyntheticPTBDataset
from .model import LatentTraitSeverityModel
from .train import TrainingConfig, train_model

__all__ = [
    "SyntheticPTBDataset",
    "LatentTraitSeverityModel",
    "TrainingConfig",
    "train_model",
]
