# Latent Trait Model for PTB Disease Severity

This project provides a lightweight Python package for experimenting with latent trait models that estimate pulmonary tuberculosis (PTB) disease severity. The repository scaffold includes modules for loading clinical datasets, defining a probabilistic latent trait model, and training or evaluating the model.

## Project structure

```
latent_trait_model_PTB_disease_severity/
├── README.md                  # Project overview and setup
├── pyproject.toml             # Poetry/PEP 517 build configuration
├── requirements.txt           # Runtime dependencies
├── src/
│   └── latent_trait_model/
│       ├── __init__.py
│       ├── data.py            # Dataset loading / preprocessing utilities
│       ├── model.py           # Latent trait model definition
│       └── train.py           # Training and evaluation loop
└── tests/
    └── test_model.py          # Smoke test for the modelling pipeline
```

## Getting started

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the sample training script**:

   ```bash
   python -m latent_trait_model.train --epochs 10
   ```

4. **Run the tests**:

   ```bash
   pytest
   ```

## Next steps

- Replace the synthetic dataset generator with real PTB clinical data loaders.
- Implement additional evaluation metrics (e.g., AUROC, calibration curves).
- Integrate hyperparameter search and experiment tracking.
- Package the model for deployment in clinical decision support workflows.

## License

This project inherits the repository license.
