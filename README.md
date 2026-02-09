# deeponet-pricing

DeepONet-based option pricing for Heston and rBergomi models.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

### Train via CLI

```bash
# Heston implied-vol surface
deeponet-pricing train configs/heston_ivol.yaml

# rBergomi with MC data generation
deeponet-pricing train configs/rbergomi_curve.yaml

# Override config values
deeponet-pricing train configs/heston_ivol.yaml --iterations 100000 --lr 5e-4
```

### Evaluate

```bash
deeponet-pricing evaluate configs/heston_ivol.yaml --model outputs/heston_ivol/heston_ivol_deeponet --plot
deeponet-pricing evaluate configs/heston_ivol.yaml --model outputs/heston_ivol/heston_ivol_deeponet --verify-prices
```

### Python API

```python
from deeponet_pricing.models import HestonIVolSolver
from deeponet_pricing.config import ExperimentConfig
from deeponet_pricing.training import Trainer

# Config-driven
cfg = ExperimentConfig.from_yaml("configs/heston_ivol.yaml")
trainer = Trainer(cfg)
solver = trainer.train()

# Or manual
solver = HestonIVolSolver()
solver.create_data(branch, trunk, targets)
solver.create_model(branch_layers=[5, 256, 256, 256, 256], trunk_layers=[3, 256, 256, 256, 256])
solver.train(iterations=50000, learning_rate=5e-4)
solver.save("my_model")
```

## Models

| Model | Branch Input | Trunk Input | Output |
|-------|-------------|-------------|--------|
| **Heston IV** | (r, κ, θ, σ_v, ρ) | (x, v, τ) | σ_IV |
| **rBergomi** | (ξ₀(t₁)…ξ₀(tₘ), η, ρ, a) | (K, T) | V |

## Structure

```
src/deeponet_pricing/
├── config.py          # YAML-driven dataclass configs
├── cli.py             # CLI entry point
├── models/            # BaseSolver, HestonIVolSolver, RBergomiSolver
├── data/              # Data generation (COS, MC, Sobolev)
├── training/          # Trainer + MLflow callbacks
├── evaluation/        # Bespoke eval per model (bps metrics, plots)
└── utils/             # Heston pricing, rBergomi simulator/curves
```
