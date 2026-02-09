"""Abstract base class for DeepONet solvers with min-max normalisation."""

from __future__ import annotations

import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import deepxde as dde
import numpy as np
import torch


@dataclass
class FeatureSpec:
    """Description of a single branch or trunk input feature."""

    name: str
    min_val: float
    max_val: float
    description: str


class BaseSolver(ABC):
    """Base DeepONet solver with encapsulated normalisation.

    Subclasses implement ``branch_features()`` and ``trunk_features()``
    to declare their input dimensions and metadata.
    """

    @abstractmethod
    def branch_features(self) -> list[FeatureSpec]: ...

    @abstractmethod
    def trunk_features(self) -> list[FeatureSpec]: ...

    @property
    def branch_input_dim(self) -> int:
        return len(self.branch_features())

    @property
    def trunk_input_dim(self) -> int:
        return len(self.trunk_features())

    @property
    def branch_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.branch_features())

    @property
    def trunk_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.trunk_features())

    def __init__(self) -> None:
        self.branch_min: np.ndarray | None = None
        self.branch_max: np.ndarray | None = None
        self.trunk_min: np.ndarray | None = None
        self.trunk_max: np.ndarray | None = None
        self.y_scale: float = 1.0

        self.data: dde.data.TripleCartesianProd | None = None
        self.net: dde.nn.DeepONetCartesianProd | None = None
        self.model: dde.Model | None = None

    # -- normalisation helpers -------------------------------------------

    def _norm_branch(self, x: np.ndarray) -> np.ndarray:
        return (x - self.branch_min) / (self.branch_max - self.branch_min + 1e-8)

    def _norm_trunk(self, x: np.ndarray) -> np.ndarray:
        return (x - self.trunk_min) / (self.trunk_max - self.trunk_min + 1e-8)

    def _device(self) -> torch.device:
        if self.net is None:
            return torch.device("cpu")
        return next(self.net.parameters()).device

    # -- data / model / train --------------------------------------------

    def create_data(
        self,
        branch: np.ndarray,
        trunk: np.ndarray,
        targets: np.ndarray,
        val_split: float = 0.1,
    ) -> dde.data.TripleCartesianProd:
        """Build DeepXDE ``TripleCartesianProd`` with auto-normalisation."""
        assert branch.shape[1] == self.branch_input_dim
        assert trunk.shape[1] == self.trunk_input_dim
        assert targets.shape == (branch.shape[0], trunk.shape[0])

        self.branch_min = branch.min(axis=0)
        self.branch_max = branch.max(axis=0)
        self.trunk_min = trunk.min(axis=0)
        self.trunk_max = trunk.max(axis=0)
        self.y_scale = float(np.max(np.abs(targets))) + 1e-8

        b = self._norm_branch(branch).astype(np.float32)
        t = self._norm_trunk(trunk).astype(np.float32)
        y = (targets / self.y_scale).astype(np.float32)

        n = branch.shape[0]
        n_val = max(1, int(n * val_split))
        idx = np.random.permutation(n)
        tr, va = idx[: n - n_val], idx[n - n_val :]

        self.data = dde.data.TripleCartesianProd(
            X_train=(b[tr], t), y_train=y[tr],
            X_test=(b[va], t), y_test=y[va],
        )
        return self.data

    def create_model(
        self,
        branch_layers: list[int],
        trunk_layers: list[int],
        activation: str = "tanh",
        initializer: str = "Glorot uniform",
    ) -> dde.Model:
        if self.data is None:
            raise ValueError("Call create_data() first")

        assert branch_layers[0] == self.branch_input_dim
        assert trunk_layers[0] == self.trunk_input_dim

        self.net = dde.nn.DeepONetCartesianProd(
            branch_layers, trunk_layers, activation, initializer,
        )
        self.model = dde.Model(self.data, self.net)
        return self.model

    def train(
        self,
        iterations: int = 20_000,
        learning_rate: float = 1e-3,
        display_every: int = 1000,
        use_lbfgs: bool = True,
        callbacks: list[Any] | None = None,
    ) -> tuple[Any, Any]:
        if self.model is None:
            raise ValueError("Call create_model() first")

        self.model.compile("adam", lr=learning_rate)
        lh, ts = self.model.train(
            iterations=iterations,
            display_every=display_every,
            callbacks=callbacks,
        )
        if use_lbfgs:
            self.model.compile("L-BFGS")
            lh, ts = self.model.train(display_every=display_every, callbacks=callbacks)
        return lh, ts

    # -- prediction (Cartesian-product) ----------------------------------

    def predict_batch(
        self, branch: np.ndarray, trunk: np.ndarray
    ) -> np.ndarray:
        """Predict for multiple branch inputs × full trunk grid."""
        if self.net is None:
            raise ValueError("Model must be trained before prediction")

        b = torch.tensor(self._norm_branch(branch), dtype=torch.float32, device=self._device())
        t = torch.tensor(self._norm_trunk(trunk), dtype=torch.float32, device=self._device())

        self.net.eval()
        with torch.no_grad():
            pred = self.net((b, t)).cpu().numpy()
        return pred.reshape(branch.shape[0], trunk.shape[0]) * self.y_scale

    # -- save / load -----------------------------------------------------

    def save(self, filepath: str) -> None:
        if self.net is None:
            raise ValueError("No model to save")
        torch.save(self.net.state_dict(), f"{filepath}.pth")
        np.savez(
            f"{filepath}_norm_params.npz",
            branch_min=self.branch_min, branch_max=self.branch_max,
            trunk_min=self.trunk_min, trunk_max=self.trunk_max,
            y_scale=self.y_scale,
        )

    def load(
        self,
        filepath: str,
        branch_layers: list[int],
        trunk_layers: list[int],
        activation: str = "tanh",
        initializer: str = "Glorot uniform",
    ) -> None:
        base = filepath.replace(".pth", "").replace(".pt", "")
        norm = np.load(f"{base}_norm_params.npz")
        self.branch_min = norm["branch_min"]
        self.branch_max = norm["branch_max"]
        self.trunk_min = norm["trunk_min"]
        self.trunk_max = norm["trunk_max"]
        self.y_scale = float(norm["y_scale"])

        self.net = dde.nn.DeepONetCartesianProd(
            branch_layers, trunk_layers, activation, initializer,
        )

        candidates = [f"{base}.pth", filepath] + sorted(glob.glob(f"{base}-*.pt"))
        model_path = next((c for c in candidates if Path(c).exists()), None)
        if model_path is None:
            raise FileNotFoundError(f"No weights found for {filepath}")

        state = torch.load(model_path, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.net.load_state_dict(state)
        self.net.eval()


