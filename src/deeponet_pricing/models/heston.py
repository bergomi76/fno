"""Heston implied-volatility DeepONet solver.

Learns G: (r, κ, θ, σ_v, ρ) → σ_IV(x, v, τ)
where x = log(S/K), v = variance, τ = time-to-maturity.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import BaseSolver, FeatureSpec


class HestonIVolSolver(BaseSolver):
    """DeepONet for Heston implied-volatility surfaces."""

    def branch_features(self) -> list[FeatureSpec]:
        return [
            FeatureSpec("r", 0.0, 0.1, "Risk-free rate"),
            FeatureSpec("kappa", 0.5, 5.0, "Mean-reversion speed"),
            FeatureSpec("theta", 0.01, 0.1, "Long-run variance"),
            FeatureSpec("sigma_v", 0.1, 1.0, "Vol-of-vol"),
            FeatureSpec("rho", -0.95, -0.3, "Spot-vol correlation"),
        ]

    def trunk_features(self) -> list[FeatureSpec]:
        return [
            FeatureSpec("x", -0.5, 0.5, "Log-moneyness"),
            FeatureSpec("v", 0.01, 0.1, "Instantaneous variance"),
            FeatureSpec("tau", 0.04, 2.0, "Time to maturity"),
        ]

    # -- convenience predict for a single param set ----------------------

    def predict(
        self,
        r: float, kappa: float, theta: float, sigma_v: float, rho: float,
        x: np.ndarray, v: np.ndarray, tau: np.ndarray,
    ) -> np.ndarray:
        """Predict IV for one Heston param set at given (x, v, τ) points."""
        if self.net is None:
            raise ValueError("Model must be trained before prediction")

        branch = np.array([[r, kappa, theta, sigma_v, rho]], dtype=np.float32)
        trunk = np.column_stack([x, v, tau]).astype(np.float32)

        b = torch.tensor(self._norm_branch(branch), dtype=torch.float32, device=self._device())
        t = torch.tensor(self._norm_trunk(trunk), dtype=torch.float32, device=self._device())

        self.net.eval()
        with torch.no_grad():
            pred = self.net((b, t)).cpu().numpy()
        return pred.flatten() * self.y_scale
