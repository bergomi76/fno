"""Sobolev Triple Cartesian Product data class for DeepONet.

Extends TripleCartesianProd with gradient-matching (Sobolev) loss:
    Loss = w_data·||V_pred − V||² + w_grad·||∂V_pred/∂ξ − bucket_vegas||²
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from deepxde.data.triple import TripleCartesianProd


class SobolevTripleCartesianProd(TripleCartesianProd):
    """TripleCartesianProd with gradient-matching loss for Sobolev training."""

    def __init__(
        self,
        X_train: tuple[np.ndarray, np.ndarray],
        y_train: np.ndarray,
        X_test: tuple[np.ndarray, np.ndarray],
        y_test: np.ndarray,
        grad_train: np.ndarray,
        grad_test: np.ndarray | None = None,
        n_grad_dims: int | None = None,
        data_weight: float = 1.0,
        grad_weight: float = 1.0,
        grad_subsample: int = 20,
    ) -> None:
        super().__init__(X_train, y_train, X_test, y_test)
        self.grad_train = grad_train
        self.n_grad_dims = n_grad_dims or grad_train.shape[-1]
        self.data_weight = data_weight
        self.grad_weight = grad_weight
        self.grad_subsample = grad_subsample

        n_test = X_test[0].shape[0]
        self.grad_test = grad_test if grad_test is not None else grad_train[:n_test]

        n_locs = X_train[1].shape[0]
        self.grad_loc_indices = np.linspace(0, n_locs - 1, min(grad_subsample, n_locs), dtype=int)

        self._grad_train_t: torch.Tensor | None = None
        self._grad_test_t: torch.Tensor | None = None
        self._device: torch.device | None = None

    def _grad_tensor(self, training: bool, device: torch.device) -> torch.Tensor:
        src = self.grad_train if training else self.grad_test
        attr = "_grad_train_t" if training else "_grad_test_t"
        cached = getattr(self, attr)
        if cached is None or self._device != device:
            t = torch.tensor(src, dtype=torch.float32, device=device)
            setattr(self, attr, t)
            self._device = device
            return t
        return cached

    def losses(
        self, targets: Any, outputs: Any, loss_fn: Any,
        inputs: Any, model: Any, aux: Any = None,
    ) -> list[torch.Tensor] | torch.Tensor:
        losses = []
        if self.data_weight > 0:
            losses.append(self.data_weight * loss_fn(targets, outputs))
        if self.grad_weight > 0:
            branch_input, trunk_input = inputs
            is_train = (branch_input.shape[0] == self.grad_train.shape[0])
            losses.append(self.grad_weight * self._grad_loss(inputs, model, is_train))
        return losses if len(losses) > 1 else losses[0]

    def _grad_loss(self, inputs: Any, model: Any, training: bool) -> torch.Tensor:
        branch, trunk = inputs
        device = next(model.net.parameters()).device
        grad_target = self._grad_tensor(training, device)

        if isinstance(branch, np.ndarray):
            branch = torch.tensor(branch, dtype=torch.float32, device=device)
        if not branch.requires_grad:
            branch = branch.clone().detach().requires_grad_(True)
        if isinstance(trunk, np.ndarray):
            trunk = torch.tensor(trunk, dtype=torch.float32, device=device)

        pred = model.net((branch, trunk))
        total = torch.tensor(0.0, device=device)

        for loc in self.grad_loc_indices:
            grads = torch.autograd.grad(
                pred[:, loc], branch,
                grad_outputs=torch.ones(pred.shape[0], device=device),
                retain_graph=True, create_graph=True,
            )[0][:, : self.n_grad_dims]
            total = total + torch.mean((grads - grad_target[:, loc, :]) ** 2)

        return total / len(self.grad_loc_indices)
