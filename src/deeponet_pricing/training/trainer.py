"""DeepONet trainer with MLflow experiment tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import deepxde as dde
import numpy as np

from deeponet_pricing.config import ExperimentConfig
from deeponet_pricing.models.base import BaseSolver
from deeponet_pricing.models.heston import HestonIVolSolver
from deeponet_pricing.models.rbergomi import RBergomiSolver

logger = logging.getLogger(__name__)


class MLFlowCallback(dde.callbacks.Callback):
    """Log train/test loss to MLflow at regular intervals."""

    def __init__(self, log_every: int = 100) -> None:
        super().__init__()
        self.log_every = log_every

    def on_epoch_end(self) -> None:
        step = self.model.train_state.step
        if step % self.log_every != 0:
            return
        try:
            import mlflow
            loss_train = self.model.train_state.loss_train
            loss_test = self.model.train_state.loss_test
            if loss_train is not None:
                for i, l in enumerate(np.atleast_1d(loss_train)):
                    mlflow.log_metric(f"loss_train_{i}", float(l), step=step)
            if loss_test is not None:
                for i, l in enumerate(np.atleast_1d(loss_test)):
                    mlflow.log_metric(f"loss_test_{i}", float(l), step=step)
        except Exception:
            pass


class Trainer:
    """Config-driven DeepONet trainer with MLflow."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.cfg = config
        self.solver: BaseSolver | None = None
        self.branch: np.ndarray | None = None
        self.trunk: np.ndarray | None = None
        self.targets: np.ndarray | None = None

    # -- data loading / generation ---------------------------------------

    def load_data(self) -> None:
        src = self.cfg.data.source
        if src.endswith((".pt", ".pth")):
            self._load_torch(src)
        elif src.endswith(".npz"):
            self._load_npz(src)
        else:
            self._generate_data()

    def _load_torch(self, path: str) -> None:
        import torch
        logger.info(f"Loading data from {path}")
        data = torch.load(path, weights_only=False)
        self.branch = data["branch"].numpy() if hasattr(data["branch"], "numpy") else data["branch"]
        self.trunk = data["trunk"].numpy() if hasattr(data["trunk"], "numpy") else data["trunk"]
        self.targets = data["targets"].numpy() if hasattr(data["targets"], "numpy") else data["targets"]
        logger.info(f"Loaded: branch={self.branch.shape}, trunk={self.trunk.shape}, targets={self.targets.shape}")

    def _load_npz(self, path: str) -> None:
        logger.info(f"Loading data from {path}")
        data = np.load(path)
        self.branch = data["branch"]
        self.trunk = data["trunk"]
        self.targets = data["y"] if "y" in data else data["targets"]
        logger.info(f"Loaded: branch={self.branch.shape}, trunk={self.trunk.shape}, targets={self.targets.shape}")

    def _generate_data(self) -> None:
        logger.info(f"Generating {self.cfg.model} data...")
        params = self.cfg.data.params or {}
        if self.cfg.model == "heston":
            from deeponet_pricing.data.heston import generate_heston_ivol_data
            self.branch, self.trunk, self.targets = generate_heston_ivol_data(**params)
        elif self.cfg.model == "rbergomi":
            from deeponet_pricing.data.rbergomi import generate_rbergomi_curve_data
            self.branch, self.trunk, self.targets = generate_rbergomi_curve_data(**params)
        else:
            raise ValueError(f"Unknown model: {self.cfg.model}")

    # -- solver creation -------------------------------------------------

    def _create_solver(self) -> BaseSolver:
        if self.cfg.model == "heston":
            return HestonIVolSolver()
        elif self.cfg.model == "rbergomi":
            sp = self.cfg.data.params.get("sensor_points") if self.cfg.data.params else None
            return RBergomiSolver(sensor_points=np.array(sp) if sp else None)
        raise ValueError(f"Unknown model: {self.cfg.model}")

    # -- train -----------------------------------------------------------

    def train(self) -> BaseSolver:
        if self.branch is None:
            self.load_data()

        self.solver = self._create_solver()
        self.solver.create_data(
            self.branch, self.trunk, self.targets,
            val_split=self.cfg.training.val_split,
        )
        self.solver.create_model(
            branch_layers=self.cfg.network.branch_layers,
            trunk_layers=self.cfg.network.trunk_layers,
            activation=self.cfg.network.activation,
            initializer=self.cfg.network.initializer,
        )

        n_params = sum(p.numel() for p in self.solver.net.parameters())
        logger.info(f"Model parameters: {n_params:,}")

        callbacks = self._build_callbacks()

        import mlflow
        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow_experiment_name)
        logger.info(f"MLflow: {self.cfg.mlflow_experiment_name} @ {self.cfg.mlflow_tracking_uri}")
        
        with mlflow.start_run(run_name=self.cfg.name):
            mlflow.log_params(self.cfg.to_flat_dict())
            mlflow.log_param("n_params", n_params)

            # Save config YAML as artifact
            out = Path(self.cfg.output.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            config_path = out / "config.yaml"
            self.cfg.to_yaml(config_path)
            mlflow.log_artifact(str(config_path))
            logger.info(f"Logged config artifact: {config_path}")

            self.solver.train(
                iterations=self.cfg.training.iterations,
                learning_rate=self.cfg.training.lr,
                display_every=self.cfg.training.display_every,
                use_lbfgs=self.cfg.training.use_lbfgs,
                callbacks=callbacks,
            )

            # save with version number
            version_suffix = f"_v{self.cfg.version.replace('.', '_')}"
            model_name_with_version = self.cfg.output.model_name + version_suffix
            save_path = str(out / model_name_with_version)
            self.solver.save(save_path)
            logger.info(f"Model saved → {save_path}")

            # quick eval
            pred = self.solver.predict_batch(self.branch, self.trunk)
            mae = float(np.mean(np.abs(pred - self.targets)))
            mlflow.log_metric("mae", mae)
            logger.info(f"Train MAE: {mae:.6f}")
            
            # Log any evaluation plots if they exist
            eval_plot_patterns = [
                "evaluation_plots.png",
                "iv_surface.png",
                "rbergomi_eval.png",
            ]
            for pattern in eval_plot_patterns:
                plot_path = out / pattern
                if plot_path.exists():
                    mlflow.log_artifact(str(plot_path))
                    logger.info(f"Logged plot artifact: {plot_path}")

        return self.solver

    def _build_callbacks(self) -> list[Any]:
        cbs: list[Any] = []
        cbs.append(MLFlowCallback(log_every=self.cfg.training.display_every))
        return cbs
