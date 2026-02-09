"""Config dataclasses for DeepONet training experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class NetworkConfig:
    """DeepONet architecture."""

    branch_layers: list[int]
    trunk_layers: list[int]
    activation: str = "tanh"
    initializer: str = "Glorot uniform"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    iterations: int = 20_000
    lr: float = 1e-3
    display_every: int = 1000
    use_lbfgs: bool = True
    val_split: float = 0.1
    seed: int = 42


@dataclass
class DataConfig:
    """Data generation / loading parameters.

    For Heston: grid_size, n_samples, parameter ranges, etc.
    For rBergomi: num_curves, num_strikes, num_maturities, MC config, etc.
    Extra fields go into ``params`` and are forwarded to the data generator.
    """

    source: str = ""  # path to pre-generated .pt/.npz, or empty to generate
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Where to write results."""

    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    model_name: str = "model"


@dataclass
class ExperimentConfig:
    """Top-level experiment config loaded from YAML.

    ``model`` selects the solver: ``"heston"`` or ``"rbergomi"``.
    """

    network: NetworkConfig

    version: str = "1.0"
    name: str = "experiment"
    model: str = "heston"  # "heston" | "rbergomi"
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if isinstance(self.output.output_dir, str):
            self.output.output_dir = Path(self.output.output_dir)
        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = f"sqlite:///{self.output.output_dir}/mlflow.db"
        if self.mlflow_experiment_name is None:
            self.mlflow_experiment_name = self.name

    # -- YAML I/O --------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)

        net = NetworkConfig(**raw.get("network", {}))
        train = TrainingConfig(**raw.get("training", {}))

        data_raw = raw.get("data", {})
        data = DataConfig(
            source=data_raw.pop("source", ""),
            params=data_raw.pop("params", data_raw),
        )

        out_raw = raw.get("output", {})
        if "output_dir" in out_raw:
            out_raw["output_dir"] = Path(out_raw["output_dir"])
        out = OutputConfig(**out_raw)

        return cls(
            network=net,
            version=raw.get("version", "1.0"),
            name=raw.get("name", "experiment"),
            model=raw.get("model", "heston"),
            mlflow_tracking_uri=raw.get("mlflow_tracking_uri"),
            mlflow_experiment_name=raw.get("mlflow_experiment_name"),
            training=train,
            data=data,
            output=out,
        )

    def to_yaml(self, path: str | Path | None = None) -> str:
        def _clean(obj: Any) -> Any:
            """Convert Path to str, recurse through dicts."""
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(x) for x in obj]
            return obj

        d = asdict(self)
        d["output"] = _clean(d["output"])
        text = yaml.dump(d, default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        return text

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten for MLflow param logging."""
        d = asdict(self)
        flat: dict[str, Any] = {}

        def _flatten(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _flatten(f"{prefix}.{k}" if prefix else k, v)
            elif isinstance(obj, (list, tuple)):
                flat[prefix] = str(obj)
            elif isinstance(obj, Path):
                flat[prefix] = str(obj)
            else:
                flat[prefix] = obj

        _flatten("", d)
        return flat
