"""Tests for config loading and serialisation."""

from pathlib import Path
import tempfile

import pytest

from deeponet_pricing.config import ExperimentConfig, NetworkConfig, TrainingConfig, DataConfig, OutputConfig


def test_config_defaults():
    cfg = ExperimentConfig(
        name="test", model="heston",
        network=NetworkConfig(branch_layers=[5, 128, 128], trunk_layers=[3, 128, 128]),
    )
    assert cfg.network.activation == "tanh"
    assert cfg.training.iterations == 20_000
    assert cfg.training.use_lbfgs is True
    assert str(cfg.output.output_dir) == "outputs"


def test_config_roundtrip_yaml():
    cfg = ExperimentConfig(
        name="roundtrip",
        model="rbergomi",
        network=NetworkConfig(branch_layers=[12, 128, 64], trunk_layers=[2, 64, 64]),
        training=TrainingConfig(iterations=5000, lr=5e-4),
        data=DataConfig(source="data.npz"),
        output=OutputConfig(output_dir="out", model_name="test_model"),
    )
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(cfg.to_yaml())
        f.flush()
        loaded = ExperimentConfig.from_yaml(f.name)

    assert loaded.name == "roundtrip"
    assert loaded.model == "rbergomi"
    assert loaded.network.branch_layers == [12, 128, 64]
    assert loaded.training.lr == 5e-4
    assert loaded.output.model_name == "test_model"


def test_flat_dict():
    cfg = ExperimentConfig(
        name="flat", model="heston",
        network=NetworkConfig(branch_layers=[5, 128], trunk_layers=[3, 128]),
    )
    d = cfg.to_flat_dict()
    assert d["model"] == "heston"
    assert "network.activation" in d
    assert "training.iterations" in d


def test_load_heston_yaml():
    p = Path(__file__).parent.parent / "configs" / "heston_ivol.yaml"
    if p.exists():
        cfg = ExperimentConfig.from_yaml(str(p))
        assert cfg.model == "heston"
        assert cfg.network.branch_layers[0] == 5


def test_load_rbergomi_yaml():
    p = Path(__file__).parent.parent / "configs" / "rbergomi_curve.yaml"
    if p.exists():
        cfg = ExperimentConfig.from_yaml(str(p))
        assert cfg.model == "rbergomi"
        assert cfg.data.params is not None
