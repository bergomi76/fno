"""Tests for model solvers."""

import numpy as np
import pytest


def test_heston_solver_create_data():
    from deeponet_pricing.models.heston import HestonIVolSolver

    solver = HestonIVolSolver()
    branch = np.random.rand(10, 5).astype(np.float32)
    trunk = np.random.rand(20, 3).astype(np.float32)
    targets = np.random.rand(10, 20).astype(np.float32)

    data = solver.create_data(branch, trunk, targets, val_split=0.2)
    assert solver.branch_min is not None
    assert solver.y_scale > 0
    assert data is not None


def test_heston_solver_create_model():
    from deeponet_pricing.models.heston import HestonIVolSolver

    solver = HestonIVolSolver()
    branch = np.random.rand(10, 5).astype(np.float32)
    trunk = np.random.rand(20, 3).astype(np.float32)
    targets = np.random.rand(10, 20).astype(np.float32)

    solver.create_data(branch, trunk, targets)
    model = solver.create_model(branch_layers=[5, 32, 32], trunk_layers=[3, 32, 32])
    assert model is not None
    assert solver.net is not None


def test_heston_predict_batch():
    from deeponet_pricing.models.heston import HestonIVolSolver

    solver = HestonIVolSolver()
    branch = np.random.rand(10, 5).astype(np.float32)
    trunk = np.random.rand(20, 3).astype(np.float32)
    targets = np.random.rand(10, 20).astype(np.float32)

    solver.create_data(branch, trunk, targets)
    solver.create_model(branch_layers=[5, 32, 32], trunk_layers=[3, 32, 32])
    pred = solver.predict_batch(branch, trunk)
    assert pred.shape == (10, 20)


def test_rbergomi_solver_create():
    from deeponet_pricing.models.rbergomi import RBergomiSolver

    solver = RBergomiSolver()
    assert solver.branch_input_dim == 12  # 9 sensors + 3 params
    assert solver.trunk_input_dim == 2


def test_rbergomi_custom_sensors():
    from deeponet_pricing.models.rbergomi import RBergomiSolver

    sp = np.array([0.0, 0.5, 1.0])
    solver = RBergomiSolver(sensor_points=sp)
    assert solver.branch_input_dim == 6  # 3 sensors + 3 params
    assert solver.n_sensors == 3


def test_save_load_roundtrip(tmp_path):
    from deeponet_pricing.models.heston import HestonIVolSolver

    solver = HestonIVolSolver()
    branch = np.random.rand(10, 5).astype(np.float32)
    trunk = np.random.rand(20, 3).astype(np.float32)
    targets = np.random.rand(10, 20).astype(np.float32)

    solver.create_data(branch, trunk, targets)
    solver.create_model(branch_layers=[5, 32, 32], trunk_layers=[3, 32, 32])
    pred_before = solver.predict_batch(branch, trunk)

    path = str(tmp_path / "test_model")
    solver.save(path)

    solver2 = HestonIVolSolver()
    solver2.load(path, branch_layers=[5, 32, 32], trunk_layers=[3, 32, 32])
    pred_after = solver2.predict_batch(branch, trunk)

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)
