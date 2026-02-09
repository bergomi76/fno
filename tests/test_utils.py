"""Tests for utility functions."""

import numpy as np
import pytest


def test_cos_price_heston_atm():
    from deeponet_pricing.utils.heston_pricing import cos_price_heston

    price = cos_price_heston(
        S=np.array([1.0]), v=np.array([0.04]), tau=np.array([1.0]),
        r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, K=1.0,
    )
    assert price.shape == (1,)
    assert 0 < price[0] < 1


def test_bs_call_price():
    from deeponet_pricing.utils.heston_pricing import bs_call_price

    p = bs_call_price(S=np.array([1.0]), K=1.0, tau=1.0, r=0.05, sigma=np.array([0.2]))
    assert 0 < p[0] < 1


def test_xi_curve_generation():
    from deeponet_pricing.utils.rbergomi.curves import generate_xi_curve, DEFAULT_SENSOR_POINTS

    curve = generate_xi_curve({"level": 0.04, "slope": 0.01})
    assert len(curve) == len(DEFAULT_SENSOR_POINTS)
    assert np.all(curve > 0)


def test_xi_curve_sampling():
    from deeponet_pricing.utils.rbergomi.curves import sample_xi_curves

    curves = sample_xi_curves(10, rng=np.random.default_rng(42))
    assert curves.shape[0] == 10
    assert np.all(curves > 0)


def test_rbergomi_simulator():
    from deeponet_pricing.utils.rbergomi.simulator import RoughBergomi

    rb = RoughBergomi(n=50, N=100, T=0.5, a=-0.43)
    S, V = rb.simulate(xi=0.04, eta=1.9, rho=-0.7, seed=42)
    assert S.shape == (100, 26)
    assert np.all(S > 0)


def test_rbergomi_pricing():
    from deeponet_pricing.utils.rbergomi.simulator import RoughBergomi

    rb = RoughBergomi(n=50, N=1000, T=1.0, a=-0.43)
    prices = rb.price_european(xi=0.04, eta=1.9, rho=-0.7, K=np.array([0.9, 1.0, 1.1]), seed=42)
    assert len(prices) == 3
    assert prices[0] > prices[1] > prices[2]  # OTM put < ATM < ITM call
