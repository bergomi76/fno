"""Heston COS-method pricer and Black-Scholes call price."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as scipy_norm


def _chi_coefficients(c: float, d: float, k: np.ndarray, a: float, b: float) -> np.ndarray:
    result = np.zeros(len(k))
    result[0] = np.exp(d) - np.exp(c)
    kn = k[1:]
    w = kn * np.pi / (b - a)
    w2 = w * w
    cos_d, cos_c = np.cos(w * (d - a)), np.cos(w * (c - a))
    sin_d, sin_c = np.sin(w * (d - a)), np.sin(w * (c - a))
    result[1:] = (1 / (1 + w2)) * (
        cos_d * np.exp(d) - cos_c * np.exp(c)
        + w * sin_d * np.exp(d) - w * sin_c * np.exp(c)
    )
    return result


def _psi_coefficients(c: float, d: float, k: np.ndarray, a: float, b: float) -> np.ndarray:
    result = np.zeros(len(k))
    result[0] = d - c
    kn = k[1:]
    w = kn * np.pi / (b - a)
    result[1:] = (np.sin(w * (d - a)) - np.sin(w * (c - a))) / w
    return result


def _heston_cf(
    u: np.ndarray, tau: float, r: float, v0: float,
    kappa: float, theta: float, sigma_v: float, rho: float,
) -> np.ndarray:
    u = np.asarray(u, dtype=np.complex128)
    d = np.sqrt((rho * sigma_v * 1j * u - kappa) ** 2 + sigma_v**2 * (1j * u + u**2))
    g = (kappa - rho * sigma_v * 1j * u - d) / (kappa - rho * sigma_v * 1j * u + d)
    C = r * 1j * u * tau + (kappa * theta / sigma_v**2) * (
        (kappa - rho * sigma_v * 1j * u - d) * tau
        - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g))
    )
    D = ((kappa - rho * sigma_v * 1j * u - d) / sigma_v**2) * (
        (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    )
    return np.exp(C + D * v0)


def cos_price_heston(
    S: np.ndarray, v: np.ndarray, tau: np.ndarray,
    r: float, kappa: float, theta: float, sigma_v: float, rho: float,
    K: float = 1.0, N: int = 256, L: float = 12.0,
) -> np.ndarray:
    """Price European calls under Heston via COS method (Fang & Oosterlee 2008)."""
    orig = np.asarray(S).shape
    S, v, tau = np.asarray(S).ravel(), np.asarray(v).ravel(), np.asarray(tau).ravel()
    prices = np.zeros(len(S))

    for t in np.unique(tau):
        if t < 1e-10:
            prices[tau == t] = np.maximum(S[tau == t] - K, 0)
            continue
        mask = tau == t
        idx = np.where(mask)[0]
        St, vt = S[mask], v[mask]
        x = np.log(St / K)

        c1 = r * t + (1 - np.exp(-kappa * t)) * (theta - vt.mean()) / (2 * kappa) + theta * t / 2
        c2 = (
            sigma_v * t * kappa * np.exp(-kappa * t) * (vt.mean() - theta) * (8 * kappa * rho - 4 * sigma_v)
            + kappa * rho * sigma_v * (1 - np.exp(-kappa * t)) * (16 * theta - 8 * vt.mean())
            + 2 * theta * kappa * t * (-4 * kappa * rho * sigma_v + sigma_v**2 + 4 * kappa**2)
            + sigma_v**2 * ((theta - 2 * vt.mean()) * np.exp(-2 * kappa * t) + theta * (6 * np.exp(-kappa * t) - 7) + 2 * vt.mean())
            + 8 * kappa**2 * (vt.mean() - theta) * (1 - np.exp(-kappa * t))
        ) / (8 * kappa**3)
        c2 = max(c2, 1e-8)

        a_s, b_s = c1 - L * np.sqrt(c2), c1 + L * np.sqrt(c2)
        k = np.arange(N)
        omega = k * np.pi / (b_s - a_s)
        chi_k = _chi_coefficients(0, b_s, k, a_s, b_s)
        psi_k = _psi_coefficients(0, b_s, k, a_s, b_s)
        V_k = 2 / (b_s - a_s) * (chi_k - K * psi_k)

        for j, (xj, vj) in enumerate(zip(x, vt)):
            cf = _heston_cf(omega, t, r, vj, kappa, theta, sigma_v, rho)
            exp_t = np.exp(1j * k * np.pi * (xj - a_s) / (b_s - a_s))
            s = np.real(cf * exp_t) * V_k
            s[0] *= 0.5
            prices[idx[j]] = max(np.exp(-r * t) * K * s.sum(), 0)

    return prices.reshape(orig)


def bs_call_price(
    S: np.ndarray, K: np.ndarray | float, tau: np.ndarray | float,
    r: float, sigma: np.ndarray,
) -> np.ndarray:
    """Vectorised Black-Scholes European call price."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    tau = np.clip(np.asarray(tau, dtype=np.float64), 1e-8, None)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-8, None)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return S * scipy_norm.cdf(d1) - K * np.exp(-r * tau) * scipy_norm.cdf(d2)
