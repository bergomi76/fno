"""Heston implied-vol data generation via COS method."""

from __future__ import annotations

import logging

import numpy as np
from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess as jaeckel_iv
from tqdm import tqdm

from deeponet_pricing.utils.heston_pricing import cos_price_heston

logger = logging.getLogger(__name__)


def _bs_implied_vol_vec(
    prices: np.ndarray, S: np.ndarray, K: float, tau: np.ndarray, r: float,
) -> np.ndarray:
    """BS IV inversion using Jäckel's 'Let's Be Rational' algorithm.

    Machine-precision accurate and handles deep ITM/OTM & near-zero expiry.
    """
    from py_lets_be_rational.exceptions import BelowIntrinsicException
    
    prices = np.asarray(prices, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    # Convert from spot/BS world to forward/Black world:
    #   F = S * exp(r * tau),  undiscounted_price = price * exp(r * tau)
    df = np.exp(r * tau)
    F = S * df
    undiscounted = prices * df
    q = 1.0  # call

    iv = []
    for p, f, t in zip(undiscounted, F, tau):
        if t < 1e-12 or p < 1e-12:
            iv.append(0.0)
        else:
            try:
                iv.append(jaeckel_iv(p, f, K, t, q))
            except BelowIntrinsicException:
                # Price below intrinsic: use small positive vol
                iv.append(1e-4)
    
    return np.array(iv, dtype=np.float32)


def generate_heston_ivol_data(
    n_samples: int = 512,
    n_x: int = 24,
    n_v: int = 24,
    n_tau: int = 24,
    r_range: tuple[float, float] = (0.01, 0.08),
    kappa_range: tuple[float, float] = (0.5, 5.0),
    theta_range: tuple[float, float] = (0.02, 0.10),
    sigma_v_range: tuple[float, float] = (0.2, 0.6),
    rho_range: tuple[float, float] = (-0.9, -0.1),
    x_range: tuple[float, float] = (-0.4, 0.4),
    v_range: tuple[float, float] = (0.02, 0.20),
    tau_range: tuple[float, float] = (0.05, 1.0),
    log_tau: bool = True,
    adaptive_cos_n: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Heston IV data via COS pricing + BS inversion.

    Uses Latin Hypercube Sampling for better parameter space coverage.

    Args:
        log_tau: If True, use log-spaced τ grid (denser at short maturities)
        adaptive_cos_n: If True, increase COS N for short maturities

    Returns:
        (branch, trunk, targets) where:
            branch: (n_samples, 5) — (r, κ, θ, σ_v, ρ)
            trunk:  (n_grid, 3)    — (x, v, τ)
            targets: (n_samples, n_grid) — implied vols
    """
    from scipy.stats import qmc

    rng = np.random.default_rng(seed)

    # Latin Hypercube Sampling for branch params (better space-filling than uniform)
    param_names = ["r", "kappa", "theta", "sigma_v", "rho"]
    param_ranges = [r_range, kappa_range, theta_range, sigma_v_range, rho_range]
    
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    lhs_samples = sampler.random(n=n_samples)
    
    lower = np.array([r[0] for r in param_ranges])
    upper = np.array([r[1] for r in param_ranges])
    branch = qmc.scale(lhs_samples, lower, upper).astype(np.float32)

    if verbose:
        logger.info(f"Latin Hypercube Sampling: {n_samples} samples in 5D parameter space")

    # Trunk grid - use log-spacing for τ to concentrate points at short maturities
    x_vals = np.linspace(*x_range, n_x)
    v_vals = np.linspace(*v_range, n_v)
    
    if log_tau:
        # Log-spaced: 10x more density at short maturities where COS struggles
        tau_vals = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), n_tau)
        if verbose:
            logger.info(f"Using log-spaced τ grid: {tau_vals[0]:.3f} to {tau_vals[-1]:.3f}")
    else:
        tau_vals = np.linspace(*tau_range, n_tau)
    
    xg, vg, tg = np.meshgrid(x_vals, v_vals, tau_vals, indexing="ij")
    trunk = np.column_stack([xg.ravel(), vg.ravel(), tg.ravel()]).astype(np.float32)
    n_grid = trunk.shape[0]

    if verbose:
        logger.info(f"Grid: {n_x}×{n_v}×{n_tau} = {n_grid:,} trunk points")
        logger.info(f"Total evaluations: {n_samples * n_grid:,}")

    S = np.exp(trunk[:, 0])
    v = trunk[:, 1]
    tau = trunk[:, 2]
    K = 1.0

    targets = np.zeros((n_samples, n_grid), dtype=np.float32)
    valid_count = 0
    feller_warnings = 0

    for i in tqdm(range(n_samples), desc="COS pricing", disable=not verbose):
        r_i, kappa_i, theta_i, sv_i, rho_i = branch[i]
        
        # Feller condition: 2κθ ≥ σ_v² (sufficient but not necessary for well-posedness)
        feller_lhs = 2 * kappa_i * theta_i
        feller_rhs = sv_i ** 2
        if feller_lhs < feller_rhs:
            feller_warnings += 1
        
        # Adaptive COS N: use higher N for short maturities to avoid Gibbs oscillations
        # For τ < 0.1, use N=512; otherwise N=128
        if adaptive_cos_n:
            N_cos = 512 if (tau < 0.1).any() else 128
        else:
            N_cos = 128
        
        prices = cos_price_heston(S, v, tau, r_i, kappa_i, theta_i, sv_i, rho_i, K=K, N=N_cos)
        ivols = _bs_implied_vol_vec(prices, S, K, tau, r_i)
        
        # Check for excessive NaNs (> 10% bad is suspicious)
        nan_ratio = np.isnan(ivols).mean()
        if nan_ratio > 0.1:
            if verbose:
                logger.warning(f"Sample {i}: {nan_ratio:.1%} NaNs, skipping")
            continue
        
        targets[i] = ivols
        valid_count += 1

    # Drop samples with any remaining NaN IVs
    valid = ~np.any(np.isnan(targets), axis=1)
    branch, targets = branch[valid], targets[valid]
    
    if verbose:
        logger.info(f"Generation complete:")
        logger.info(f"  Valid samples: {valid.sum()}/{n_samples}")
        if feller_warnings > 0:
            logger.info(f"  Feller condition warnings: {feller_warnings} (kept anyway)")
        logger.info(f"  Final shapes: branch={branch.shape}, trunk={trunk.shape}, targets={targets.shape}")
        logger.info(f"  IV range: [{targets.min():.4f}, {targets.max():.4f}], mean={targets.mean():.4f}")

    return branch, trunk, targets
