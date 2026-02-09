"""Heston IV DeepONet evaluation with bps metrics, core mask, and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deeponet_pricing.models.heston import HestonIVolSolver
from deeponet_pricing.utils.heston_pricing import bs_call_price, cos_price_heston


# -- metrics ----------------------------------------------------------------


def compute_iv_metrics(
    pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None,
) -> dict:
    """IV error metrics in basis points."""
    p = pred[mask].ravel() if mask is not None else pred.ravel()
    t = true[mask].ravel() if mask is not None else true.ravel()
    err = (p - t) * 10_000
    ae = np.abs(err)
    safe = np.maximum(t, 0.01)
    rel = np.abs(p - t) / safe * 100
    return {
        "iv_mae_bps": float(ae.mean()),
        "iv_rmse_bps": float(np.sqrt((err**2).mean())),
        "iv_bias_bps": float(err.mean()),
        "iv_std_bps": float(err.std()),
        "iv_p95_bps": float(np.percentile(ae, 95)),
        "iv_p99_bps": float(np.percentile(ae, 99)),
        "iv_max_bps": float(ae.max()),
        "iv_errors_bps": err,
        "rel_mae_pct": float(rel.mean()),
        "rel_p95_pct": float(np.percentile(rel, 95)),
        "rel_p99_pct": float(np.percentile(rel, 99)),
        "rel_max_pct": float(rel.max()),
    }


def compute_trimmed_metrics(iv_errors_bps: np.ndarray, pct: float = 99) -> dict:
    ae = np.abs(iv_errors_bps)
    thr = float(np.percentile(ae, pct))
    m = ae <= thr
    te = iv_errors_bps[m]
    return {
        "trimmed_mae_bps": float(np.abs(te).mean()),
        "trimmed_rmse_bps": float(np.sqrt((te**2).mean())),
        "trimmed_max_bps": float(np.abs(te).max()),
        "n_removed": int((~m).sum()),
        "threshold_bps": thr,
    }


def create_core_mask(
    trunk: np.ndarray, x_trim: float = 0.15, v_trim: float = 0.10, tau_trim: float = 0.15,
) -> np.ndarray:
    """Boolean mask excluding trunk boundary regions."""
    trunk = np.asarray(trunk)  # Ensure it's a numpy array
    x, v, tau = trunk[:, 0], trunk[:, 1], trunk[:, 2]
    xr, vr, tr = x.max() - x.min(), v.max() - v.min(), tau.max() - tau.min()
    return (
        (x >= x.min() + x_trim * xr) & (x <= x.max() - x_trim * xr)
        & (v >= v.min() + v_trim * vr)
        & (tau >= tau.min() + tau_trim * tr)
    )


def verify_prices(
    branch: np.ndarray, trunk: np.ndarray, pred_iv: np.ndarray, n_samples: int = 10,
) -> dict:
    """COS reprice verification: BS(pred_IV) vs Heston COS."""
    idx = np.random.choice(branch.shape[0], min(n_samples, branch.shape[0]), replace=False)
    S, K = np.exp(trunk[:, 0]), 1.0
    v, tau = trunk[:, 1], trunk[:, 2]
    pe, pre = [], []
    for i in idx:
        r, kappa, theta, sv, rho = branch[i]
        hp = cos_price_heston(S, v, tau, r, kappa, theta, sv, rho, K=K)
        bp = bs_call_price(S, K, tau, r, pred_iv[i])
        pe.append(np.abs(bp - hp))
        pre.append(np.abs(bp - hp) / np.maximum(hp, 1e-6))
    pe, pre = np.concatenate(pe), np.concatenate(pre)
    return {
        "price_mae": float(pe.mean()),
        "price_rmse": float(np.sqrt((pe**2).mean())),
        "price_mape": float(pre.mean() * 100),
        "price_p95_rel": float(np.percentile(pre, 95) * 100),
    }


# -- full evaluation -------------------------------------------------------


def evaluate_heston(
    solver: HestonIVolSolver,
    branch: np.ndarray,
    trunk: np.ndarray,
    targets: np.ndarray,
    do_verify: bool = False,
    output_dir: str | None = None,
    do_plot: bool = False,
) -> dict:
    """Full Heston IV evaluation with printed report."""
    pred = solver.predict_batch(branch, trunk)

    m = compute_iv_metrics(pred, targets)
    errs = m.pop("iv_errors_bps")

    core_mask = create_core_mask(trunk)
    cm = compute_iv_metrics(pred[:, core_mask], targets[:, core_mask])
    cm.pop("iv_errors_bps")

    t99 = compute_trimmed_metrics(errs, 99)
    t95 = compute_trimmed_metrics(errs, 95)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    _table("FULL (bps)", m)
    print(f"\n  CORE ({core_mask.sum()}/{len(core_mask)} trunk pts retained)")
    _table("CORE (bps)", cm)
    print(f"\n  TRIMMED p99 (>{t99['threshold_bps']:.1f} bps removed)")
    print(f"    MAE:  {t99['trimmed_mae_bps']:.2f}  RMSE: {t99['trimmed_rmse_bps']:.2f}")
    print(f"  TRIMMED p95 (>{t95['threshold_bps']:.1f} bps removed)")
    print(f"    MAE:  {t95['trimmed_mae_bps']:.2f}  RMSE: {t95['trimmed_rmse_bps']:.2f}")

    mae = m["iv_mae_bps"]
    verdict = (
        "Excellent (<10 bps)" if mae < 10
        else "Good (<50 bps)" if mae < 50
        else "Acceptable (<100 bps)" if mae < 100
        else "Needs improvement"
    )
    print(f"\n  Verdict: {verdict}")
    print("=" * 60)

    result = {**m, "core": cm, "trimmed_99": t99, "pred": pred, "iv_errors_bps": errs}

    if do_verify:
        pm = verify_prices(branch, trunk, pred)
        print(f"\n  Price MAE: {pm['price_mae']:.6f}  MAPE: {pm['price_mape']:.4f}%")
        result["prices"] = pm

    if do_plot and output_dir:
        plot_heston_results(branch, trunk, pred, targets, errs, Path(output_dir))

    return result


def _table(label: str, m: dict) -> None:
    print(f"\n  {label}:")
    for k in ("iv_mae_bps", "iv_rmse_bps", "iv_bias_bps", "iv_p95_bps", "iv_p99_bps", "iv_max_bps"):
        if k in m:
            print(f"    {k.replace('iv_', '').replace('_bps', ''):>8s}: {m[k]:8.2f}")


# -- plotting ---------------------------------------------------------------


def plot_heston_results(
    branch: np.ndarray, trunk: np.ndarray,
    pred: np.ndarray, true: np.ndarray,
    iv_errors_bps: np.ndarray, output_dir: Path,
) -> None:
    """6-panel evaluation + IV surface comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    x_u = np.unique(trunk[:, 0])
    v_u = np.unique(trunk[:, 1])
    tau_u = np.unique(trunk[:, 2])
    nx, nv, ntau = len(x_u), len(v_u), len(tau_u)

    si = 0
    r = branch[si, 0]
    pg = pred[si].reshape(nx, nv, ntau)
    tg = true[si].reshape(nx, nv, ntau)
    vm = nv // 2
    strikes = np.exp(x_u)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ti_list = [0, ntau // 4, ntau // 2, 3 * ntau // 4, ntau - 1]
    cols = plt.cm.viridis(np.linspace(0, 1, len(ti_list)))

    # P1: call prices
    ax = axes[0, 0]
    for c, ti in zip(cols, ti_list):
        tv, pv = tg[:, vm, ti], pg[:, vm, ti]
        tp = bs_call_price(strikes, 1.0, tau_u[ti], r, tv)
        pp = bs_call_price(strikes, 1.0, tau_u[ti], r, pv)
        ax.plot(strikes, tp, "-", color=c, label=f"τ={tau_u[ti]:.2f}")
        ax.plot(strikes, pp, "--", color=c, alpha=0.7)
    ax.set_xlabel("K"); ax.set_ylabel("Price"); ax.set_title("Call prices"); ax.legend(fontsize=7)

    # P2: IV smile
    ax = axes[0, 1]
    for c, ti in zip(cols, ti_list):
        ax.plot(strikes, tg[:, vm, ti] * 100, "-", color=c, label=f"τ={tau_u[ti]:.2f}")
        ax.plot(strikes, pg[:, vm, ti] * 100, "--", color=c, alpha=0.7)
    ax.set_xlabel("K"); ax.set_ylabel("IV (%)"); ax.set_title("IV smile"); ax.legend(fontsize=7)

    # P3: price error
    ax = axes[0, 2]
    for c, ti in zip(cols, ti_list):
        tv, pv = tg[:, vm, ti], pg[:, vm, ti]
        tp = bs_call_price(strikes, 1.0, tau_u[ti], r, tv)
        pp = bs_call_price(strikes, 1.0, tau_u[ti], r, pv)
        ax.plot(strikes, (pp - tp) * 10_000, "-", color=c, label=f"τ={tau_u[ti]:.2f}")
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("K"); ax.set_ylabel("bps"); ax.set_title("Price error"); ax.legend(fontsize=7)

    # P4: ATM term structure
    ax = axes[1, 0]
    ax.plot(tau_u, tg[nx // 2, vm, :] * 100, "b-", lw=2, label="True")
    ax.plot(tau_u, pg[nx // 2, vm, :] * 100, "r--", lw=2, label="Pred")
    ax.set_xlabel("τ"); ax.set_ylabel("ATM IV (%)"); ax.set_title("ATM term structure"); ax.legend()

    # P5: error histogram
    ax = axes[1, 1]
    ax.hist(np.clip(iv_errors_bps, -200, 200), bins=100, alpha=0.7, density=True)
    ax.set_xlabel("IV Error (bps)"); ax.set_title(f"Error dist (MAE={np.abs(iv_errors_bps).mean():.1f} bps)")

    # P6: scatter
    ax = axes[1, 2]
    n = min(20_000, pred.size)
    idx = np.random.choice(pred.size, n, replace=False)
    ax.scatter(true.ravel()[idx] * 100, pred.ravel()[idx] * 100, alpha=0.1, s=1)
    ax.plot([0, 100], [0, 100], "r--")
    ax.set_xlabel("True IV (%)"); ax.set_ylabel("Pred IV (%)"); ax.set_title("Pred vs True")

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_plots.png", dpi=150)
    plt.close()
    print(f"Saved → {output_dir / 'evaluation_plots.png'}")

    # surface comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    tau_2d, k_2d = np.meshgrid(tau_u, strikes)
    ts = tg[:, vm, :] * 100
    ps = pg[:, vm, :] * 100
    vmin, vmax = min(ts.min(), ps.min()), max(ts.max(), ps.max())
    for ax, surf, title in zip(axes[:2], [ts, ps], ["True IV", "Pred IV"]):
        im = ax.pcolormesh(tau_2d, k_2d, surf, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label="IV (%)")
        ax.set_xlabel("τ"); ax.set_ylabel("K"); ax.set_title(title)
    im = axes[2].pcolormesh(tau_2d, k_2d, np.abs(ps - ts) * 100, shading="auto", cmap="Reds")
    plt.colorbar(im, ax=axes[2], label="bps"); axes[2].set_xlabel("τ"); axes[2].set_ylabel("K"); axes[2].set_title("|Error|")
    plt.tight_layout()
    plt.savefig(output_dir / "iv_surface.png", dpi=150)
    plt.close()
    print(f"Saved → {output_dir / 'iv_surface.png'}")
