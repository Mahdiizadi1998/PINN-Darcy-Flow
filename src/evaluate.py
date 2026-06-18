"""
Evaluation and visualization module.

Generates publication-quality figures for:
1. Predicted vs reference pressure fields
2. Pointwise error maps
3. Loss convergence history
4. Permeability field visualization
5. Cross-section profiles
6. PDE residual spatial distribution
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Callable, Dict

from src.model import DarcyPINN
from src.data import generate_evaluation_grid, solve_darcy_fd
from src.train import TrainingHistory


# Publication-quality plot settings
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def evaluate_model(
    model: DarcyPINN,
    permeability_fn: Callable,
    source_fn: Callable,
    n_grid: int = 100,
    device: torch.device = torch.device("cpu"),
    exact_solution_fn: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate the trained model on a regular grid and compute metrics.

    Args:
        model: Trained PINN
        permeability_fn: K(x, y)
        source_fn: f(x, y)
        n_grid: Evaluation grid resolution
        device: torch device
        exact_solution_fn: Optional exact solution for error computation

    Returns:
        Dictionary with prediction, reference, errors, and metrics
    """
    model.eval()
    x_flat, y_flat, X, Y = generate_evaluation_grid(n_grid, device)

    with torch.no_grad():
        p_pred = model(x_flat, y_flat).cpu().numpy().reshape(n_grid, n_grid)
        K_field = permeability_fn(x_flat, y_flat).cpu().numpy().reshape(n_grid, n_grid)

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    results = {
        "p_pred": p_pred,
        "K_field": K_field,
        "X": X_np,
        "Y": Y_np,
    }

    # Compare against exact solution if available
    if exact_solution_fn is not None:
        with torch.no_grad():
            p_exact = exact_solution_fn(x_flat, y_flat).cpu().numpy().reshape(n_grid, n_grid)

        error = np.abs(p_pred - p_exact)
        rel_l2 = np.linalg.norm(p_pred - p_exact) / np.linalg.norm(p_exact)
        max_error = np.max(error)
        mean_error = np.mean(error)

        results.update({
            "p_exact": p_exact,
            "pointwise_error": error,
            "relative_l2": rel_l2,
            "max_error": max_error,
            "mean_error": mean_error,
        })

        print(f"\n{'='*40}")
        print(f"  EVALUATION METRICS")
        print(f"{'='*40}")
        print(f"  Relative L2 error:    {rel_l2:.6e}")
        print(f"  Max pointwise error:  {max_error:.6e}")
        print(f"  Mean pointwise error: {mean_error:.6e}")
        print(f"{'='*40}\n")

    else:
        # Compare against finite difference solution
        p_fd, X_fd, Y_fd, _ = solve_darcy_fd(permeability_fn, source_fn, n_grid=n_grid - 2, device=device)

        # Interpolate FD to same grid (approximate — FD grid is slightly different)
        from scipy.interpolate import RegularGridInterpolator
        x_fd_1d = np.linspace(0, 1, p_fd.shape[0])
        y_fd_1d = np.linspace(0, 1, p_fd.shape[1])
        interp = RegularGridInterpolator((x_fd_1d, y_fd_1d), p_fd, method="linear")

        pts = np.stack([X_np.ravel(), Y_np.ravel()], axis=-1)
        p_fd_interp = interp(pts).reshape(n_grid, n_grid)

        error = np.abs(p_pred - p_fd_interp)
        norm_fd = np.linalg.norm(p_fd_interp)
        rel_l2 = np.linalg.norm(p_pred - p_fd_interp) / (norm_fd + 1e-10)

        results.update({
            "p_reference": p_fd_interp,
            "pointwise_error": error,
            "relative_l2": rel_l2,
            "max_error": np.max(error),
            "mean_error": np.mean(error),
        })

        print(f"\n{'='*40}")
        print(f"  EVALUATION METRICS (vs FD reference)")
        print(f"{'='*40}")
        print(f"  Relative L2 error:    {rel_l2:.6e}")
        print(f"  Max pointwise error:  {np.max(error):.6e}")
        print(f"  Mean pointwise error: {np.mean(error):.6e}")
        print(f"{'='*40}\n")

    return results


def plot_prediction_vs_reference(
    results: Dict[str, np.ndarray],
    save_path: str = "figures/prediction_vs_reference.png",
    case_name: str = "",
):
    """
    Three-panel plot: PINN prediction | Reference | Pointwise error.
    """
    X, Y = results["X"], results["Y"]
    p_pred = results["p_pred"]
    p_ref = results.get("p_exact", results.get("p_reference"))
    error = results.get("pointwise_error")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # PINN prediction
    im0 = axes[0].pcolormesh(X, Y, p_pred, shading="auto", cmap="RdBu_r")
    axes[0].set_title("PINN Prediction")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Reference
    if p_ref is not None:
        im1 = axes[1].pcolormesh(X, Y, p_ref, shading="auto", cmap="RdBu_r")
        axes[1].set_title("Reference Solution")
    else:
        axes[1].text(0.5, 0.5, "No reference", ha="center", va="center", transform=axes[1].transAxes)
        im1 = None
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    if im1 is not None:
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Pointwise error
    if error is not None:
        im2 = axes[2].pcolormesh(X, Y, error, shading="auto", cmap="hot_r")
        rel_l2 = results.get("relative_l2", 0)
        axes[2].set_title(f"Pointwise Error (Rel. L²: {rel_l2:.2e})")
    else:
        axes[2].text(0.5, 0.5, "No error data", ha="center", va="center", transform=axes[2].transAxes)
        im2 = None
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    if im2 is not None:
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if case_name:
        fig.suptitle(f"Darcy Flow — {case_name}", fontsize=15, y=1.02)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_loss_history(
    history: TrainingHistory,
    save_path: str = "figures/loss_history.png",
):
    """
    Two-panel plot: loss curves (log scale) and adaptive weights.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Loss curves
    ax1.semilogy(history.epoch, history.loss_pde, label="PDE residual", linewidth=1.5)
    ax1.semilogy(history.epoch, history.loss_bc, label="Boundary", linewidth=1.5)
    ax1.semilogy(history.epoch, history.loss_total, label="Total", linewidth=2, color="black", linestyle="--")
    if any(l > 0 for l in history.loss_data):
        ax1.semilogy(history.epoch, history.loss_data, label="Data", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Adaptive weights
    ax2.plot(history.epoch, history.lambda_pde, label="λ_PDE", linewidth=1.5)
    ax2.plot(history.epoch, history.lambda_bc, label="λ_BC", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Weight")
    ax2.set_title("Adaptive Loss Weights")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_permeability_field(
    results: Dict[str, np.ndarray],
    save_path: str = "figures/permeability_field.png",
    case_name: str = "",
):
    """Visualize the permeability field K(x,y)."""
    X, Y = results["X"], results["Y"]
    K = results["K_field"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(X, Y, K, shading="auto", cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = "Permeability Field K(x, y)"
    if case_name:
        title += f" — {case_name}"
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="K")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_cross_sections(
    results: Dict[str, np.ndarray],
    save_path: str = "figures/cross_sections.png",
    case_name: str = "",
):
    """
    1D cross-sections of the pressure field at y=0.25, 0.5, 0.75.
    Compares PINN prediction vs reference on the same axes.
    """
    X, Y = results["X"], results["Y"]
    p_pred = results["p_pred"]
    p_ref = results.get("p_exact", results.get("p_reference"))

    n = X.shape[0]
    x_1d = X[:, 0]

    y_positions = [0.25, 0.5, 0.75]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, y_pos in zip(axes, y_positions):
        j = int(y_pos * (n - 1))
        ax.plot(x_1d, p_pred[:, j], "b-", linewidth=2, label="PINN")
        if p_ref is not None:
            ax.plot(x_1d, p_ref[:, j], "r--", linewidth=2, label="Reference")
        ax.set_xlabel("x")
        ax.set_ylabel("p(x, y)")
        ax.set_title(f"y = {y_pos}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if case_name:
        fig.suptitle(f"Cross-Section Profiles — {case_name}", fontsize=14, y=1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_pde_residual_map(
    model: DarcyPINN,
    permeability_fn: Callable,
    source_fn: Callable,
    n_grid: int = 80,
    device: torch.device = torch.device("cpu"),
    save_path: str = "figures/pde_residual_map.png",
    case_name: str = "",
):
    """
    Spatial map of the PDE residual — shows where the model satisfies
    or violates the physics.
    """
    from src.physics import compute_darcy_residual

    model.eval()
    x_1d = torch.linspace(0.01, 0.99, n_grid, device=device)  # avoid exact boundary
    y_1d = torch.linspace(0.01, 0.99, n_grid, device=device)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing="ij")

    x_flat = X.reshape(-1, 1).requires_grad_(True)
    y_flat = Y.reshape(-1, 1).requires_grad_(True)

    pde_result = compute_darcy_residual(model, x_flat, y_flat, permeability_fn, source_fn)
    residual = pde_result["residual"].detach().cpu().numpy().reshape(n_grid, n_grid)

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    max_abs = np.percentile(np.abs(residual), 98)  # robust color range
    im = ax.pcolormesh(
        X_np, Y_np, residual, shading="auto",
        cmap="RdBu_r", vmin=-max_abs, vmax=max_abs
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = "PDE Residual"
    if case_name:
        title += f" — {case_name}"
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Residual")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_figures(
    model: DarcyPINN,
    results: Dict[str, np.ndarray],
    history: TrainingHistory,
    permeability_fn: Callable,
    source_fn: Callable,
    device: torch.device,
    case_name: str = "",
    output_dir: str = "figures",
):
    """Generate all publication figures in one call."""
    prefix = f"{output_dir}/{case_name}_" if case_name else f"{output_dir}/"

    plot_prediction_vs_reference(results, f"{prefix}prediction_vs_reference.png", case_name)
    plot_loss_history(history, f"{prefix}loss_history.png")
    plot_permeability_field(results, f"{prefix}permeability_field.png", case_name)
    plot_cross_sections(results, f"{prefix}cross_sections.png", case_name)
    plot_pde_residual_map(model, permeability_fn, source_fn, device=device,
                          save_path=f"{prefix}pde_residual_map.png", case_name=case_name)

    print(f"\nAll figures saved to {output_dir}/")
