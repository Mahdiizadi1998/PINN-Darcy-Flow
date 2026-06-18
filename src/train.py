"""
Training loop for PINN with:
- Adaptive loss weighting (gradient normalization, Wang et al. 2021)
- Periodic collocation point resampling
- Cosine annealing learning rate schedule
- Logging and checkpointing
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field

from src.model import DarcyPINN
from src.physics import compute_darcy_residual, compute_boundary_loss, compute_data_loss
from src.data import sample_interior_points, sample_boundary_points


@dataclass
class TrainingConfig:
    """All training hyperparameters in one place."""

    # Optimization
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    epochs: int = 15000
    optimizer: str = "adam"  # "adam" or "lbfgs"

    # Collocation sampling
    n_interior: int = 4000
    n_boundary_per_edge: int = 200
    resample_every: int = 500  # re-draw interior points periodically

    # Loss weights (initial — will be adapted if adaptive=True)
    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    lambda_data: float = 1.0

    # Adaptive weighting
    adaptive_weights: bool = True
    adaptive_alpha: float = 0.9  # EMA smoothing for weight updates

    # Sparse data (set to 0 to disable)
    n_data_points: int = 0
    data_noise_std: float = 0.0

    # Logging
    log_every: int = 500
    save_every: int = 5000

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingHistory:
    """Stores loss curves and metrics during training."""

    epoch: List[int] = field(default_factory=list)
    loss_total: List[float] = field(default_factory=list)
    loss_pde: List[float] = field(default_factory=list)
    loss_bc: List[float] = field(default_factory=list)
    loss_data: List[float] = field(default_factory=list)
    lambda_pde: List[float] = field(default_factory=list)
    lambda_bc: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)


def compute_adaptive_weights(
    model: DarcyPINN,
    loss_pde: torch.Tensor,
    loss_bc: torch.Tensor,
    current_lambda_pde: float,
    current_lambda_bc: float,
    alpha: float = 0.9,
) -> tuple:
    """
    Gradient-normalization based adaptive loss weighting.

    Idea (Wang et al., 2021): balance the loss terms so that the gradients
    from each term have similar magnitudes. Without this, the BC loss often
    dominates early training, and the PDE loss gets neglected.

    The method computes the mean absolute gradient of each loss w.r.t.
    the last layer's weights, then rescales weights to equalize them.

    Args:
        model: The PINN model
        loss_pde: PDE residual loss (scalar)
        loss_bc: Boundary condition loss (scalar)
        current_lambda_pde: Current PDE weight
        current_lambda_bc: Current BC weight
        alpha: EMA smoothing factor (higher = more stable, slower adaptation)

    Returns:
        new_lambda_pde, new_lambda_bc: Updated loss weights
    """
    # Get gradients of each loss w.r.t. last layer parameters
    last_layer_params = list(model.output_layer.parameters())

    grad_pde = torch.autograd.grad(
        loss_pde, last_layer_params, retain_graph=True, allow_unused=True
    )
    grad_bc = torch.autograd.grad(
        loss_bc, last_layer_params, retain_graph=True, allow_unused=True
    )

    # Mean absolute gradient magnitude
    def grad_norm(grads):
        total = 0.0
        count = 0
        for g in grads:
            if g is not None:
                total += g.abs().mean().item()
                count += 1
        return total / max(count, 1)

    norm_pde = grad_norm(grad_pde)
    norm_bc = grad_norm(grad_bc)

    # Target: equal gradient contributions
    # New weight = mean_norm / individual_norm
    mean_norm = (norm_pde + norm_bc) / 2.0
    target_lambda_pde = mean_norm / (norm_pde + 1e-8)
    target_lambda_bc = mean_norm / (norm_bc + 1e-8)

    # EMA smoothing to avoid oscillations
    new_lambda_pde = alpha * current_lambda_pde + (1 - alpha) * target_lambda_pde
    new_lambda_bc = alpha * current_lambda_bc + (1 - alpha) * target_lambda_bc

    return new_lambda_pde, new_lambda_bc


def train(
    model: DarcyPINN,
    permeability_fn: Callable,
    source_fn: Callable,
    config: TrainingConfig,
    exact_solution_fn: Optional[Callable] = None,
    x_data: Optional[torch.Tensor] = None,
    y_data: Optional[torch.Tensor] = None,
    p_data: Optional[torch.Tensor] = None,
) -> TrainingHistory:
    """
    Main training loop.

    Args:
        model: PINN model (will be modified in-place)
        permeability_fn: K(x, y) → permeability
        source_fn: f(x, y) → source term
        config: Training hyperparameters
        exact_solution_fn: Optional analytical solution for monitoring error
        x_data, y_data, p_data: Optional sparse observations

    Returns:
        TrainingHistory with loss curves
    """
    device = torch.device(config.device)
    model = model.to(device)

    # Optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=config.learning_rate,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Learning rate scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.min_learning_rate
    )

    # Boundary points (fixed throughout training)
    x_bc, y_bc, p_bc = sample_boundary_points(config.n_boundary_per_edge, device)

    # Initialize adaptive weights
    lambda_pde = config.lambda_pde
    lambda_bc = config.lambda_bc

    # Move sparse data to device if provided
    has_data = x_data is not None and y_data is not None and p_data is not None
    if has_data:
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        p_data = p_data.to(device)

    # Training history
    history = TrainingHistory()
    start_time = time.time()

    print(f"Training PINN on {device}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Interior points: {config.n_interior}")
    print(f"  Boundary points: {4 * config.n_boundary_per_edge}")
    if has_data:
        print(f"  Data points: {x_data.shape[0]}")
    print(f"  Epochs: {config.epochs}")
    print("-" * 60)

    for epoch in range(1, config.epochs + 1):
        model.train()

        # Resample interior collocation points periodically
        if epoch == 1 or epoch % config.resample_every == 0:
            x_int, y_int = sample_interior_points(
                config.n_interior, device, requires_grad=True
            )

        # --- Compute losses ---

        # PDE residual loss
        pde_result = compute_darcy_residual(
            model, x_int, y_int, permeability_fn, source_fn
        )
        loss_pde = torch.mean(pde_result["residual"] ** 2)

        # Boundary condition loss
        loss_bc = compute_boundary_loss(model, x_bc, y_bc, p_bc)

        # Sparse data loss (optional)
        loss_data = torch.tensor(0.0, device=device)
        if has_data:
            loss_data = compute_data_loss(model, x_data, y_data, p_data)

        # Adaptive weight update
        if config.adaptive_weights and epoch % 100 == 0 and epoch > 100:
            lambda_pde, lambda_bc = compute_adaptive_weights(
                model, loss_pde, loss_bc,
                lambda_pde, lambda_bc,
                alpha=config.adaptive_alpha,
            )

        # Total loss
        loss_total = (
            lambda_pde * loss_pde
            + lambda_bc * loss_bc
            + config.lambda_data * loss_data
        )

        # --- Optimization step ---
        if config.optimizer == "adam":
            optimizer.zero_grad()
            loss_total.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        elif config.optimizer == "lbfgs":
            def closure():
                optimizer.zero_grad()
                pde_res = compute_darcy_residual(
                    model, x_int, y_int, permeability_fn, source_fn
                )
                l_pde = torch.mean(pde_res["residual"] ** 2)
                l_bc = compute_boundary_loss(model, x_bc, y_bc, p_bc)
                l_total = lambda_pde * l_pde + lambda_bc * l_bc
                l_total.backward()
                return l_total
            optimizer.step(closure)

        scheduler.step()

        # --- Logging ---
        if epoch % config.log_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]

            history.epoch.append(epoch)
            history.loss_total.append(loss_total.item())
            history.loss_pde.append(loss_pde.item())
            history.loss_bc.append(loss_bc.item())
            history.loss_data.append(loss_data.item())
            history.lambda_pde.append(lambda_pde)
            history.lambda_bc.append(lambda_bc)
            history.learning_rate.append(lr)
            history.wall_time.append(elapsed)

            # Compute exact error if available
            error_str = ""
            if exact_solution_fn is not None:
                with torch.no_grad():
                    x_test = x_int.detach()
                    y_test = y_int.detach()
                    p_pred = model(x_test, y_test)
                    p_exact = exact_solution_fn(x_test, y_test)
                    rel_l2 = torch.norm(p_pred - p_exact) / torch.norm(p_exact)
                    error_str = f" | Rel. L2: {rel_l2:.4e}"

            print(
                f"Epoch {epoch:>6d} | "
                f"Loss: {loss_total.item():.4e} | "
                f"PDE: {loss_pde.item():.4e} | "
                f"BC: {loss_bc.item():.4e}"
                f"{error_str} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

    print("-" * 60)
    print(f"Training complete. Total time: {time.time() - start_time:.1f}s")

    return history
