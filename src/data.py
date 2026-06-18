"""
Data generation module.

Handles:
1. Collocation point sampling (interior + boundary)
2. Reference solution via finite differences (for cases without analytical solutions)
3. Sparse observation data (simulating well measurements)
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict


def sample_interior_points(
    n_points: int, device: torch.device, requires_grad: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random collocation points in the interior of Ω = (0, 1)².

    Using random (Latin Hypercube or uniform) sampling instead of a fixed grid
    avoids aliasing artifacts and provides better coverage in high dimensions.

    Args:
        n_points: Number of interior points
        device: torch device
        requires_grad: Must be True for autograd-based PDE residual

    Returns:
        x, y: (n_points, 1) tensors
    """
    x = torch.rand(n_points, 1, device=device, requires_grad=requires_grad)
    y = torch.rand(n_points, 1, device=device, requires_grad=requires_grad)
    return x, y


def sample_boundary_points(
    n_per_edge: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample points on all four edges of ∂Ω.

    For Dirichlet BC p = 0, the prescribed values are all zeros.

    Args:
        n_per_edge: Points per edge (total = 4 * n_per_edge)
        device: torch device

    Returns:
        x_bc, y_bc: (4*n_per_edge, 1) boundary coordinates
        p_bc: (4*n_per_edge, 1) prescribed pressure (zeros)
    """
    t = torch.linspace(0, 1, n_per_edge, device=device).unsqueeze(1)
    zeros = torch.zeros(n_per_edge, 1, device=device)
    ones = torch.ones(n_per_edge, 1, device=device)

    # Four edges: bottom (y=0), top (y=1), left (x=0), right (x=1)
    x_bc = torch.cat([t, t, zeros, ones], dim=0)
    y_bc = torch.cat([zeros, ones, t, t], dim=0)
    p_bc = torch.zeros_like(x_bc)

    return x_bc, y_bc, p_bc


def sample_sparse_observations(
    exact_solution_fn,
    n_obs: int,
    noise_std: float = 0.0,
    device: torch.device = torch.device("cpu"),
    seed: int = 123,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate sparse 'observation' data — simulating pressure measurements at well locations.

    Used for the data-assimilation scenario where we have a few real measurements
    to supplement the physics loss.

    Args:
        exact_solution_fn: Function (x, y) → p that provides ground truth
        n_obs: Number of observation points
        noise_std: Gaussian noise standard deviation (0 = perfect measurements)
        device: torch device
        seed: Random seed for reproducible observation locations

    Returns:
        x_obs, y_obs, p_obs: (n_obs, 1) tensors
    """
    torch.manual_seed(seed)
    x_obs = torch.rand(n_obs, 1, device=device)
    y_obs = torch.rand(n_obs, 1, device=device)

    with torch.no_grad():
        p_obs = exact_solution_fn(x_obs, y_obs)
        if noise_std > 0:
            p_obs = p_obs + noise_std * torch.randn_like(p_obs)

    return x_obs, y_obs, p_obs


def generate_evaluation_grid(
    n_grid: int = 100, device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a regular grid for evaluating and visualizing the solution.

    Returns:
        x_flat, y_flat: (n_grid², 1) flattened grid coordinates
        X, Y: (n_grid, n_grid) meshgrid arrays for plotting
    """
    x_1d = torch.linspace(0, 1, n_grid, device=device)
    y_1d = torch.linspace(0, 1, n_grid, device=device)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing="ij")

    x_flat = X.reshape(-1, 1).requires_grad_(True)
    y_flat = Y.reshape(-1, 1).requires_grad_(True)

    return x_flat, y_flat, X, Y


def solve_darcy_fd(
    permeability_fn,
    source_fn,
    n_grid: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Darcy equation using second-order finite differences.

    This provides a reference solution for cases without analytical answers.
    Not meant for production use — just a verification baseline.

    Discretization:
        -[K_{i+½,j}(p_{i+1,j} - p_{i,j}) - K_{i-½,j}(p_{i,j} - p_{i-1,j})] / h²
        -[K_{i,j+½}(p_{i,j+1} - p_{i,j}) - K_{i,j-½}(p_{i,j} - p_{i,j-1})] / h²
        = f_{i,j}

    Args:
        permeability_fn: K(x, y) function
        source_fn: f(x, y) function
        n_grid: Number of interior grid points per dimension
        device: torch device (used for K and f evaluation)

    Returns:
        p_grid: (n_grid+2, n_grid+2) pressure solution including boundaries
        X, Y: meshgrid arrays
        x_1d, y_1d: 1D coordinate arrays
    """
    n = n_grid
    h = 1.0 / (n + 1)

    # Interior grid
    x_1d = np.linspace(h, 1 - h, n)
    y_1d = np.linspace(h, 1 - h, n)
    X_int, Y_int = np.meshgrid(x_1d, y_1d, indexing="ij")

    # Evaluate K and f on the grid
    with torch.no_grad():
        x_t = torch.tensor(X_int.reshape(-1, 1), dtype=torch.float32, device=device)
        y_t = torch.tensor(Y_int.reshape(-1, 1), dtype=torch.float32, device=device)
        K_vals = permeability_fn(x_t, y_t).cpu().numpy().reshape(n, n)
        f_vals = source_fn(x_t, y_t).cpu().numpy().reshape(n, n)

    # Harmonic average for permeability at cell interfaces
    # K_{i+½,j} = 2·K_{i,j}·K_{i+1,j} / (K_{i,j} + K_{i+1,j})
    K_padded = np.pad(K_vals, 1, mode="edge")

    # Build sparse system Ap = b
    N = n * n
    A = np.zeros((N, N))
    b = np.zeros(N)

    def idx(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            k = idx(i, j)

            # Harmonic averages at interfaces
            K_c = K_vals[i, j]
            K_xp = 2 * K_c * K_padded[i + 2, j + 1] / (K_c + K_padded[i + 2, j + 1] + 1e-12)
            K_xm = 2 * K_c * K_padded[i, j + 1] / (K_c + K_padded[i, j + 1] + 1e-12)
            K_yp = 2 * K_c * K_padded[i + 1, j + 2] / (K_c + K_padded[i + 1, j + 2] + 1e-12)
            K_ym = 2 * K_c * K_padded[i + 1, j] / (K_c + K_padded[i + 1, j] + 1e-12)

            # Diagonal
            A[k, k] = (K_xp + K_xm + K_yp + K_ym) / h**2

            # Off-diagonals (neighbors in interior)
            if i > 0:
                A[k, idx(i - 1, j)] = -K_xm / h**2
            if i < n - 1:
                A[k, idx(i + 1, j)] = -K_xp / h**2
            if j > 0:
                A[k, idx(i, j - 1)] = -K_ym / h**2
            if j < n - 1:
                A[k, idx(i, j + 1)] = -K_yp / h**2

            # RHS (boundary terms are zero for Dirichlet p=0)
            b[k] = f_vals[i, j]

    # Solve
    p_interior = np.linalg.solve(A, b).reshape(n, n)

    # Pad with boundary zeros
    p_grid = np.zeros((n + 2, n + 2))
    p_grid[1:-1, 1:-1] = p_interior

    x_full = np.linspace(0, 1, n + 2)
    y_full = np.linspace(0, 1, n + 2)
    X_full, Y_full = np.meshgrid(x_full, y_full, indexing="ij")

    return p_grid, X_full, Y_full, x_full
