"""
Physics module: Darcy flow PDE residual computation via automatic differentiation.

The steady-state Darcy equation:
    -∇·(K(x,y) ∇p(x,y)) = f(x,y)

Expanded:
    -(∂K/∂x · ∂p/∂x + K · ∂²p/∂x² + ∂K/∂y · ∂p/∂y + K · ∂²p/∂y²) = f(x,y)

All derivatives are computed exactly via PyTorch autograd — no finite differences,
no mesh, no discretization error. This is the core advantage of PINNs.
"""

import torch
from typing import Callable, Tuple, Dict
from src.model import DarcyPINN


def compute_darcy_residual(
    model: DarcyPINN,
    x: torch.Tensor,
    y: torch.Tensor,
    permeability_fn: Callable,
    source_fn: Callable,
) -> Dict[str, torch.Tensor]:
    """
    Compute the PDE residual of the Darcy equation at given collocation points.

    The residual r(x,y) = -∇·(K∇p) - f should be zero if p is the true solution.

    Args:
        model: PINN that predicts p(x, y)
        x: (N, 1) x-coordinates with requires_grad=True
        y: (N, 1) y-coordinates with requires_grad=True
        permeability_fn: K(x, y) → (N, 1) permeability values
        source_fn: f(x, y) → (N, 1) source term values

    Returns:
        Dictionary containing:
            - residual: (N, 1) PDE residual at each point
            - p: (N, 1) predicted pressure
            - dp_dx, dp_dy: (N, 1) first derivatives
            - flux_x, flux_y: (N, 1) Darcy flux components q = K·∇p
    """
    # Forward pass: predict pressure
    p = model(x, y)

    # Permeability and source at collocation points
    K = permeability_fn(x, y)
    f = source_fn(x, y)

    # --- First-order derivatives: ∂p/∂x, ∂p/∂y ---
    # create_graph=True is essential: we need to differentiate again for second derivatives
    dp_dx = torch.autograd.grad(
        outputs=p,
        inputs=x,
        grad_outputs=torch.ones_like(p),
        create_graph=True,
        retain_graph=True,
    )[0]

    dp_dy = torch.autograd.grad(
        outputs=p,
        inputs=y,
        grad_outputs=torch.ones_like(p),
        create_graph=True,
        retain_graph=True,
    )[0]

    # --- Darcy flux: q = K · ∇p ---
    flux_x = K * dp_dx
    flux_y = K * dp_dy

    # --- Divergence: ∇·(K∇p) = ∂(K·∂p/∂x)/∂x + ∂(K·∂p/∂y)/∂y ---
    dflux_x_dx = torch.autograd.grad(
        outputs=flux_x,
        inputs=x,
        grad_outputs=torch.ones_like(flux_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    dflux_y_dy = torch.autograd.grad(
        outputs=flux_y,
        inputs=y,
        grad_outputs=torch.ones_like(flux_y),
        create_graph=True,
        retain_graph=True,
    )[0]

    divergence = dflux_x_dx + dflux_y_dy

    # PDE residual: -∇·(K∇p) - f = 0
    residual = -divergence - f

    return {
        "residual": residual,
        "p": p,
        "dp_dx": dp_dx,
        "dp_dy": dp_dy,
        "flux_x": flux_x,
        "flux_y": flux_y,
    }


def compute_boundary_loss(
    model: DarcyPINN,
    x_bc: torch.Tensor,
    y_bc: torch.Tensor,
    p_bc: torch.Tensor,
) -> torch.Tensor:
    """
    Compute boundary condition loss: MSE between prediction and prescribed BC.

    For Dirichlet BC p = 0 on ∂Ω, p_bc is a zero tensor.

    Args:
        model: PINN model
        x_bc, y_bc: (N_b, 1) boundary point coordinates
        p_bc: (N_b, 1) prescribed pressure values on boundary

    Returns:
        Scalar MSE loss for boundary condition
    """
    p_pred = model(x_bc, y_bc)
    return torch.mean((p_pred - p_bc) ** 2)


def compute_data_loss(
    model: DarcyPINN,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    p_data: torch.Tensor,
) -> torch.Tensor:
    """
    Compute data-fitting loss from sparse observations.

    Used in the data-assimilation scenario where a few pressure measurements
    (e.g., from well sensors) are available.

    Args:
        model: PINN model
        x_data, y_data: (N_d, 1) observation point coordinates
        p_data: (N_d, 1) observed pressure values

    Returns:
        Scalar MSE loss for data fitting
    """
    p_pred = model(x_data, y_data)
    return torch.mean((p_pred - p_data) ** 2)


# ============================================================================
#  Permeability fields and source terms for test cases
# ============================================================================

class HomogeneousCase:
    """
    Case 1: Constant permeability with analytical solution.

    K(x,y) = 1
    f(x,y) = 2π²·sin(πx)·sin(πy)
    Exact solution: p(x,y) = sin(πx)·sin(πy)

    This is the validation case — we know the exact answer.
    """

    name = "homogeneous"

    @staticmethod
    def permeability(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    @staticmethod
    def source(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 2.0 * (torch.pi ** 2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    @staticmethod
    def exact_solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    @staticmethod
    def has_exact_solution() -> bool:
        return True


class LayeredCase:
    """
    Case 2: Layered permeability with discontinuity.

    K(x,y) = 1 + 4·𝟙(0.3 ≤ y ≤ 0.7)   (high-perm channel)
    f(x,y) = 10·exp(-50·((x-0.5)² + (y-0.5)²))   (Gaussian source)

    The discontinuous K tests the network's ability to handle
    sharp interfaces — common in real reservoirs (shale barriers,
    high-permeability channels).
    """

    name = "layered"

    @staticmethod
    def permeability(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        K_base = torch.ones_like(x)
        channel = ((y >= 0.3) & (y <= 0.7)).float()
        return K_base + 4.0 * channel

    @staticmethod
    def source(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
        return 10.0 * torch.exp(-50.0 * r2)

    @staticmethod
    def exact_solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("No analytical solution for layered case")

    @staticmethod
    def has_exact_solution() -> bool:
        return False


class LogNormalCase:
    """
    Case 3: Log-normal random permeability field.

    K(x,y) = exp(G(x,y)) where G is a Gaussian random field
    f(x,y) = multi-well pattern (injection + production)

    This is the realistic scenario: heterogeneous permeability drawn
    from a geostatistical model, with an injection well at (0.2, 0.2)
    and a production well at (0.8, 0.8).
    """

    name = "lognormal"

    def __init__(self, grid_size: int = 64, length_scale: float = 0.15, seed: int = 42):
        """
        Pre-generate the random permeability field on a grid.

        Args:
            grid_size: Resolution of the permeability grid
            length_scale: Correlation length of the Gaussian random field
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.K_field = self._generate_grf(grid_size, length_scale, seed)

    def _generate_grf(
        self, n: int, length_scale: float, seed: int
    ) -> torch.Tensor:
        """Generate a 2D Gaussian random field via spectral method."""
        torch.manual_seed(seed)

        # Frequency grid
        kx = torch.fft.fftfreq(n, d=1.0 / n)
        ky = torch.fft.fftfreq(n, d=1.0 / n)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")

        # Spectral density (squared exponential covariance)
        r2 = KX**2 + KY**2
        S = torch.exp(-2.0 * (torch.pi * length_scale) ** 2 * r2)

        # Generate random field
        noise = torch.randn(n, n) + 1j * torch.randn(n, n)
        field = torch.fft.ifft2(torch.sqrt(S) * noise).real

        # Normalize and exponentiate for log-normal
        field = (field - field.mean()) / (field.std() + 1e-8)
        K = torch.exp(0.5 * field)  # moderate variance

        return K

    def permeability(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Interpolate permeability from pre-generated grid to arbitrary points."""
        # Map coordinates to grid indices
        n = self.grid_size
        ix = (x * (n - 1)).long().clamp(0, n - 1).squeeze(-1).cpu()
        iy = (y * (n - 1)).long().clamp(0, n - 1).squeeze(-1).cpu()
        K_values = self.K_field[ix, iy].unsqueeze(-1)
        return K_values.to(x.device)

    @staticmethod
    def source(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Multi-well pattern: injector at (0.2, 0.2), producer at (0.8, 0.8)."""
        r_inj = (x - 0.2) ** 2 + (y - 0.2) ** 2
        r_prod = (x - 0.8) ** 2 + (y - 0.8) ** 2
        well_radius = 0.01
        injection = 50.0 * torch.exp(-r_inj / (2.0 * well_radius))
        production = -50.0 * torch.exp(-r_prod / (2.0 * well_radius))
        return injection + production

    @staticmethod
    def exact_solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("No analytical solution for log-normal case")

    @staticmethod
    def has_exact_solution() -> bool:
        return False


def get_test_case(name: str):
    """Factory function to get a test case by name."""
    cases = {
        "homogeneous": HomogeneousCase,
        "layered": LayeredCase,
        "lognormal": LogNormalCase,
    }
    if name not in cases:
        raise ValueError(f"Unknown test case: {name}. Available: {list(cases.keys())}")

    case_class = cases[name]
    if name == "lognormal":
        return case_class()  # needs instantiation for GRF
    return case_class()  # return instance for uniform interface
