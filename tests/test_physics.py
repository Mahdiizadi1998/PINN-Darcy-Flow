"""
Tests for the physics module.

Verifies that the autograd-based PDE residual computation is correct
by testing on functions with known analytical derivatives.

Run: python -m pytest tests/test_physics.py -v
"""

import torch
import pytest
from src.model import DarcyPINN, HardBCDarcyPINN
from src.physics import compute_darcy_residual


class TestDarcyResidual:
    """Test PDE residual computation against known solutions."""

    def test_zero_residual_on_exact_solution(self):
        """
        If we feed the EXACT solution into the residual computation,
        the residual should be (near) zero.

        Test case: K=1, p(x,y) = sin(πx)sin(πy)
        Then: -∇²p = 2π²sin(πx)sin(πy) = f(x,y)

        We can't feed the exact solution through the network, but we
        can test the autograd machinery by creating a "network" that
        outputs the exact function.
        """
        # Create points
        N = 500
        x = torch.rand(N, 1, requires_grad=True)
        y = torch.rand(N, 1, requires_grad=True)

        # The "prediction" is the exact solution
        p_exact = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

        # Compute derivatives manually via autograd
        dp_dx = torch.autograd.grad(
            p_exact, x, torch.ones_like(p_exact), create_graph=True
        )[0]
        dp_dy = torch.autograd.grad(
            p_exact, y, torch.ones_like(p_exact), create_graph=True
        )[0]

        K = torch.ones_like(x)  # constant permeability
        flux_x = K * dp_dx
        flux_y = K * dp_dy

        dflux_x_dx = torch.autograd.grad(
            flux_x, x, torch.ones_like(flux_x), create_graph=True
        )[0]
        dflux_y_dy = torch.autograd.grad(
            flux_y, y, torch.ones_like(flux_y), create_graph=True
        )[0]

        divergence = dflux_x_dx + dflux_y_dy

        # Source term for this exact solution
        f = 2.0 * (torch.pi ** 2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

        residual = -divergence - f

        # Should be numerically zero
        assert residual.abs().max().item() < 1e-5, (
            f"Residual should be ~0 for exact solution, got max={residual.abs().max().item():.2e}"
        )

    def test_nonzero_residual_for_wrong_solution(self):
        """If we use a wrong solution, the residual should be non-zero."""
        N = 500
        x = torch.rand(N, 1, requires_grad=True)
        y = torch.rand(N, 1, requires_grad=True)

        # Wrong solution: p = x² + y² (doesn't satisfy our PDE)
        p_wrong = x**2 + y**2

        dp_dx = torch.autograd.grad(
            p_wrong, x, torch.ones_like(p_wrong), create_graph=True
        )[0]
        dp_dy = torch.autograd.grad(
            p_wrong, y, torch.ones_like(p_wrong), create_graph=True
        )[0]

        d2p_dx2 = torch.autograd.grad(
            dp_dx, x, torch.ones_like(dp_dx), create_graph=True
        )[0]
        d2p_dy2 = torch.autograd.grad(
            dp_dy, y, torch.ones_like(dp_dy), create_graph=True
        )[0]

        # For K=1: -∇²p = -(2 + 2) = -4
        # But f = 2π²sin(πx)sin(πy) ≠ -4
        f = 2.0 * (torch.pi ** 2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        residual = -(d2p_dx2 + d2p_dy2) - f

        # Should NOT be zero
        assert residual.abs().mean().item() > 0.1, (
            "Residual should be non-zero for wrong solution"
        )

    def test_model_forward_shape(self):
        """Verify model output shapes."""
        model = DarcyPINN(n_frequencies=32, hidden_width=64, n_residual_blocks=2)
        N = 100
        x = torch.rand(N, 1)
        y = torch.rand(N, 1)
        p = model(x, y)
        assert p.shape == (N, 1), f"Expected (100, 1), got {p.shape}"

    def test_hard_bc_model_boundary_values(self):
        """HardBCDarcyPINN should produce exactly zero on boundaries."""
        model = HardBCDarcyPINN(n_frequencies=32, hidden_width=64, n_residual_blocks=2)
        N = 200

        # Points on x=0 boundary
        x_left = torch.zeros(N, 1)
        y_left = torch.rand(N, 1)
        p_left = model(x_left, y_left)
        assert p_left.abs().max().item() < 1e-7, "p should be 0 on x=0 boundary"

        # Points on x=1 boundary
        x_right = torch.ones(N, 1)
        y_right = torch.rand(N, 1)
        p_right = model(x_right, y_right)
        assert p_right.abs().max().item() < 1e-7, "p should be 0 on x=1 boundary"

        # Points on y=0 boundary
        x_bottom = torch.rand(N, 1)
        y_bottom = torch.zeros(N, 1)
        p_bottom = model(x_bottom, y_bottom)
        assert p_bottom.abs().max().item() < 1e-7, "p should be 0 on y=0 boundary"

    def test_fourier_features_increase_dimension(self):
        """Verify Fourier embedding increases input dimensionality."""
        from src.model import FourierFeatureEmbedding
        ff = FourierFeatureEmbedding(in_dim=2, n_frequencies=64, sigma=2.0)
        x = torch.rand(50, 2)
        out = ff(x)
        assert out.shape == (50, 128), f"Expected (50, 128), got {out.shape}"

    def test_gradient_flow(self):
        """Verify gradients flow through PDE residual computation."""
        model = DarcyPINN(n_frequencies=32, hidden_width=64, n_residual_blocks=2)
        x = torch.rand(50, 1, requires_grad=True)
        y = torch.rand(50, 1, requires_grad=True)

        K_fn = lambda x, y: torch.ones_like(x)
        f_fn = lambda x, y: torch.ones_like(x)

        result = compute_darcy_residual(model, x, y, K_fn, f_fn)
        loss = torch.mean(result["residual"] ** 2)
        loss.backward()

        # Check that model parameters received gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad, "Gradients should flow through PDE residual to model parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
