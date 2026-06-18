"""
PINN architecture for Darcy flow.

Design choices:
- Fourier feature embedding to overcome spectral bias (Tancik et al., 2020)
- Residual connections for stable deep training
- Tanh activation for smooth, infinitely differentiable outputs (required for autograd PDE residual)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class FourierFeatureEmbedding(nn.Module):
    """
    Maps low-dimensional coordinates to a higher-dimensional space using
    random Fourier features: γ(x) = [sin(2πBx), cos(2πBx)]

    This breaks the spectral bias of standard MLPs, which learn low-frequency
    functions first and struggle with fine spatial details.

    Args:
        in_dim: Input dimension (2 for spatial coordinates x, y)
        n_frequencies: Number of random frequency components
        sigma: Standard deviation of the frequency matrix B.
               Controls the range of learnable frequencies.
               Higher sigma → captures finer spatial details but harder to train.
    """

    def __init__(self, in_dim: int = 2, n_frequencies: int = 64, sigma: float = 2.0):
        super().__init__()
        self.n_frequencies = n_frequencies
        # Fixed random frequency matrix (not learned)
        B = torch.randn(in_dim, n_frequencies) * sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) tensor of (x, y) coordinates
        Returns:
            (N, 2 * n_frequencies) embedded features
        """
        projection = 2.0 * math.pi * coords @ self.B  # (N, n_frequencies)
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block: activation → linear → activation → linear + skip.

    Residual connections help with:
    1. Gradient flow in deeper networks
    2. Training stability for PDE loss landscapes (which are notoriously ill-conditioned)
    """

    def __init__(self, width: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DarcyPINN(nn.Module):
    """
    Physics-Informed Neural Network for steady-state Darcy flow.

    Architecture:
        (x, y) → Fourier Features → FC layers with residual blocks → p(x, y)

    Args:
        n_frequencies: Number of Fourier embedding frequencies
        sigma: Fourier feature frequency scale
        hidden_width: Width of hidden layers
        n_residual_blocks: Number of residual blocks in the trunk
        use_fourier_features: If False, use raw (x, y) input (for ablation study)
    """

    def __init__(
        self,
        n_frequencies: int = 64,
        sigma: float = 2.0,
        hidden_width: int = 128,
        n_residual_blocks: int = 4,
        use_fourier_features: bool = True,
    ):
        super().__init__()
        self.use_fourier_features = use_fourier_features

        # Input embedding
        if use_fourier_features:
            self.embedding = FourierFeatureEmbedding(
                in_dim=2, n_frequencies=n_frequencies, sigma=sigma
            )
            input_dim = self.embedding.out_dim
        else:
            self.embedding = None
            input_dim = 2

        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_width)

        # Residual trunk
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_width) for _ in range(n_residual_blocks)]
        )

        # Output projection
        self.output_layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width // 2),
            nn.Tanh(),
            nn.Linear(hidden_width // 2, 1),
        )

        # Initialize weights using Xavier (good for Tanh networks)
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization tuned for Tanh activation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict pressure at given coordinates.

        Args:
            x: (N, 1) x-coordinates (must have requires_grad=True for PDE loss)
            y: (N, 1) y-coordinates (must have requires_grad=True for PDE loss)

        Returns:
            p: (N, 1) predicted pressure values
        """
        coords = torch.cat([x, y], dim=-1)  # (N, 2)

        if self.use_fourier_features:
            h = self.embedding(coords)
        else:
            h = coords

        h = self.input_layer(h)
        h = self.residual_blocks(h)
        p = self.output_layer(h)

        return p

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HardBCDarcyPINN(DarcyPINN):
    """
    Variant that enforces Dirichlet BCs exactly through output transformation.

    Instead of penalizing BC violations in the loss, we multiply the network
    output by a distance function that is zero on the boundary:

        p(x, y) = x(1-x) · y(1-y) · NN(x, y)

    This guarantees p = 0 on ∂Ω exactly, regardless of the network output.
    The optimizer only needs to handle the PDE residual — one fewer loss term.

    Trade-off: the distance function modifies the gradient landscape and can
    make optimization harder for some problems.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Raw network output (unbounded)
        nn_output = super().forward(x, y)

        # Distance function: zero on all four boundaries of [0,1]²
        distance = x * (1.0 - x) * y * (1.0 - y)

        # Enforced BC: p = 0 on ∂Ω exactly
        return distance * nn_output
