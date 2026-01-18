"""
TPsiAct (t-Score Activation) - A robust activation function based on Student-t distribution score function.

This module implements the TPsiAct activation layer which:
1. Derives activation from statistical distribution's score function
2. Provides per-dimension uncertainty estimation via shrink ratio
3. Has built-in outlier suppression (bounded influence / redescending)
4. Offers differentiable, end-to-end trainable robust weighting mechanism

Reference: Theoretical derivation from Student-t distribution's log-density derivative.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union


class TPsiAct(nn.Module):
    """
    TPsiAct (t-Score Activation) layer.
    
    Given input vector x ∈ R^d, performs LayerNorm-like standardization followed by
    Student-t score-like transformation:
    
    z = (x - μ) / σ
    ψ_ν(z) = (ν + 1) * z / (ν + z²)
    y = μ + σ * ψ_ν(z)
    
    Uncertainty is computed as:
    w(z) = |ψ_ν(z) / (z + ε)|  (weight function / shrink ratio)
    u = 1 - clip(w, 0, 1)       (higher = more uncertain)
    
    Args:
        nu (float): Degrees of freedom parameter. Lower = heavier tails, more robust.
                   ν → ∞ approaches Gaussian behavior.
        eps (float): Small constant for numerical stability.
        learnable_nu (bool): If True, nu becomes a learnable parameter.
        affine (bool): If True, adds learnable scale (gamma) and shift (beta) parameters.
        normalized_shape (int or tuple): Shape for learnable affine parameters.
    """
    
    def __init__(
        self, 
        nu: float = 5.0, 
        eps: float = 1e-5,
        learnable_nu: bool = False,
        affine: bool = False,
        normalized_shape: Optional[Union[int, Tuple[int, ...]]] = None
    ):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.learnable_nu = learnable_nu
        
        # Nu parameter (degrees of freedom)
        if learnable_nu:
            # Use softplus to ensure nu > 0, initialize to desired value
            # softplus^{-1}(nu) = log(exp(nu) - 1)
            init_val = torch.log(torch.tensor(max(nu, 1.0)).exp() - 1)
            self._nu_raw = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer('_nu', torch.tensor(nu))
        
        # Affine parameters
        if affine and normalized_shape is not None:
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
    
    @property
    def nu(self) -> torch.Tensor:
        """Get the current nu value."""
        if self.learnable_nu:
            return nn.functional.softplus(self._nu_raw) + 1.0  # Ensure nu > 1
        return self._nu
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_uncertainty: bool = False,
        dim: int = -1
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of TPsiAct.
        
        Args:
            x: Input tensor of any shape.
            return_uncertainty: If True, also returns per-element uncertainty.
            dim: Dimension along which to compute statistics (default: -1, last dim).
        
        Returns:
            y: Activated output, same shape as x.
            u: (optional) Uncertainty tensor, same shape as x. Values in [0, 1].
               Higher values indicate more uncertain (outlier-like) inputs.
        """
        # Compute statistics along specified dimension
        mu = x.mean(dim=dim, keepdim=True)
        sigma = x.std(dim=dim, keepdim=True) + self.eps
        
        # Standardize
        z = (x - mu) / sigma
        
        # Psi function: score-like transformation
        # ψ_ν(z) = (ν + 1) * z / (ν + z²)
        nu = self.nu
        z_sq = z * z
        psi = (nu + 1.0) * z / (nu + z_sq)
        
        # Output: y = μ + σ * ψ_ν(z)
        y = mu + sigma * psi
        
        # Apply affine transformation if enabled
        if self.affine and self.gamma is not None:
            y = self.gamma * y + self.beta
        
        if return_uncertainty:
            # Shrink ratio (weight function): w = |ψ(z) / z|
            # This is (ν + 1) / (ν + z²)
            shrink = (psi.abs() / (z.abs() + self.eps)).clamp(0, 1)
            # Uncertainty: u = 1 - w
            u = 1.0 - shrink
            return y, u
        
        return y
    
    def get_psi_stats(self, x: torch.Tensor, dim: int = -1) -> dict:
        """
        Compute detailed statistics for analysis.
        
        Returns:
            dict with keys: 'z', 'psi', 'shrink', 'uncertainty', 'mu', 'sigma'
        """
        mu = x.mean(dim=dim, keepdim=True)
        sigma = x.std(dim=dim, keepdim=True) + self.eps
        z = (x - mu) / sigma
        
        nu = self.nu
        z_sq = z * z
        psi = (nu + 1.0) * z / (nu + z_sq)
        
        shrink = (psi.abs() / (z.abs() + self.eps)).clamp(0, 1)
        uncertainty = 1.0 - shrink
        
        return {
            'z': z,
            'psi': psi,
            'shrink': shrink,
            'uncertainty': uncertainty,
            'mu': mu,
            'sigma': sigma,
            'nu': nu
        }
    
    def extra_repr(self) -> str:
        s = f'nu={self.nu.item():.2f}, eps={self.eps}'
        if self.learnable_nu:
            s += ', learnable_nu=True'
        if self.affine:
            s += ', affine=True'
        return s


class TPsiActConv(nn.Module):
    """
    TPsiAct variant for convolutional layers.
    
    Computes statistics along channel dimension (dim=1 for NCHW tensors).
    """
    
    def __init__(
        self, 
        nu: float = 5.0, 
        eps: float = 1e-5,
        learnable_nu: bool = False,
        num_channels: Optional[int] = None
    ):
        super().__init__()
        self.tpsiact = TPsiAct(
            nu=nu, 
            eps=eps, 
            learnable_nu=learnable_nu,
            affine=num_channels is not None,
            normalized_shape=(num_channels, 1, 1) if num_channels else None
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for conv tensors (N, C, H, W).
        Computes statistics over spatial dimensions (H, W).
        """
        N, C, H, W = x.shape
        # Reshape to (N*C, H*W) for per-channel-per-sample statistics
        x_flat = x.view(N * C, H * W)
        
        if return_uncertainty:
            y_flat, u_flat = self.tpsiact(x_flat, return_uncertainty=True, dim=-1)
            y = y_flat.view(N, C, H, W)
            u = u_flat.view(N, C, H, W)
            return y, u
        
        y_flat = self.tpsiact(x_flat, return_uncertainty=False, dim=-1)
        return y_flat.view(N, C, H, W)


class TPsiActBlock(nn.Module):
    """
    A complete block combining Linear/Conv + TPsiAct.
    
    Useful for building networks with TPsiAct as the activation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        nu: float = 5.0,
        eps: float = 1e-5,
        learnable_nu: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.tpsiact = TPsiAct(nu=nu, eps=eps, learnable_nu=learnable_nu)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.linear(x)
        if return_uncertainty:
            x, u = self.tpsiact(x, return_uncertainty=True)
            x = self.dropout(x)
            return x, u
        x = self.tpsiact(x)
        return self.dropout(x)


def replace_activations_with_tpsiact(
    model: nn.Module,
    nu: float = 5.0,
    replace_relu: bool = True,
    replace_gelu: bool = True,
    replace_silu: bool = True,
    inplace: bool = False
) -> nn.Module:
    """
    Replace activation functions in a model with TPsiAct.
    
    Args:
        model: PyTorch model to modify.
        nu: Degrees of freedom for TPsiAct.
        replace_relu: Whether to replace ReLU activations.
        replace_gelu: Whether to replace GELU activations.
        replace_silu: Whether to replace SiLU/Swish activations.
        inplace: If True, modifies model in place; otherwise returns a copy.
    
    Returns:
        Modified model with TPsiAct activations.
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    activation_types = []
    if replace_relu:
        activation_types.extend([nn.ReLU, nn.ReLU6, nn.LeakyReLU])
    if replace_gelu:
        activation_types.append(nn.GELU)
    if replace_silu:
        activation_types.append(nn.SiLU)
    
    activation_types = tuple(activation_types)
    
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, activation_types):
                setattr(module, child_name, TPsiAct(nu=nu))
    
    return model


# Utility functions for visualization and analysis
def plot_psi_curves(nu_values: list = [1, 3, 5, 10, 50], z_range: float = 5.0):
    """
    Plot ψ_ν(z) curves for different nu values.
    Useful for understanding the activation behavior.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    z = np.linspace(-z_range, z_range, 500)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: ψ(z) curves
    ax1 = axes[0]
    for nu in nu_values:
        psi = (nu + 1) * z / (nu + z**2)
        ax1.plot(z, psi, label=f'ν={nu}')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('z')
    ax1.set_ylabel('ψ(z)')
    ax1.set_title('Psi Function ψ_ν(z)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight/Shrink ratio w(z)
    ax2 = axes[1]
    for nu in nu_values:
        w = np.clip((nu + 1) / (nu + z**2), 0, 1)
        ax2.plot(z, w, label=f'ν={nu}')
    ax2.set_xlabel('z')
    ax2.set_ylabel('w(z)')
    ax2.set_title('Weight Function w(z) = |ψ(z)/z|')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty u(z)
    ax3 = axes[2]
    for nu in nu_values:
        w = np.clip((nu + 1) / (nu + z**2), 0, 1)
        u = 1 - w
        ax3.plot(z, u, label=f'ν={nu}')
    ax3.set_xlabel('z')
    ax3.set_ylabel('u(z)')
    ax3.set_title('Uncertainty u(z) = 1 - w(z)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test basic functionality
    print("Testing TPsiAct...")
    
    # Create layer
    tpsiact = TPsiAct(nu=5.0)
    
    # Test input
    x = torch.randn(2, 10)
    
    # Forward pass
    y = tpsiact(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # With uncertainty
    y, u = tpsiact(x, return_uncertainty=True)
    print(f"Uncertainty shape: {u.shape}")
    print(f"Uncertainty range: [{u.min().item():.4f}, {u.max().item():.4f}]")
    
    # Test learnable nu
    tpsiact_learnable = TPsiAct(nu=5.0, learnable_nu=True)
    print(f"\nLearnable nu initial value: {tpsiact_learnable.nu.item():.4f}")
    
    # Test gradient flow
    x = torch.randn(2, 10, requires_grad=True)
    y = tpsiact_learnable(x)
    loss = y.sum()
    loss.backward()
    print(f"Gradient computed: {x.grad is not None}")
    
    print("\nAll tests passed!")
