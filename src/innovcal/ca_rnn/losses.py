"""Proper-score and differentiable calibration objectives."""

import math

import torch

from innovcal.ca_rnn.distributions import joint_gaussian


def gaussian_nll(
    target: torch.Tensor, mean: torch.Tensor, scale_tril: torch.Tensor
) -> torch.Tensor:
    """Mean joint Gaussian negative log likelihood."""
    return -joint_gaussian(mean, scale_tril).log_prob(target).mean()


def make_projection_matrix(
    dimension: int,
    n_random: int = 4,
    seed: int = 123,
    include_equal_weight: bool = True,
) -> torch.Tensor:
    """Create deterministic unit-norm coordinate, portfolio, and random projections."""
    if dimension < 1 or n_random < 0:
        raise ValueError("dimension must be positive and n_random nonnegative")
    projections = [torch.eye(dimension)]
    if include_equal_weight:
        projections.append(torch.ones(1, dimension) / math.sqrt(dimension))
    if n_random:
        generator = torch.Generator().manual_seed(seed)
        random = torch.randn(n_random, dimension, generator=generator)
        random = random / random.norm(dim=1, keepdim=True).clamp_min(1e-12)
        projections.append(random)
    return torch.cat(projections, dim=0)


def projected_pits(
    target: torch.Tensor,
    mean: torch.Tensor,
    scale_tril: torch.Tensor,
    projections: torch.Tensor,
) -> torch.Tensor:
    """Exact differentiable PIT values for projections of a joint Gaussian."""
    projections = projections.to(device=mean.device, dtype=mean.dtype)
    projected_target = target @ projections.T
    projected_mean = mean @ projections.T
    covariance = scale_tril @ scale_tril.transpose(-1, -2)
    variance = torch.einsum("rd,bdk,rk->br", projections, covariance, projections)
    standardized = (projected_target - projected_mean) / variance.clamp_min(
        1e-12
    ).sqrt()
    return 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))


def calibration_loss(
    pits: torch.Tensor, grid_size: int = 19, temperature: float = 0.03
) -> torch.Tensor:
    """Smoothed finite-grid discrepancy from projected PIT uniformity."""
    if pits.ndim != 2 or grid_size < 2 or temperature <= 0:
        raise ValueError("invalid PIT array or calibration settings")
    grid = torch.linspace(
        1.0 / (grid_size + 1),
        grid_size / (grid_size + 1),
        grid_size,
        device=pits.device,
        dtype=pits.dtype,
    )
    smooth_cdf = torch.sigmoid((grid[None, :, None] - pits[:, None, :]) / temperature)
    empirical = smooth_cdf.mean(dim=0)
    return (empirical - grid[:, None]).square().mean()


def sequential_calibration_loss(pits: torch.Tensor, max_lag: int = 5) -> torch.Tensor:
    """Penalize selected lagged PIT moments centered at the uniform-law mean."""
    if pits.ndim != 2 or max_lag < 1:
        raise ValueError("invalid PIT array or lag count")
    centered = pits - 0.5
    terms = []
    for lag in range(1, min(max_lag, len(pits) - 1) + 1):
        covariance = (centered[lag:] * centered[:-lag]).mean(dim=0)
        terms.append(covariance.square().mean())
    return torch.stack(terms).mean() if terms else pits.new_zeros(())
