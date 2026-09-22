"""Controlled multivariate sequential data-generating processes."""

import numpy as np


def simulate_multivariate_series(
    n_observations: int = 2000,
    dimension: int = 4,
    regime: str = "gaussian",
    seed: int = 123,
    shift_start: int | None = None,
    shift_severity: float = 0.0,
    shift_mechanism: str = "scale_and_correlation",
) -> np.ndarray:
    """Simulate a stable nonlinear autoregression under prespecified shifts.

    Regimes are ``gaussian``, ``heavy_tail``, ``heteroskedastic``, and
    ``nonlinear``. A nonzero shift severity begins at ``shift_start`` and
    changes ``scale``, ``correlation``, or both without changing the seed.
    """
    if n_observations < 50 or dimension < 1 or shift_severity < 0:
        raise ValueError("invalid simulation settings")
    if regime not in {"gaussian", "heavy_tail", "heteroskedastic", "nonlinear"}:
        raise ValueError(f"unknown regime: {regime}")
    if shift_mechanism not in {"scale", "correlation", "scale_and_correlation"}:
        raise ValueError(f"unknown shift mechanism: {shift_mechanism}")
    rng = np.random.default_rng(seed)
    transition = 0.18 * np.eye(dimension) + 0.04 * (
        np.ones((dimension, dimension)) - np.eye(dimension)
    )
    scale = np.linspace(0.8, 1.2, dimension)
    values = np.zeros((n_observations, dimension))
    conditional_scale = np.ones(dimension)
    shift_start = n_observations + 1 if shift_start is None else int(shift_start)
    for index in range(1, n_observations):
        shifted = index >= shift_start
        severity = shift_severity if shifted else 0.0
        corr_weight = min(0.85, 0.3 + 0.4 * severity) if shift_mechanism != "scale" else 0.3
        correlation = corr_weight * np.ones((dimension, dimension)) + (
            1.0 - corr_weight
        ) * np.eye(dimension)
        scale_multiplier = 1.0 + severity if shift_mechanism != "correlation" else 1.0
        covariance = correlation * np.outer(scale, scale) * scale_multiplier**2
        if regime == "heavy_tail":
            shock = rng.multivariate_normal(np.zeros(dimension), covariance)
            shock *= np.sqrt(5.0 / rng.chisquare(5.0))
        else:
            shock = rng.multivariate_normal(np.zeros(dimension), covariance)
        if regime == "heteroskedastic":
            conditional_scale = np.sqrt(0.05 + 0.9 * conditional_scale**2 + 0.05 * values[index - 1] ** 2)
            shock *= conditional_scale
        mean = transition @ values[index - 1]
        if regime == "nonlinear":
            mean += 0.12 * np.tanh(values[index - 1][::-1])
        values[index] = mean + shock
    return values
