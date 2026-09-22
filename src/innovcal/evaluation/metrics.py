"""Sample-based multivariate forecast evaluation."""

import numpy as np
from scipy import stats


def _validate(target: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if target.ndim != 2 or samples.ndim != 3:
        raise ValueError("target and samples require shapes (cases, d) and (draws, cases, d)")
    if samples.shape[1:] != target.shape or not np.isfinite(samples).all():
        raise ValueError("forecast samples do not match finite targets")
    return target, samples


def empirical_pits(
    target: np.ndarray,
    samples: np.ndarray,
    projections: np.ndarray | None = None,
) -> np.ndarray:
    """Compute randomized-rank PITs after optional linear projection."""
    target, samples = _validate(target, samples)
    if projections is None:
        projections = np.eye(target.shape[1])
    projections = np.asarray(projections, dtype=float)
    if projections.ndim != 2 or projections.shape[1] != target.shape[1]:
        raise ValueError("projections must have shape (n_projections, dimension)")
    projected_target = np.einsum("bd,rd->br", target, projections)
    projected_samples = np.einsum("sbd,rd->sbr", samples, projections)
    return (
        (projected_samples < projected_target[None]).sum(axis=0) + 0.5
    ) / (samples.shape[0] + 1.0)


def gaussian_projected_pits(
    target: np.ndarray,
    mean: np.ndarray,
    scale_tril: np.ndarray,
    projections: np.ndarray,
) -> np.ndarray:
    """Exact PITs for Gaussian linear projections, without forecast-draw noise."""
    target = np.asarray(target, dtype=float)
    mean = np.asarray(mean, dtype=float)
    scale_tril = np.asarray(scale_tril, dtype=float)
    projections = np.asarray(projections, dtype=float)
    if (
        target.ndim != 2
        or mean.shape != target.shape
        or scale_tril.shape != (len(target), target.shape[1], target.shape[1])
        or projections.ndim != 2
        or projections.shape[1] != target.shape[1]
    ):
        raise ValueError("incompatible Gaussian forecast and projection shapes")
    if not all(np.isfinite(array).all() for array in (target, mean, scale_tril, projections)):
        raise ValueError("Gaussian forecast and projections must be finite")
    projected_scale = np.einsum("odk,rd->ork", scale_tril, projections)
    standard_deviation = np.linalg.norm(projected_scale, axis=-1)
    if np.any(standard_deviation <= 0):
        raise ValueError("projected standard deviations must be positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        z = ((target - mean) @ projections.T) / standard_deviation
    if np.isnan(z).any():
        raise ValueError("Gaussian projected standardized values contain NaN")
    return stats.norm.cdf(z)


def pit_diagnostics(pits: np.ndarray, max_lag: int = 5) -> dict[str, float]:
    pits = np.asarray(pits, dtype=float)
    if pits.ndim != 2:
        raise ValueError("pits must have shape (cases, projections)")
    grid = np.linspace(0.05, 0.95, 19)
    cdf = np.mean(pits[:, None, :] <= grid[None, :, None], axis=0)
    calibration_error = float(np.mean((cdf - grid[:, None]) ** 2))
    centered = pits - 0.5
    autocorrelations = []
    for lag in range(1, min(max_lag, len(pits) - 1) + 1):
        numerator = np.mean(centered[lag:] * centered[:-lag], axis=0)
        denominator = np.mean(centered**2, axis=0)
        autocorrelations.extend(np.abs(numerator / np.maximum(denominator, 1e-12)))
    ks = [
        stats.kstest(pits[:, index], "uniform", method="asymp").statistic
        for index in range(pits.shape[1])
    ]
    return {
        "pit_calibration_error": calibration_error,
        "pit_mean_absolute_autocorrelation": float(np.mean(autocorrelations)) if autocorrelations else 0.0,
        "pit_mean_ks": float(np.mean(ks)),
    }


def energy_score(target: np.ndarray, samples: np.ndarray, seed: int = 123) -> float:
    target, samples = _validate(target, samples)
    first = np.linalg.norm(samples - target[None], axis=-1).mean(axis=0)
    rng = np.random.default_rng(seed)
    paired = samples[rng.permutation(len(samples))]
    second = np.linalg.norm(samples - paired, axis=-1).mean(axis=0)
    return float(np.mean(first - 0.5 * second))


def variogram_score(
    target: np.ndarray, samples: np.ndarray, power: float = 0.5
) -> float:
    """Mean normalized pairwise variogram score; lower is better."""
    target, samples = _validate(target, samples)
    if power <= 0 or target.shape[1] < 2:
        raise ValueError("power must be positive and at least two assets are required")
    left, right = np.triu_indices(target.shape[1], k=1)
    observed = np.abs(target[:, left] - target[:, right]) ** power
    predicted = np.mean(np.abs(samples[:, :, left] - samples[:, :, right]) ** power, axis=0)
    return float(np.mean((observed - predicted) ** 2))


def evaluate_samples(
    target: np.ndarray,
    samples: np.ndarray,
    interval_level: float = 0.9,
    projections: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate marginal calibration, sharpness, RMSE, and Energy Score."""
    target, samples = _validate(target, samples)
    alpha = 1.0 - interval_level
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    below, above = target < lower, target > upper
    interval_score = (upper - lower) + (2.0 / alpha) * (
        (lower - target) * below + (target - upper) * above
    )
    coordinate_pits = empirical_pits(target, samples)
    projected_pits = empirical_pits(target, samples, projections)
    result = {
        "mean_squared_error": float(np.mean((samples.mean(axis=0) - target) ** 2)),
        "rmse": float(np.sqrt(np.mean((samples.mean(axis=0) - target) ** 2))),
        "energy_score": energy_score(target, samples),
        "variogram_score": variogram_score(target, samples),
        "coverage": float(np.mean((target >= lower) & (target <= upper))),
        "interval_width": float(np.mean(upper - lower)),
        "interval_score": float(np.mean(interval_score)),
    }
    result.update(pit_diagnostics(projected_pits))
    coordinate = pit_diagnostics(coordinate_pits)
    result.update({f"coordinate_{name}": value for name, value in coordinate.items()})
    return result
