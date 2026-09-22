"""Dependence-aware paired uncertainty for forecast comparisons."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from innovcal.evaluation.metrics import _validate, empirical_pits


def _origin_scores(
    target: np.ndarray, samples: np.ndarray, interval_level: float = 0.9
) -> dict[str, np.ndarray]:
    target, samples = _validate(target, samples)
    alpha = 1.0 - interval_level
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    interval = (upper - lower) + (2.0 / alpha) * (
        (lower - target) * (target < lower) + (target - upper) * (target > upper)
    )
    first = np.linalg.norm(samples - target[None], axis=-1).mean(axis=0)
    permutation = np.random.default_rng(123).permutation(len(samples))
    second = np.linalg.norm(samples - samples[permutation], axis=-1).mean(axis=0)
    left, right = np.triu_indices(target.shape[1], k=1)
    variogram = np.mean(
        (
            np.abs(target[:, left] - target[:, right]) ** 0.5
            - np.mean(np.abs(samples[:, :, left] - samples[:, :, right]) ** 0.5, axis=0)
        ) ** 2,
        axis=1,
    )
    return {
        "mean_squared_error": np.mean((samples.mean(axis=0) - target) ** 2, axis=1),
        "energy_score": first - 0.5 * second,
        "variogram_score": variogram,
        "interval_score": interval.mean(axis=1),
        "interval_width": (upper - lower).mean(axis=1),
        "coverage": ((target >= lower) & (target <= upper)).mean(axis=1),
    }


def _calibration_error(pits: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 19)
    empirical = np.mean(pits[:, None, :] <= grid[None, :, None], axis=0)
    return float(np.mean((empirical - grid[:, None]) ** 2))


def _pit_dependence(pits: np.ndarray, max_lag: int = 5) -> float:
    centered = pits - 0.5
    variance = np.mean(centered**2, axis=0)
    values = []
    for lag in range(1, min(max_lag, len(pits) - 1) + 1):
        covariance = np.mean(centered[lag:] * centered[:-lag], axis=0)
        values.extend(np.abs(covariance / np.maximum(variance, 1e-12)))
    return float(np.mean(values)) if values else 0.0


def moving_block_indices(
    n_observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a non-circular moving-block bootstrap index vector."""
    if not 1 <= block_length <= n_observations:
        raise ValueError("block_length must lie between one and the sample size")
    n_blocks = int(np.ceil(n_observations / block_length))
    starts = rng.integers(0, n_observations - block_length + 1, size=n_blocks)
    indices = np.concatenate(
        [np.arange(start, start + block_length) for start in starts]
    )
    return indices[:n_observations]


def paired_block_bootstrap(
    target: np.ndarray,
    forecast_samples: dict[str, np.ndarray],
    comparisons: Sequence[tuple[str, str]],
    projections: np.ndarray,
    block_length: int = 20,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 707,
    interval_level: float = 0.9,
) -> pd.DataFrame:
    """Estimate paired differences and moving-block bootstrap uncertainty.

    Differences are candidate minus comparator. Negative values favor the
    candidate for loss metrics. Raw coverage and width remain descriptive;
    ``coverage_error`` measures absolute deviation from nominal coverage.
    """
    if n_bootstrap < 2 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    target = np.asarray(target, dtype=float)
    missing = {
        name for pair in comparisons for name in pair if name not in forecast_samples
    }
    if missing:
        raise ValueError(f"missing forecasts for models: {sorted(missing)}")
    score_cache = {
        name: _origin_scores(target, samples)
        for name, samples in forecast_samples.items()
    }
    pit_cache = {
        name: empirical_pits(target, samples, projections)
        for name, samples in forecast_samples.items()
    }
    rng = np.random.default_rng(seed)
    bootstrap_indices = [
        moving_block_indices(len(target), block_length, rng) for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - confidence
    rows = []
    for candidate, comparator in comparisons:
        additive = {
            metric: score_cache[candidate][metric] - score_cache[comparator][metric]
            for metric in score_cache[candidate]
        }
        for metric, differences in additive.items():
            draws = np.array([differences[index].mean() for index in bootstrap_indices])
            rows.append(
                _summary_row(candidate, comparator, metric, differences.mean(), draws, alpha)
            )
        candidate_coverage = score_cache[candidate]["coverage"]
        comparator_coverage = score_cache[comparator]["coverage"]
        observed = abs(candidate_coverage.mean() - interval_level) - abs(
            comparator_coverage.mean() - interval_level
        )
        draws = np.array(
            [
                abs(candidate_coverage[index].mean() - interval_level)
                - abs(comparator_coverage[index].mean() - interval_level)
                for index in bootstrap_indices
            ]
        )
        rows.append(
            _summary_row(
                candidate, comparator, "coverage_error", observed, draws, alpha
            )
        )
        for metric, function in (
            ("pit_calibration_error", _calibration_error),
            ("pit_mean_absolute_autocorrelation", _pit_dependence),
        ):
            observed = function(pit_cache[candidate]) - function(pit_cache[comparator])
            draws = np.array(
                [
                    function(pit_cache[candidate][index])
                    - function(pit_cache[comparator][index])
                    for index in bootstrap_indices
                ]
            )
            rows.append(
                _summary_row(candidate, comparator, metric, observed, draws, alpha)
            )
    frame = pd.DataFrame(rows)
    frame.insert(0, "n_origins", len(target))
    frame.insert(1, "block_length", block_length)
    frame.insert(2, "n_bootstrap", n_bootstrap)
    return frame


def paired_seed_block_bootstrap(
    target: np.ndarray,
    forecast_samples: dict[str, np.ndarray],
    comparisons: Sequence[tuple[str, str]],
    projections: np.ndarray,
    block_length: int = 20,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 808,
    interval_level: float = 0.9,
) -> pd.DataFrame:
    """Paired hierarchical bootstrap over training seeds and forecast origins.

    Forecast arrays have shape ``(seed, draw, origin, dimension)``. Each draw
    resamples training seeds and, within each selected seed, a moving block of
    matched forecast origins. The estimand is the mean paired effect over the
    supplied optimization-seed distribution and observed test period.
    """
    target = np.asarray(target, dtype=float)
    arrays = {name: np.asarray(value, dtype=float) for name, value in forecast_samples.items()}
    missing = {name for pair in comparisons for name in pair if name not in arrays}
    if missing:
        raise ValueError(f"missing forecasts for models: {sorted(missing)}")
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all forecast arrays must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 4 or shape[2:] != target.shape:
        raise ValueError("forecasts must have shape (seed, draw, origin, dimension)")
    n_seeds = shape[0]
    if n_seeds < 2:
        raise ValueError("at least two training seeds are required")
    score_cache = {
        name: [_origin_scores(target, value[index], interval_level) for index in range(n_seeds)]
        for name, value in arrays.items()
    }
    pit_cache = {
        name: [empirical_pits(target, value[index], projections) for index in range(n_seeds)]
        for name, value in arrays.items()
    }
    rng = np.random.default_rng(seed)
    seed_indices = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    block_indices = [
        [moving_block_indices(len(target), block_length, rng) for _ in range(n_seeds)]
        for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - confidence
    rows = []
    for candidate, comparator in comparisons:
        for metric in score_cache[candidate][0]:
            per_seed = np.array(
                [
                    (score_cache[candidate][s][metric] - score_cache[comparator][s][metric]).mean()
                    for s in range(n_seeds)
                ]
            )
            draws = np.array(
                [
                    np.mean(
                        [
                            (
                                score_cache[candidate][s][metric][index]
                                - score_cache[comparator][s][metric][index]
                            ).mean()
                            for s, index in zip(seed_indices[b], block_indices[b], strict=True)
                        ]
                    )
                    for b in range(n_bootstrap)
                ]
            )
            rows.append(_summary_row(candidate, comparator, metric, per_seed.mean(), draws, alpha))
        for metric, function in (
            ("coverage_error", lambda x: abs(x.mean() - interval_level)),
            ("pit_calibration_error", _calibration_error),
            ("pit_mean_absolute_autocorrelation", _pit_dependence),
        ):
            cache = score_cache if metric == "coverage_error" else pit_cache
            key = "coverage" if metric == "coverage_error" else None

            def evaluate(
                name: str,
                s: int,
                index: np.ndarray | None = None,
                *,
                selected_cache=cache,
                selected_key=key,
                selected_function=function,
            ) -> float:
                values = (
                    selected_cache[name][s][selected_key]
                    if selected_key
                    else selected_cache[name][s]
                )
                return selected_function(values if index is None else values[index])

            observed = np.mean(
                [evaluate(candidate, s) - evaluate(comparator, s) for s in range(n_seeds)]
            )
            draws = np.array(
                [
                    np.mean(
                        [
                            evaluate(candidate, s, index) - evaluate(comparator, s, index)
                            for s, index in zip(seed_indices[b], block_indices[b], strict=True)
                        ]
                    )
                    for b in range(n_bootstrap)
                ]
            )
            rows.append(_summary_row(candidate, comparator, metric, observed, draws, alpha))
    frame = pd.DataFrame(rows)
    frame.insert(0, "n_seeds", n_seeds)
    frame.insert(1, "n_origins", len(target))
    frame.insert(2, "block_length", block_length)
    frame.insert(3, "n_bootstrap", n_bootstrap)
    return frame


def _summary_row(
    candidate: str,
    comparator: str,
    metric: str,
    observed: float,
    draws: np.ndarray,
    alpha: float,
) -> dict[str, float | str]:
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "candidate": candidate,
        "comparator": comparator,
        "metric": metric,
        "difference": float(observed),
        "bootstrap_se": float(draws.std(ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "probability_difference_below_zero": float(np.mean(draws < 0)),
    }
