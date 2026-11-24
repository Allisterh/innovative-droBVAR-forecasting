"""Dependence-aware paired uncertainty for forecast comparisons."""

from collections.abc import Mapping, Sequence

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


def _pit_dependence(
    pits: np.ndarray,
    max_lag: int = 5,
) -> float:
    """Mean absolute normalized half-centered PIT moment through ``max_lag``."""
    contributions = _pit_dependence_contributions(pits, max_lag=max_lag)
    return _pit_dependence_from_contributions(contributions)


def _pit_dependence_contributions(
    pits: np.ndarray,
    max_lag: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original-series variance and lag-product contributions.

    Lag products are constructed before resampling. A moving-block bootstrap
    can therefore resample these contribution rows without ever interpreting
    the join between two sampled blocks as an observed temporal adjacency.
    """
    pits = np.asarray(pits, dtype=float)
    if pits.ndim != 2:
        raise ValueError("pits must have shape (origin, projection)")
    centered = pits - 0.5
    n_origins, n_projections = centered.shape
    effective_max_lag = min(max_lag, n_origins - 1)
    lag_products = np.full(
        (n_origins, effective_max_lag, n_projections), np.nan, dtype=float
    )
    for lag in range(1, effective_max_lag + 1):
        lag_products[lag:, lag - 1, :] = centered[lag:] * centered[:-lag]
    return centered**2, lag_products


def _pit_dependence_from_contributions(
    contributions: tuple[np.ndarray, np.ndarray],
    indices: np.ndarray | None = None,
) -> float:
    """Evaluate the normalized lagged PIT moment from original-series products."""
    variance_terms, lag_products = contributions
    if indices is not None:
        indices = np.asarray(indices, dtype=int)
        variance_terms = variance_terms[indices]
        lag_products = lag_products[indices]
    variance = np.mean(variance_terms, axis=0)
    with np.errstate(invalid="ignore"):
        covariance = np.nanmean(lag_products, axis=0)
    ratios = np.abs(covariance / np.maximum(variance[None, :], 1e-12))
    finite = np.isfinite(ratios)
    return float(np.mean(ratios[finite])) if np.any(finite) else 0.0


def moving_block_indices(
    n_observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a non-circular moving-block bootstrap index vector."""
    indices, _ = moving_block_resample(n_observations, block_length, rng)
    return indices


def moving_block_resample(
    n_observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw moving-block indices and labels identifying sampled block joins."""
    if not 1 <= block_length <= n_observations:
        raise ValueError("block_length must lie between one and the sample size")
    n_blocks = int(np.ceil(n_observations / block_length))
    starts = rng.integers(0, n_observations - block_length + 1, size=n_blocks)
    blocks = [np.arange(start, start + block_length) for start in starts]
    indices = np.concatenate(blocks)[:n_observations]
    block_ids = np.repeat(np.arange(n_blocks), block_length)[:n_observations]
    return indices, block_ids


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
    simultaneous_metrics: Sequence[str] | None = None,
    familywise_confidence: float | None = None,
) -> pd.DataFrame:
    """Estimate paired differences and moving-block bootstrap uncertainty.

    Differences are candidate minus comparator. Negative values favor the
    candidate for loss metrics. Raw coverage and width remain descriptive;
    ``coverage_error`` measures absolute deviation from nominal coverage.
    Optional max-|z| intervals control family-wise error across the declared
    metrics separately within each candidate--comparator pair.
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
        name: _origin_scores(target, samples, interval_level)
        for name, samples in forecast_samples.items()
    }
    pit_cache = {
        name: empirical_pits(target, samples, projections)
        for name, samples in forecast_samples.items()
    }
    rng = np.random.default_rng(seed)
    bootstrap_indices = [
        moving_block_indices(len(target), block_length, rng)
        for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - confidence
    rows = []
    replicate_draws: list[np.ndarray] = []
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
            replicate_draws.append(draws)
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
        replicate_draws.append(draws)
        for metric in ("pit_calibration_error", "pit_mean_absolute_autocorrelation"):
            if metric == "pit_calibration_error":
                candidate_values = pit_cache[candidate]
                comparator_values = pit_cache[comparator]
                function = _calibration_error
            else:
                candidate_values = _pit_dependence_contributions(pit_cache[candidate])
                comparator_values = _pit_dependence_contributions(pit_cache[comparator])
                function = _pit_dependence_from_contributions
            observed = function(candidate_values) - function(comparator_values)
            if metric == "pit_calibration_error":
                draws = np.array(
                    [
                        function(candidate_values[index])
                        - function(comparator_values[index])
                        for index in bootstrap_indices
                    ]
                )
            else:
                draws = np.array(
                    [
                        function(candidate_values, index)
                        - function(comparator_values, index)
                        for index in bootstrap_indices
                    ]
                )
            rows.append(
                _summary_row(candidate, comparator, metric, observed, draws, alpha)
            )
            replicate_draws.append(draws)
    frame = pd.DataFrame(rows)
    frame.insert(0, "n_origins", len(target))
    frame.insert(1, "block_length", block_length)
    frame.insert(2, "n_bootstrap", n_bootstrap)
    if simultaneous_metrics is not None:
        frame = _add_simultaneous_intervals(
            frame,
            replicate_draws,
            simultaneous_metrics,
            confidence if familywise_confidence is None else familywise_confidence,
        )
    return frame


def _paired_seed_block_bootstrap_core(
    target: np.ndarray,
    forecast_samples: dict[str, np.ndarray],
    comparisons: Sequence[tuple[str, str]],
    projections: np.ndarray,
    block_length: int = 20,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 808,
    interval_level: float = 0.9,
    simultaneous_metrics: Sequence[str] | None = None,
    familywise_confidence: float | None = None,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Paired hierarchical bootstrap over training seeds and forecast origins.

    Forecast arrays have shape ``(seed, draw, origin, dimension)``. Each draw
    resamples paired training seeds and one shared moving-block date sequence
    per bootstrap replicate. The shared date sequence is applied to every
    selected seed and every method because all fits forecast the same market
    dates. The estimand is the mean paired effect over the
    supplied optimization-seed distribution and observed test period.

    If ``simultaneous_metrics`` is supplied, max-|z| bootstrap intervals are
    added for that metric family. Family-wise error is controlled separately
    within each candidate--comparator pair; the ordinary percentile intervals
    remain available in ``ci_lower`` and ``ci_upper``.
    """
    target = np.asarray(target, dtype=float)
    arrays = {name: np.asarray(value) for name, value in forecast_samples.items()}
    missing = {name for pair in comparisons for name in pair if name not in arrays}
    if missing:
        raise ValueError(f"missing forecasts for models: {sorted(missing)}")
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all forecast arrays must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 4 or shape[2:] != target.shape:
        raise ValueError("forecasts must have shape (seed, draw, origin, dimension)")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("forecast arrays must be finite")
    n_seeds = shape[0]
    if n_seeds < 1:
        raise ValueError("at least one fitted model is required")
    def per_seed_cache(value: np.ndarray, function) -> list:
        first = function(value[0])
        if value.strides[0] == 0:
            return [first] * n_seeds
        return [first, *[function(value[index]) for index in range(1, n_seeds)]]

    score_cache = {
        name: per_seed_cache(
            value, lambda samples: _origin_scores(target, samples, interval_level)
        )
        for name, value in arrays.items()
    }
    pit_cache = {
        name: per_seed_cache(
            value, lambda samples: empirical_pits(target, samples, projections)
        )
        for name, value in arrays.items()
    }
    pit_dependence_cache = {
        name: [_pit_dependence_contributions(value) for value in values]
        for name, values in pit_cache.items()
    }
    rng = np.random.default_rng(seed)
    seed_indices = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    block_resamples = [
        moving_block_resample(len(target), block_length, rng)
        for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - confidence
    rows = []
    replicate_draws: list[np.ndarray] = []
    nonlinear_draw_cache: dict[tuple[str, str], np.ndarray] = {}

    def nonlinear_draws(name: str, metric: str) -> np.ndarray:
        """Evaluate one nonlinear diagnostic once per model, seed, and draw."""
        cache_key = (name, metric)
        if cache_key in nonlinear_draw_cache:
            return nonlinear_draw_cache[cache_key]
        if metric == "coverage_error":
            values = [item["coverage"] for item in score_cache[name]]

            def function(value, index):
                return abs(value[index].mean() - interval_level)

        elif metric == "pit_calibration_error":
            values = pit_cache[name]

            def function(value, index):
                return _calibration_error(value[index])

        else:
            values = pit_dependence_cache[name]
            function = _pit_dependence_from_contributions
        unique_seed_count = 1 if arrays[name].strides[0] == 0 else n_seeds
        per_actual_seed = np.empty((n_bootstrap, unique_seed_count), dtype=float)
        for b, (index, _) in enumerate(block_resamples):
            for s in range(unique_seed_count):
                per_actual_seed[b, s] = function(values[s], index)
        if unique_seed_count == 1:
            selected = np.repeat(per_actual_seed, n_seeds, axis=1)
        else:
            selected = np.take_along_axis(per_actual_seed, seed_indices, axis=1)
        nonlinear_draw_cache[cache_key] = selected.mean(axis=1)
        return nonlinear_draw_cache[cache_key]

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
                                score_cache[candidate][s][metric][block_resamples[b][0]]
                                - score_cache[comparator][s][metric][block_resamples[b][0]]
                            ).mean()
                            for s in seed_indices[b]
                        ]
                    )
                    for b in range(n_bootstrap)
                ]
            )
            rows.append(
                _summary_row(
                    candidate, comparator, metric, per_seed.mean(), draws, alpha
                )
            )
            replicate_draws.append(draws)
        for metric, function in (
            ("coverage_error", lambda x: abs(x.mean() - interval_level)),
            ("pit_calibration_error", _calibration_error),
            ("pit_mean_absolute_autocorrelation", _pit_dependence_from_contributions),
        ):
            cache = (
                score_cache
                if metric == "coverage_error"
                else pit_dependence_cache
                if metric == "pit_mean_absolute_autocorrelation"
                else pit_cache
            )
            key = "coverage" if metric == "coverage_error" else None

            def evaluate(
                name: str,
                s: int,
                index: np.ndarray | None = None,
                *,
                selected_cache=cache,
                selected_key=key,
                selected_function=function,
                selected_metric=metric,
            ) -> float:
                values = (
                    selected_cache[name][s][selected_key]
                    if selected_key
                    else selected_cache[name][s]
                )
                if selected_metric == "pit_mean_absolute_autocorrelation":
                    return selected_function(values, index)
                return selected_function(values if index is None else values[index])

            observed = np.mean(
                [evaluate(candidate, s) - evaluate(comparator, s) for s in range(n_seeds)]
            )
            draws = nonlinear_draws(candidate, metric) - nonlinear_draws(
                comparator, metric
            )
            rows.append(_summary_row(candidate, comparator, metric, observed, draws, alpha))
            replicate_draws.append(draws)
    frame = pd.DataFrame(rows)
    frame.insert(0, "n_seeds", n_seeds)
    frame.insert(1, "n_origins", len(target))
    frame.insert(2, "block_length", block_length)
    frame.insert(3, "n_bootstrap", n_bootstrap)
    if simultaneous_metrics is not None:
        frame = _add_simultaneous_intervals(
            frame,
            replicate_draws,
            simultaneous_metrics,
            confidence if familywise_confidence is None else familywise_confidence,
        )
    return frame, replicate_draws


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
    simultaneous_metrics: Sequence[str] | None = None,
    familywise_confidence: float | None = None,
) -> pd.DataFrame:
    """Return synchronized seed/date bootstrap intervals.

    See :func:`_paired_seed_block_bootstrap_core` for the resampling design.
    Simultaneous max-|z| intervals are optional and are formed separately
    within each candidate--comparator pair.
    """
    frame, _ = _paired_seed_block_bootstrap_core(
        target,
        forecast_samples,
        comparisons,
        projections,
        block_length,
        n_bootstrap,
        confidence,
        seed,
        interval_level,
        simultaneous_metrics,
        familywise_confidence,
    )
    return frame


def _add_simultaneous_intervals(
    frame: pd.DataFrame,
    replicate_draws: Sequence[np.ndarray],
    metrics: Sequence[str],
    confidence: float,
    column_prefix: str = "familywise_",
) -> pd.DataFrame:
    """Attach max-|z| intervals for a declared within-comparison family."""
    selected_metrics = tuple(metrics)
    if not selected_metrics or len(set(selected_metrics)) != len(selected_metrics):
        raise ValueError("simultaneous_metrics must contain distinct metric names")
    if not 0 < confidence < 1:
        raise ValueError("familywise_confidence must lie between zero and one")
    unknown = set(selected_metrics) - set(frame["metric"])
    if unknown:
        raise ValueError(f"unknown simultaneous metrics: {sorted(unknown)}")
    if len(frame) != len(replicate_draws):
        raise ValueError("one bootstrap replicate vector is required per result row")

    output = frame.copy()
    columns = {
        name: f"{column_prefix}{name}"
        for name in (
            "ci_lower",
            "ci_upper",
            "critical_value",
            "family_size",
            "confidence",
            "resolved",
        )
    }
    for name in ("ci_lower", "ci_upper", "critical_value", "family_size", "confidence"):
        output[columns[name]] = np.nan
    output[columns["resolved"]] = pd.array([pd.NA] * len(output), dtype="boolean")

    selected = set(selected_metrics)
    for _, group in output.groupby(["candidate", "comparator"], sort=False):
        positions = [
            output.index.get_loc(index)
            for index in group.index
            if output.loc[index, "metric"] in selected
        ]
        if not positions:
            continue
        matrix = np.column_stack([replicate_draws[position] for position in positions])
        observed = output.iloc[positions]["difference"].to_numpy(dtype=float)
        standard_errors = matrix.std(axis=0, ddof=1)
        centered = matrix - observed[None, :]
        standardized = np.zeros_like(centered)
        nondegenerate = standard_errors > np.finfo(float).eps
        standardized[:, nondegenerate] = (
            centered[:, nondegenerate] / standard_errors[None, nondegenerate]
        )
        maximum = np.max(np.abs(standardized), axis=1)
        critical_value = float(np.quantile(maximum, confidence))
        lower = observed - critical_value * standard_errors
        upper = observed + critical_value * standard_errors
        indices = output.index[positions]
        output.loc[indices, columns["ci_lower"]] = lower
        output.loc[indices, columns["ci_upper"]] = upper
        output.loc[indices, columns["critical_value"]] = critical_value
        output.loc[indices, columns["family_size"]] = len(positions)
        output.loc[indices, columns["confidence"]] = confidence
        output.loc[indices, columns["resolved"]] = (lower > 0) | (upper < 0)
    return output


def combined_seed_draw_bootstrap(
    target: np.ndarray,
    forecast_samples: dict[str, np.ndarray],
    comparisons: Sequence[tuple[str, str]],
    projections: np.ndarray,
    *,
    block_length: int = 20,
    n_bootstrap: int = 2000,
    n_draw_resamples: int = 200,
    confidence: float = 0.95,
    seed: int = 808,
    draw_seed: int = 1607,
    interval_level: float = 0.9,
    draw_pairing: str | Mapping[tuple[str, str], str] = "paired",
    mc_metrics: Sequence[str] | None = None,
    simultaneous_metrics: Sequence[str] | None = None,
    familywise_confidence: float | None = None,
) -> pd.DataFrame:
    """Propagate seed/date and predictive-draw Monte Carlo uncertainty.

    The synchronized hierarchical bootstrap estimates optimization-seed and
    forecast-origin uncertainty using all retained predictive draws. A second
    nonparametric bootstrap resamples the predictive-draw axis. Its centered
    deviations are convolved with the seed/date replicates under a documented
    independence assumption. This avoids an expensive fully nested bootstrap
    while adding the predictive-integration variance component to the main
    intervals.

    ``draw_pairing`` may be ``"paired"`` for matched models trained and sampled
    with common random numbers, ``"independent"`` for unrelated methods, or a
    mapping from each comparison tuple to one of those modes. Existing
    seed/date intervals remain in ``ci_lower`` and ``ci_upper``; propagated
    results use the ``combined_`` prefix.
    """
    if n_draw_resamples < 2:
        raise ValueError("n_draw_resamples must be at least two")
    arrays = _normalize_seed_draw_arrays(target, forecast_samples)
    frame, seed_date_replicates = _paired_seed_block_bootstrap_core(
        target,
        arrays,
        comparisons,
        projections,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        interval_level=interval_level,
        simultaneous_metrics=None,
        familywise_confidence=None,
    )
    available_metrics = tuple(frame["metric"].drop_duplicates())
    metric_names = available_metrics if mc_metrics is None else tuple(mc_metrics)
    if not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError("mc_metrics must contain distinct metric names")
    unknown = set(metric_names) - set(available_metrics)
    if unknown:
        raise ValueError(f"unknown Monte Carlo metrics: {sorted(unknown)}")
    if simultaneous_metrics is not None:
        missing = set(simultaneous_metrics) - set(metric_names)
        if missing:
            raise ValueError(
                "simultaneous metrics require Monte Carlo propagation: "
                f"{sorted(missing)}"
            )
    pairing = _resolve_draw_pairing(comparisons, draw_pairing)
    full_mc, mc_replicates = _draw_mc_replicates(
        np.asarray(target, dtype=float),
        arrays,
        comparisons,
        np.asarray(projections, dtype=float),
        metric_names,
        interval_level,
        n_draw_resamples,
        np.random.default_rng(draw_seed),
        pairing,
    )

    output = frame.copy()
    numeric_columns = (
        "draw_mc_bias",
        "draw_mc_se",
        "variance_component_se",
        "combined_bootstrap_se",
        "combined_ci_lower",
        "combined_ci_upper",
        "combined_probability_difference_below_zero",
        "draw_mc_variance_fraction",
    )
    for column in numeric_columns:
        output[column] = np.nan
    output["n_draw_resamples"] = np.nan
    output["draw_resampling"] = pd.NA
    output["combined_interval_method"] = pd.NA

    convolution_rng = np.random.default_rng(draw_seed + 1)
    alpha = 1.0 - confidence
    combined_replicates: list[np.ndarray] = []
    # Preserve cross-metric dependence in each simultaneous family by using
    # one common sequence of draw-bootstrap replicate labels per comparison.
    convolution_indices = {
        comparison: convolution_rng.integers(
            0, n_draw_resamples, size=len(seed_date_replicates[0])
        )
        for comparison in comparisons
    }
    for position, row in output.iterrows():
        key = (row["candidate"], row["comparator"], row["metric"])
        comparison = (row["candidate"], row["comparator"])
        date_values = seed_date_replicates[position]
        if row["metric"] not in metric_names:
            combined_replicates.append(date_values)
            continue
        mc_values = mc_replicates[key]
        centered_mc = mc_values - mc_values.mean()
        selected = convolution_indices[comparison]
        combined = date_values + centered_mc[selected]
        combined_replicates.append(combined)
        mc_se = float(mc_values.std(ddof=1))
        seed_date_se = float(row["bootstrap_se"])
        variance_component_se = float(np.hypot(seed_date_se, mc_se))
        combined_se = float(combined.std(ddof=1))
        lower, upper = np.quantile(combined, [alpha / 2.0, 1.0 - alpha / 2.0])
        total_variance = seed_date_se**2 + mc_se**2
        output.loc[position, "draw_mc_bias"] = float(mc_values.mean() - full_mc[key])
        output.loc[position, "draw_mc_se"] = mc_se
        output.loc[position, "variance_component_se"] = variance_component_se
        output.loc[position, "combined_bootstrap_se"] = combined_se
        output.loc[position, "combined_ci_lower"] = lower
        output.loc[position, "combined_ci_upper"] = upper
        output.loc[position, "combined_probability_difference_below_zero"] = np.mean(
            combined < 0
        )
        output.loc[position, "draw_mc_variance_fraction"] = (
            mc_se**2 / total_variance if total_variance > 0 else 0.0
        )
        output.loc[position, "n_draw_resamples"] = n_draw_resamples
        output.loc[position, "draw_resampling"] = pairing[
            (row["candidate"], row["comparator"])
        ]
        output.loc[position, "combined_interval_method"] = (
            "independence_convolution"
        )

    if simultaneous_metrics is not None:
        output = _add_simultaneous_intervals(
            output,
            combined_replicates,
            simultaneous_metrics,
            confidence if familywise_confidence is None else familywise_confidence,
            column_prefix="combined_familywise_",
        )
    return output


def _normalize_seed_draw_arrays(
    target: np.ndarray,
    forecast_samples: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate and normalize retained forecasts to four dimensions."""
    target = np.asarray(target, dtype=float)
    arrays = {name: np.asarray(value, dtype=float) for name, value in forecast_samples.items()}
    for name, value in tuple(arrays.items()):
        if value.ndim == 3:
            arrays[name] = value[None, ...]
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all forecast arrays must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 4 or shape[2:] != target.shape:
        raise ValueError("forecasts must have shape (seed, draw, origin, dimension)")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("forecast arrays must be finite")
    return arrays


def _resolve_draw_pairing(
    comparisons: Sequence[tuple[str, str]],
    draw_pairing: str | Mapping[tuple[str, str], str],
) -> dict[tuple[str, str], str]:
    """Return one validated predictive-draw resampling mode per comparison."""
    allowed = {"paired", "independent"}
    if isinstance(draw_pairing, str):
        modes = {comparison: draw_pairing for comparison in comparisons}
    else:
        missing = set(comparisons) - set(draw_pairing)
        if missing:
            raise ValueError(f"missing draw-pairing modes: {sorted(missing)}")
        modes = {comparison: draw_pairing[comparison] for comparison in comparisons}
    invalid = {mode for mode in modes.values() if mode not in allowed}
    if invalid:
        raise ValueError(f"draw-pairing modes must be in {sorted(allowed)}")
    return modes


def _draw_mc_replicates(
    target: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    comparisons: Sequence[tuple[str, str]],
    projections: np.ndarray,
    metrics: Sequence[str],
    interval_level: float,
    n_resamples: int,
    rng: np.random.Generator,
    pairing: Mapping[tuple[str, str], str],
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], np.ndarray],
]:
    """Return full-draw estimates and draw-bootstrap comparison replicates."""
    n_seeds, n_draws = next(iter(arrays.values())).shape[:2]
    seed_counts = {
        name: (
            1
            if all(np.array_equal(value[0], value[index]) for index in range(1, n_seeds))
            else n_seeds
        )
        for name, value in arrays.items()
    }

    def evaluate(name: str, indices: np.ndarray | None = None) -> dict[str, float]:
        values = []
        for seed_index in range(seed_counts[name]):
            samples = arrays[name][seed_index]
            if indices is not None:
                samples = samples[indices]
            values.append(
                _aggregate_sample_metrics(target, samples, projections, interval_level)
            )
        return {
            metric: float(np.mean([value[metric] for value in values]))
            for metric in metrics
        }

    required_models = tuple(dict.fromkeys(name for pair in comparisons for name in pair))
    full_by_model = {name: evaluate(name) for name in required_models}
    full = {
        (candidate, comparator, metric): full_by_model[candidate][metric]
        - full_by_model[comparator][metric]
        for candidate, comparator in comparisons
        for metric in metrics
    }
    replicates = {key: np.empty(n_resamples, dtype=float) for key in full}
    for candidate, comparator in comparisons:
        mode = pairing[(candidate, comparator)]
        for replicate in range(n_resamples):
            candidate_indices = rng.integers(0, n_draws, size=n_draws)
            comparator_indices = (
                candidate_indices
                if mode == "paired"
                else rng.integers(0, n_draws, size=n_draws)
            )
            candidate_values = evaluate(candidate, candidate_indices)
            comparator_values = evaluate(comparator, comparator_indices)
            for metric in metrics:
                replicates[(candidate, comparator, metric)][replicate] = (
                    candidate_values[metric] - comparator_values[metric]
                )
    return full, replicates


def paired_draw_mc_sensitivity(
    target: np.ndarray,
    forecast_samples: dict[str, np.ndarray],
    comparisons: Sequence[tuple[str, str]],
    projections: np.ndarray,
    *,
    draw_counts: Sequence[int] | None = None,
    n_resamples: int = 100,
    confidence: float = 0.95,
    seed: int = 1606,
    interval_level: float = 0.9,
    metrics: Sequence[str] = (
        "mean_squared_error",
        "energy_score",
        "variogram_score",
        "interval_score",
        "coverage_error",
        "pit_calibration_error",
        "pit_mean_absolute_autocorrelation",
    ),
) -> pd.DataFrame:
    """Estimate predictive-draw Monte Carlo error from retained forecasts.

    Forecast arrays may have shape ``(seed, draw, origin, dimension)`` or
    ``(draw, origin, dimension)``. Draw indices are resampled with replacement
    and shared across methods and optimization seeds, preserving the common
    predictive random numbers used by the matched neural experiments. The
    resulting intervals quantify numerical integration error conditional on
    the fitted models and observed targets; they are not sampling intervals for
    forecast performance. Use :func:`combined_seed_draw_bootstrap` when this
    component should be propagated into the main seed/date intervals.
    """
    if n_resamples < 2 or not 0 < confidence < 1:
        raise ValueError("invalid Monte Carlo resampling settings")
    target = np.asarray(target, dtype=float)
    arrays = {name: np.asarray(value, dtype=float) for name, value in forecast_samples.items()}
    missing = {name for pair in comparisons for name in pair if name not in arrays}
    if missing:
        raise ValueError(f"missing forecasts for models: {sorted(missing)}")
    if not comparisons:
        raise ValueError("at least one comparison is required")
    for name, value in tuple(arrays.items()):
        if value.ndim == 3:
            arrays[name] = value[None, ...]
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all forecast arrays must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 4 or shape[2:] != target.shape:
        raise ValueError("forecasts must have shape (seed, draw, origin, dimension)")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("forecast arrays must be finite")
    n_seeds, n_available = shape[:2]
    if draw_counts is None:
        draw_counts = tuple(sorted({min(100, n_available), min(250, n_available), n_available}))
    draw_counts = tuple(int(value) for value in draw_counts)
    if (
        not draw_counts
        or len(set(draw_counts)) != len(draw_counts)
        or min(draw_counts) < 2
        or max(draw_counts) > n_available
    ):
        raise ValueError("draw_counts must be distinct and lie between 2 and the retained draw count")

    metric_names = tuple(metrics)
    available_metrics = set(
        _aggregate_sample_metrics(
            target, next(iter(arrays.values()))[0], projections, interval_level
        )
    )
    if not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError("metrics must contain distinct names")
    unknown_metrics = set(metric_names) - available_metrics
    if unknown_metrics:
        raise ValueError(f"unknown Monte Carlo sensitivity metrics: {sorted(unknown_metrics)}")

    required_models = tuple(dict.fromkeys(name for pair in comparisons for name in pair))

    def evaluate_models(indices: np.ndarray | None = None) -> dict[str, dict[str, float]]:
        evaluated = {}
        for name in required_models:
            per_seed = []
            for seed_index in range(n_seeds):
                samples = arrays[name][seed_index]
                if indices is not None:
                    samples = samples[indices]
                per_seed.append(
                    _aggregate_sample_metrics(
                        target, samples, projections, interval_level
                    )
                )
            evaluated[name] = {
                metric: float(np.mean([value[metric] for value in per_seed]))
                for metric in metric_names
            }
        return evaluated

    full = evaluate_models()
    full_differences = {
        (candidate, comparator, metric): full[candidate][metric]
        - full[comparator][metric]
        for candidate, comparator in comparisons
        for metric in metric_names
    }
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    rows = []
    for draw_count in draw_counts:
        replicate_values = {
            key: np.empty(n_resamples, dtype=float) for key in full_differences
        }
        for replicate in range(n_resamples):
            indices = rng.integers(0, n_available, size=draw_count)
            evaluated = evaluate_models(indices)
            for candidate, comparator in comparisons:
                for metric in metric_names:
                    replicate_values[(candidate, comparator, metric)][replicate] = (
                        evaluated[candidate][metric] - evaluated[comparator][metric]
                    )
        for key, values in replicate_values.items():
            candidate, comparator, metric = key
            lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
            full_difference = full_differences[key]
            rows.append(
                {
                    "candidate": candidate,
                    "comparator": comparator,
                    "metric": metric,
                    "n_seeds": n_seeds,
                    "available_draws": n_available,
                    "draw_count": draw_count,
                    "n_resamples": n_resamples,
                    "full_draw_difference": full_difference,
                    "mean_resampled_difference": float(values.mean()),
                    "mc_bias": float(values.mean() - full_difference),
                    "mc_standard_error": float(values.std(ddof=1)),
                    "mc_ci_lower": float(lower),
                    "mc_ci_upper": float(upper),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_sample_metrics(
    target: np.ndarray,
    samples: np.ndarray,
    projections: np.ndarray,
    interval_level: float,
) -> dict[str, float]:
    """Aggregate the sample-based metrics used by uncertainty analyses."""
    scores = _origin_scores(target, samples, interval_level)
    pits = empirical_pits(target, samples, projections)
    result = {name: float(values.mean()) for name, values in scores.items()}
    result["coverage_error"] = abs(result["coverage"] - interval_level)
    result["pit_calibration_error"] = _calibration_error(pits)
    result["pit_mean_absolute_autocorrelation"] = _pit_dependence(pits)
    return result


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
