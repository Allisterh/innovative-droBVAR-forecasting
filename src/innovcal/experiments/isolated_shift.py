"""Replicated, frozen-model scale-only distribution-shift experiment."""

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from innovcal.ca_rnn import (
    CARNN,
    CARNNConfig,
    TrainingConfig,
    fit_model,
    predict_distribution,
    predictive_nll,
    sample_forecasts,
)
from innovcal.ca_rnn.losses import make_projection_matrix
from innovcal.data.simulation import simulate_multivariate_series
from innovcal.data.windows import (
    Standardizer,
    WindowDataset,
    chronological_split,
    partition_window_datasets,
)
from innovcal.evaluation import (
    evaluate_samples,
    gaussian_projected_pits,
    pit_diagnostics,
)


@dataclass(frozen=True)
class IsolatedScaleShiftConfig:
    n_observations: int = 1200
    dimension: int = 4
    history: int = 20
    regime: str = "gaussian"
    severities: tuple[float, ...] = (0.0, 0.5, 1.0)
    dgp_seeds: tuple[int, ...] = tuple(range(1000, 1050))
    neural_seed: int | None = None
    projection_seed: int = 606
    forecast_seed: int | None = None
    neural_seeds: tuple[int, ...] = (606, 1606, 2606)
    forecast_seeds: tuple[int, ...] = (20_606, 21_606, 22_606)
    n_forecast_samples: int = 500
    lambda_cal: float = 100.0
    lambda_seq: float = 2000.0

    def __post_init__(self) -> None:
        if self.n_observations < 100 or self.dimension < 2 or self.history < 1:
            raise ValueError("invalid simulation shape")
        if not self.dgp_seeds or len(set(self.dgp_seeds)) != len(self.dgp_seeds):
            raise ValueError("DGP seeds must be distinct")
        if (
            0.0 not in self.severities
            or any(value < 0 for value in self.severities)
            or len(set(self.severities)) != len(self.severities)
        ):
            raise ValueError("severities must be distinct, nonnegative, and include zero")
        if (
            not self.effective_neural_seeds
            or len(set(self.effective_neural_seeds)) != len(self.effective_neural_seeds)
        ):
            raise ValueError("neural seeds must be nonempty and distinct")
        if (
            not self.effective_forecast_seeds
            or len(set(self.effective_forecast_seeds)) != len(self.effective_forecast_seeds)
        ):
            raise ValueError("forecast seeds must be nonempty and distinct")
        if min(
            *self.dgp_seeds,
            *self.effective_neural_seeds,
            *self.effective_forecast_seeds,
            self.projection_seed,
        ) < 0:
            raise ValueError("random seeds must be nonnegative")
        if self.n_forecast_samples < 2:
            raise ValueError("at least two forecast samples are required")

    @property
    def effective_neural_seeds(self) -> tuple[int, ...]:
        """Return the crossed neural seeds, honoring the legacy scalar option."""
        return (self.neural_seed,) if self.neural_seed is not None else self.neural_seeds

    @property
    def effective_forecast_seeds(self) -> tuple[int, ...]:
        """Return forecast-replicate seeds, honoring the legacy scalar option."""
        return (
            (self.forecast_seed,)
            if self.forecast_seed is not None
            else self.forecast_seeds
        )


def _test_tensors(values: np.ndarray, scaler: Standardizer, history: int):
    split = chronological_split(values)
    test_values = np.concatenate([split.validation[-history:], split.test])
    test = WindowDataset(scaler.transform(test_values), history)
    return next(iter(DataLoader(test, batch_size=len(test), shuffle=False)))


def _checkpoint_rows(
    checkpoint: Path,
    config: IsolatedScaleShiftConfig,
) -> list[dict[str, float | int | str]]:
    """Load only complete DGP-by-neural-seed checkpoint blocks."""
    if not checkpoint.exists():
        return []
    prior = pd.read_csv(checkpoint).fillna({"failure": ""})
    keys = {"dgp_seed", "neural_seed", "model", "severity", "forecast_seed"}
    if missing := keys - set(prior.columns):
        raise ValueError(
            "checkpoint predates the crossed design or is missing keys: "
            f"{sorted(missing)}"
        )
    expected_pairs = {
        (dgp_seed, neural_seed)
        for dgp_seed in config.dgp_seeds
        for neural_seed in config.effective_neural_seeds
    }
    observed_pairs = set(
        prior[["dgp_seed", "neural_seed"]].itertuples(index=False, name=None)
    )
    if not observed_pairs.issubset(expected_pairs):
        raise ValueError("checkpoint contains DGP/neural-seed pairs outside the design")
    expected_cells = {
        (model, severity, forecast_seed)
        for model in ("RNN", "CA-RNN")
        for severity in config.severities
        for forecast_seed in config.effective_forecast_seeds
    }
    for pair, group in prior.groupby(["dgp_seed", "neural_seed"], sort=False):
        cells = list(
            group[["model", "severity", "forecast_seed"]].itertuples(
                index=False, name=None
            )
        )
        if len(cells) != len(set(cells)) or set(cells) != expected_cells:
            raise ValueError(f"checkpoint block {pair} is incomplete or duplicated")
    return prior.to_dict("records")


def _write_checkpoint(
    checkpoint: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    """Atomically replace a CSV checkpoint after a complete crossed block."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(checkpoint)


def _failure_rows(
    *,
    realization: int,
    dgp_seed: int,
    neural_replicate: int,
    neural_seed: int,
    model: str,
    config: IsolatedScaleShiftConfig,
    training_seconds: float,
    failure: Exception,
) -> list[dict[str, float | int | str]]:
    message = f"{type(failure).__name__}: {failure}"
    return [
        {
            "realization": realization,
            "dgp_seed": dgp_seed,
            "neural_replicate": neural_replicate,
            "neural_seed": neural_seed,
            "forecast_replicate": forecast_replicate,
            "forecast_seed": forecast_seed,
            "model": model,
            "severity": severity,
            "failure": message,
            "training_seconds": training_seconds,
        }
        for severity in config.severities
        for forecast_replicate, forecast_seed in enumerate(
            config.effective_forecast_seeds
        )
    ]


def run_isolated_scale_shift(
    config: IsolatedScaleShiftConfig | None = None,
    training_config: TrainingConfig | None = None,
    *,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fit a crossed DGP-by-neural-seed design and evaluate paired shifts.

    Only post-split innovation scale changes: transition, correlation, innovation
    type, and pre-shift observations remain fixed within a realization. Every
    neural seed is crossed with every DGP seed, while forecast-replicate seeds
    quantify finite predictive-simulation variability using common random numbers
    across the two models. Checkpoints are committed atomically after a complete
    DGP-by-neural-seed block. Failures are returned as rows, not silently discarded.
    """
    config = config or IsolatedScaleShiftConfig()
    base = training_config or TrainingConfig()
    shift_start = int(0.8 * config.n_observations)
    projections = make_projection_matrix(config.dimension, seed=config.projection_seed)
    rows: list[dict[str, float | int | str]] = []
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        rows = _checkpoint_rows(checkpoint, config)
    finished_pairs = {
        (int(row["dgp_seed"]), int(row["neural_seed"])) for row in rows
    }
    for realization, dgp_seed in enumerate(config.dgp_seeds):
        remaining_neural_seeds = [
            (neural_replicate, neural_seed)
            for neural_replicate, neural_seed in enumerate(
                config.effective_neural_seeds
            )
            if (dgp_seed, neural_seed) not in finished_pairs
        ]
        if not remaining_neural_seeds:
            continue
        reference = simulate_multivariate_series(
            config.n_observations,
            config.dimension,
            config.regime,
            dgp_seed,
            shift_start,
            0.0,
            shift_mechanism="scale",
        )
        split = chronological_split(reference)
        scaler = Standardizer.fit(split.train)
        train, validation, _ = partition_window_datasets(split, scaler, config.history)
        tests = {}
        for severity in config.severities:
            shifted = simulate_multivariate_series(
                config.n_observations,
                config.dimension,
                config.regime,
                dgp_seed,
                shift_start,
                severity,
                shift_mechanism="scale",
            )
            tests[severity] = _test_tensors(shifted, scaler, config.history)
        for neural_replicate, neural_seed in remaining_neural_seeds:
            pair_rows: list[dict[str, float | int | str]] = []
            for name, cal_weight, seq_weight in (
                ("RNN", 0.0, 0.0),
                ("CA-RNN", config.lambda_cal, config.lambda_seq),
            ):
                started = perf_counter()
                try:
                    torch.manual_seed(neural_seed)
                    model = CARNN(CARNNConfig(input_dim=config.dimension))
                    fitted = fit_model(
                        model,
                        train,
                        validation,
                        replace(
                            base,
                            seed=neural_seed,
                            lambda_cal=cal_weight,
                            lambda_seq=seq_weight,
                        ),
                        projections,
                        device,
                    )
                    training_seconds = perf_counter() - started
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    pair_rows.extend(
                        _failure_rows(
                            realization=realization,
                            dgp_seed=dgp_seed,
                            neural_replicate=neural_replicate,
                            neural_seed=neural_seed,
                            model=name,
                            config=config,
                            training_seconds=perf_counter() - started,
                            failure=exc,
                        )
                    )
                    continue
                for severity, (context, target_tensor) in tests.items():
                    target = target_tensor.numpy()
                    try:
                        mean, scale = predict_distribution(model, context)
                        analytic_pits = gaussian_projected_pits(
                            target,
                            mean[0].cpu().numpy(),
                            scale[0].cpu().numpy(),
                            projections.cpu().numpy(),
                        )
                        analytic = {
                            f"analytic_{key}": value
                            for key, value in pit_diagnostics(analytic_pits).items()
                        }
                        test_nll = predictive_nll(model, context, target_tensor)
                    except (ValueError, RuntimeError, FloatingPointError) as exc:
                        pair_rows.extend(
                            {
                                "realization": realization,
                                "dgp_seed": dgp_seed,
                                "neural_replicate": neural_replicate,
                                "neural_seed": neural_seed,
                                "forecast_replicate": forecast_replicate,
                                "forecast_seed": forecast_seed,
                                "model": name,
                                "severity": severity,
                                "failure": f"{type(exc).__name__}: {exc}",
                                "training_seconds": training_seconds,
                                "best_epoch": fitted.best_epoch,
                            }
                            for forecast_replicate, forecast_seed in enumerate(
                                config.effective_forecast_seeds
                            )
                        )
                        continue
                    for forecast_replicate, forecast_seed in enumerate(
                        config.effective_forecast_seeds
                    ):
                        try:
                            torch.manual_seed(forecast_seed)
                            samples = sample_forecasts(
                                model, context, config.n_forecast_samples
                            ).cpu().numpy()
                            sampled = evaluate_samples(
                                target,
                                samples,
                                projections=projections.cpu().numpy(),
                            )
                            failure = ""
                        except (ValueError, RuntimeError, FloatingPointError) as exc:
                            sampled = {}
                            failure = f"{type(exc).__name__}: {exc}"
                        pair_rows.append(
                            {
                                "realization": realization,
                                "dgp_seed": dgp_seed,
                                "neural_replicate": neural_replicate,
                                "neural_seed": neural_seed,
                                "forecast_replicate": forecast_replicate,
                                "forecast_seed": forecast_seed,
                                "model": name,
                                "severity": severity,
                                "failure": failure,
                                "training_seconds": training_seconds,
                                "best_epoch": fitted.best_epoch,
                                "test_nll": test_nll,
                                **analytic,
                                **sampled,
                            }
                        )
            rows.extend(pair_rows)
            if checkpoint is not None:
                _write_checkpoint(checkpoint, rows)
    return pd.DataFrame(rows)


def summarize_isolated_shift(
    raw: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = (
        "test_nll",
        "mean_squared_error",
        "energy_score",
        "variogram_score",
        "interval_score",
        "analytic_pit_calibration_error",
        "analytic_pit_mean_absolute_autocorrelation",
    ),
) -> pd.DataFrame:
    """Mean paired differences and DGP-level Monte Carlo standard errors.

    For a crossed design, neural and forecast replicates are first averaged
    within each DGP realization. The Monte Carlo standard error is then computed
    across independent DGP seeds. Legacy single-seed result frames remain valid.
    """
    required = {"model", "severity", "failure", *metrics}
    if missing := required - set(raw.columns):
        raise ValueError(f"missing raw columns: {sorted(missing)}")
    if "dgp_seed" not in raw and "realization" not in raw:
        raise ValueError("raw results require dgp_seed or realization")
    frame = raw.copy()
    frame["_dgp"] = (
        frame["dgp_seed"] if "dgp_seed" in frame else frame["realization"]
    )
    frame["_neural"] = frame.get("neural_seed", 0)
    frame["_forecast"] = frame.get("forecast_seed", 0)
    frame["failure"] = frame["failure"].fillna("").astype(str)
    rows = []
    for severity in sorted(frame.severity.unique()):
        at_severity = frame.loc[frame.severity.eq(severity)].copy()
        for metric in metrics:
            at_severity.loc[at_severity.failure.ne(""), metric] = np.nan
            paired = at_severity.pivot(
                index=["_dgp", "_neural", "_forecast"],
                columns="model",
                values=metric,
            )
            if not {"CA-RNN", "RNN"}.issubset(paired.columns):
                raise ValueError("raw results must contain paired RNN and CA-RNN rows")
            valid = paired[["CA-RNN", "RNN"]].dropna()
            cell_difference = valid["CA-RNN"] - valid["RNN"]
            dgp_difference = cell_difference.groupby(level="_dgp").mean()
            dgp_rnn = valid["RNN"].groupby(level="_dgp").mean()
            dgp_ca_rnn = valid["CA-RNN"].groupby(level="_dgp").mean()
            count = len(dgp_difference)
            total_cells = (
                frame._dgp.nunique()
                * frame._neural.nunique()
                * frame._forecast.nunique()
            )
            rows.append(
                {
                    "severity": severity,
                    "metric": metric,
                    "n_valid_pairs": count,
                    "n_valid_crossed_cells": len(valid),
                    "n_dgp_realizations": frame._dgp.nunique(),
                    "n_neural_seeds": frame._neural.nunique(),
                    "n_forecast_replicates": frame._forecast.nunique(),
                    "failure_frequency": 1 - len(valid) / total_cells,
                    "mean_paired_difference": (
                        float(dgp_difference.mean()) if count else np.nan
                    ),
                    "mc_standard_error": (
                        float(dgp_difference.std(ddof=1) / np.sqrt(count))
                        if count >= 2 else np.nan
                    ),
                    "mean_rnn": float(dgp_rnn.mean()) if count else np.nan,
                    "mean_ca_rnn": float(dgp_ca_rnn.mean()) if count else np.nan,
                }
            )
    return pd.DataFrame(rows)


def crossed_isolated_shift_inference(
    raw: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = (
        "test_nll",
        "mean_squared_error",
        "energy_score",
        "variogram_score",
        "interval_score",
        "analytic_pit_calibration_error",
        "analytic_pit_mean_absolute_autocorrelation",
    ),
    severities: tuple[float, ...] | None = None,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 1408,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Crossed multiway bootstrap with simultaneous max-|z| intervals.

    Each bootstrap replicate independently resamples the DGP, neural-seed, and
    forecast-replicate factor labels, then averages the Cartesian product of the
    selected levels. The same factor draws are used for every requested metric
    and severity. Simultaneous intervals control one declared family containing
    all requested metric-by-severity cells. Differences are CA-RNN minus RNN.

    The procedure is a crossed random-effects sensitivity analysis. It remains
    conditional on the selected DGP family, architecture, hyperparameters, and
    projection set. With ``require_complete=True``, any failed or missing cell
    stops inference rather than silently changing the estimand.
    """
    if n_bootstrap < 2 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be nonempty and distinct")
    required = {"model", "severity", "failure", *metrics}
    if missing := required - set(raw.columns):
        raise ValueError(f"missing raw columns: {sorted(missing)}")
    if "dgp_seed" not in raw and "realization" not in raw:
        raise ValueError("raw results require dgp_seed or realization")

    frame = raw.copy()
    frame["_dgp"] = (
        frame["dgp_seed"] if "dgp_seed" in frame else frame["realization"]
    )
    frame["_neural"] = frame.get("neural_seed", 0)
    frame["_forecast"] = frame.get("forecast_seed", 0)
    frame["failure"] = frame["failure"].fillna("").astype(str)
    selected_severities = (
        tuple(sorted(frame.severity.unique())) if severities is None else severities
    )
    if (
        not selected_severities
        or len(set(selected_severities)) != len(selected_severities)
        or not set(selected_severities).issubset(set(frame.severity.unique()))
    ):
        raise ValueError("severities must be distinct values present in raw results")

    dgp_levels = tuple(sorted(frame._dgp.unique()))
    neural_levels = tuple(sorted(frame._neural.unique()))
    forecast_levels = tuple(sorted(frame._forecast.unique()))
    if len(dgp_levels) < 2:
        raise ValueError("at least two DGP seeds are required for inference")
    cell_index = pd.MultiIndex.from_product(
        [dgp_levels, neural_levels, forecast_levels],
        names=["_dgp", "_neural", "_forecast"],
    )
    labels = [
        (severity, metric)
        for severity in selected_severities
        for metric in metrics
    ]
    rnn_columns = []
    ca_columns = []
    for severity, metric in labels:
        selected = frame.loc[frame.severity.eq(severity)].copy()
        selected.loc[selected.failure.ne(""), metric] = np.nan
        paired = selected.pivot(
            index=["_dgp", "_neural", "_forecast"],
            columns="model",
            values=metric,
        ).reindex(cell_index)
        if not {"CA-RNN", "RNN"}.issubset(paired.columns):
            raise ValueError("raw results must contain paired RNN and CA-RNN rows")
        rnn_columns.append(paired["RNN"].to_numpy(dtype=float))
        ca_columns.append(paired["CA-RNN"].to_numpy(dtype=float))

    shape = (len(dgp_levels), len(neural_levels), len(forecast_levels), len(labels))
    rnn = np.column_stack(rnn_columns).reshape(shape)
    ca_rnn = np.column_stack(ca_columns).reshape(shape)
    difference = ca_rnn - rnn
    complete = np.isfinite(rnn).all(axis=-1) & np.isfinite(ca_rnn).all(axis=-1)
    family_failure_frequency = 1.0 - float(complete.mean())
    if require_complete and not complete.all():
        raise ValueError(
            "crossed inference requires a complete model-by-factor grid; "
            f"family failure frequency is {family_failure_frequency:.6f}"
        )
    with np.errstate(invalid="ignore"):
        observed = np.nanmean(difference, axis=(0, 1, 2))
        mean_rnn = np.nanmean(rnn, axis=(0, 1, 2))
        mean_ca_rnn = np.nanmean(ca_rnn, axis=(0, 1, 2))

    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, len(labels)), dtype=float)
    metric_axis = np.arange(len(labels))
    for bootstrap_index in range(n_bootstrap):
        dgp_index = rng.integers(0, len(dgp_levels), size=len(dgp_levels))
        neural_index = rng.integers(
            0, len(neural_levels), size=len(neural_levels)
        )
        forecast_index = rng.integers(
            0, len(forecast_levels), size=len(forecast_levels)
        )
        selected = difference[
            np.ix_(dgp_index, neural_index, forecast_index, metric_axis)
        ]
        with np.errstate(invalid="ignore"):
            draws[bootstrap_index] = np.nanmean(selected, axis=(0, 1, 2))
    finite_draws = np.isfinite(draws).all(axis=1)
    draws = draws[finite_draws]
    if len(draws) < 2:
        raise ValueError("too few finite crossed-bootstrap replicates")

    alpha = 1.0 - confidence
    pointwise_lower, pointwise_upper = np.quantile(
        draws, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0
    )
    bootstrap_se = draws.std(axis=0, ddof=1)
    active = np.isfinite(bootstrap_se) & (bootstrap_se > np.finfo(float).eps)
    if active.any():
        standardized = np.abs(
            (draws[:, active] - observed[active]) / bootstrap_se[active]
        )
        maximum = standardized.max(axis=1)
        critical_value = float(np.quantile(maximum, confidence))
    else:
        critical_value = 0.0
    simultaneous_lower = observed - critical_value * bootstrap_se
    simultaneous_upper = observed + critical_value * bootstrap_se

    rows = []
    total_cells = len(dgp_levels) * len(neural_levels) * len(forecast_levels)
    for index, (severity, metric) in enumerate(labels):
        valid_cells = np.isfinite(difference[..., index])
        rows.append(
            {
                "severity": severity,
                "metric": metric,
                "mean_paired_difference": float(observed[index]),
                "mean_rnn": float(mean_rnn[index]),
                "mean_ca_rnn": float(mean_ca_rnn[index]),
                "crossed_bootstrap_se": float(bootstrap_se[index]),
                "pointwise_ci_lower": float(pointwise_lower[index]),
                "pointwise_ci_upper": float(pointwise_upper[index]),
                "simultaneous_ci_lower": float(simultaneous_lower[index]),
                "simultaneous_ci_upper": float(simultaneous_upper[index]),
                "pointwise_resolved": bool(
                    pointwise_lower[index] > 0 or pointwise_upper[index] < 0
                ),
                "simultaneous_resolved": bool(
                    simultaneous_lower[index] > 0
                    or simultaneous_upper[index] < 0
                ),
                "probability_difference_below_zero": float(
                    np.mean(draws[:, index] < 0)
                ),
                "n_valid_crossed_cells": int(valid_cells.sum()),
                "metric_failure_frequency": 1.0
                - float(valid_cells.sum()) / total_cells,
                "family_failure_frequency": family_failure_frequency,
                "n_dgp_seeds": len(dgp_levels),
                "n_neural_seeds": len(neural_levels),
                "n_forecast_replicates": len(forecast_levels),
                "n_bootstrap": n_bootstrap,
                "n_finite_bootstrap": len(draws),
                "confidence": confidence,
                "family_size": len(labels),
                "simultaneous_critical_value": critical_value,
            }
        )
    return pd.DataFrame(rows)
