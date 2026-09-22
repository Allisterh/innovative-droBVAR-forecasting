"""Focused optimization-seed experiment for CA-RNN and its ablation."""

from dataclasses import dataclass, replace
from datetime import date

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


@dataclass
class MultiSeedResult:
    """Metrics and retained matched forecasts from repeated model fits."""

    metrics: pd.DataFrame
    histories: dict[tuple[int, str], pd.DataFrame]
    target: np.ndarray
    projections: np.ndarray
    forecast_samples: dict[str, np.ndarray]


@dataclass(frozen=True)
class RegimeDefinition:
    """Inclusive calendar boundaries for an expanding-window experiment."""

    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str

    def __post_init__(self) -> None:
        boundaries = [
            date.fromisoformat(value)
            for value in (
                self.train_start,
                self.train_end,
                self.validation_start,
                self.validation_end,
                self.test_start,
                self.test_end,
            )
        ]
        if boundaries != sorted(boundaries) or len(set(boundaries)) != 6:
            raise ValueError("regime boundaries must be strictly chronological")


def run_frequentist_seed_comparison(
    values: np.ndarray,
    seeds: list[int],
    history: int = 20,
    model_config: CARNNConfig | None = None,
    training_config: TrainingConfig | None = None,
    lambda_cal: float = 100.0,
    lambda_seq: float = 2000.0,
    projection_seed: int = 606,
    forecast_seed: int = 20_606,
    n_forecast_samples: int = 500,
    device: str = "cpu",
    models: tuple[str, ...] = (
        "RNN",
        "CA-RNN (temporal penalty removed)",
        "CA-RNN",
    ),
) -> MultiSeedResult:
    """Refit RNN, CA-RNN, and the temporal-penalty ablation over matched seeds.

    Within a seed, all models share initial parameters, data order, projection
    directions, and predictive random numbers. Only the objective changes.
    Projection and forecast seeds remain fixed across fits so variation between
    replications is attributable to initialization and optimization.
    """
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("provide at least two distinct training seeds")
    split = chronological_split(np.asarray(values, dtype=float))
    standardizer = Standardizer.fit(split.train)
    train, validation, test = partition_window_datasets(
        split, standardizer, history
    )
    dimension = values.shape[1]
    model_config = model_config or CARNNConfig(input_dim=dimension)
    base = training_config or TrainingConfig()
    projections = make_projection_matrix(dimension, seed=projection_seed)
    context, target_tensor = next(
        iter(DataLoader(test, batch_size=len(test), shuffle=False))
    )
    target = target_tensor.numpy()
    available = {
        "RNN": (0.0, 0.0),
        "CA-RNN (temporal penalty removed)": (lambda_cal, 0.0),
        "CA-RNN": (lambda_cal, lambda_seq),
    }
    if len(set(models)) != len(models) or not models:
        raise ValueError("models must contain distinct supported specifications")
    unknown = set(models) - set(available)
    if unknown:
        raise ValueError(f"unsupported models: {sorted(unknown)}")
    specifications = {name: available[name] for name in models}
    rows: list[dict[str, float | int | str]] = []
    histories = {}
    retained = {name: [] for name in specifications}
    for training_seed in seeds:
        for name, (calibration_weight, sequential_weight) in specifications.items():
            torch.manual_seed(training_seed)
            model = CARNN(model_config)
            config = replace(
                base,
                seed=training_seed,
                lambda_cal=calibration_weight,
                lambda_seq=sequential_weight,
            )
            fitted = fit_model(model, train, validation, config, projections, device)
            torch.manual_seed(forecast_seed)
            samples = sample_forecasts(
                model, context, n_forecast_samples
            ).cpu().numpy()
            metrics = evaluate_samples(
                target, samples, projections=projections.cpu().numpy()
            )
            mean, scale = predict_distribution(model, context)
            analytic_pits = gaussian_projected_pits(
                target,
                mean[0].cpu().numpy(),
                scale[0].cpu().numpy(),
                projections.cpu().numpy(),
            )
            analytic = pit_diagnostics(analytic_pits)
            rows.append(
                {
                    "training_seed": training_seed,
                    "model": name,
                    "best_epoch": fitted.best_epoch,
                    "test_nll": predictive_nll(model, context, target_tensor),
                    **{f"analytic_{key}": value for key, value in analytic.items()},
                    **metrics,
                }
            )
            histories[(training_seed, name)] = pd.DataFrame(fitted.history)
            retained[name].append(samples)
    forecasts = {name: np.stack(samples) for name, samples in retained.items()}
    return MultiSeedResult(
        pd.DataFrame(rows), histories, target, projections.cpu().numpy(), forecasts
    )


def run_regime_seed_comparison(
    values: np.ndarray,
    dates: np.ndarray,
    regime: RegimeDefinition,
    seeds: list[int],
    history: int = 20,
    model_config: CARNNConfig | None = None,
    training_config: TrainingConfig | None = None,
    lambda_cal: float = 100.0,
    lambda_seq: float = 2000.0,
    projection_seed: int = 606,
    forecast_seed: int = 20_606,
    n_forecast_samples: int = 500,
    device: str = "cpu",
) -> MultiSeedResult:
    """Run a matched multi-seed comparison on one calendar-defined regime."""
    values = np.asarray(values, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if len(values) != len(dates) or not dates.is_monotonic_increasing:
        raise ValueError("dates must align with values and be chronological")
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("provide at least two distinct training seeds")

    def select(start: str, end: str) -> np.ndarray:
        mask = (dates >= start) & (dates <= end)
        return values[mask].copy()

    train_values = select(regime.train_start, regime.train_end)
    validation_values = select(regime.validation_start, regime.validation_end)
    test_values = select(regime.test_start, regime.test_end)
    if min(map(len, (train_values, validation_values, test_values))) <= history:
        raise ValueError(f"regime {regime.name!r} has an insufficient partition")
    standardizer = Standardizer.fit(train_values)
    train = WindowDataset(standardizer.transform(train_values), history)
    validation_context = np.concatenate([train_values[-history:], validation_values])
    validation = WindowDataset(standardizer.transform(validation_context), history)
    test_context = np.concatenate([validation_values[-history:], test_values])
    test = WindowDataset(standardizer.transform(test_context), history)
    dimension = values.shape[1]
    model_config = model_config or CARNNConfig(input_dim=dimension)
    base = training_config or TrainingConfig()
    projections = make_projection_matrix(dimension, seed=projection_seed)
    context, target_tensor = next(
        iter(DataLoader(test, batch_size=len(test), shuffle=False))
    )
    target = target_tensor.numpy()
    specifications = {
        "RNN": (0.0, 0.0),
        "CA-RNN (temporal penalty removed)": (lambda_cal, 0.0),
        "CA-RNN": (lambda_cal, lambda_seq),
    }
    rows: list[dict[str, float | int | str]] = []
    histories = {}
    retained = {name: [] for name in specifications}
    for training_seed in seeds:
        for name, (calibration_weight, sequential_weight) in specifications.items():
            torch.manual_seed(training_seed)
            model = CARNN(model_config)
            fitted = fit_model(
                model,
                train,
                validation,
                replace(
                    base,
                    seed=training_seed,
                    lambda_cal=calibration_weight,
                    lambda_seq=sequential_weight,
                ),
                projections,
                device,
            )
            torch.manual_seed(forecast_seed)
            samples = sample_forecasts(
                model, context, n_forecast_samples
            ).cpu().numpy()
            mean, scale = predict_distribution(model, context)
            rows.append(
                {
                    "regime": regime.name,
                    "training_seed": training_seed,
                    "model": name,
                    "best_epoch": fitted.best_epoch,
                    "test_nll": predictive_nll(model, context, target_tensor),
                    **{
                        f"analytic_{key}": value
                        for key, value in pit_diagnostics(
                            gaussian_projected_pits(
                                target,
                                mean[0].cpu().numpy(),
                                scale[0].cpu().numpy(),
                                projections.cpu().numpy(),
                            )
                        ).items()
                    },
                    **evaluate_samples(
                        target, samples, projections=projections.cpu().numpy()
                    ),
                }
            )
            histories[(training_seed, name)] = pd.DataFrame(fitted.history)
            retained[name].append(samples)
    forecasts = {name: np.stack(samples) for name, samples in retained.items()}
    return MultiSeedResult(
        pd.DataFrame(rows), histories, target, projections.cpu().numpy(), forecasts
    )
