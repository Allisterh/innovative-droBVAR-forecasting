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
    neural_seed: int = 606
    projection_seed: int = 606
    forecast_seed: int = 20_606
    n_forecast_samples: int = 500
    lambda_cal: float = 100.0
    lambda_seq: float = 2000.0

    def __post_init__(self) -> None:
        if self.n_observations < 100 or self.dimension < 2 or self.history < 1:
            raise ValueError("invalid simulation shape")
        if not self.dgp_seeds or len(set(self.dgp_seeds)) != len(self.dgp_seeds):
            raise ValueError("DGP seeds must be distinct")
        if 0.0 not in self.severities or any(value < 0 for value in self.severities):
            raise ValueError("severities must include zero and be nonnegative")
        if self.n_forecast_samples < 2:
            raise ValueError("at least two forecast samples are required")


def _test_tensors(values: np.ndarray, scaler: Standardizer, history: int):
    split = chronological_split(values)
    test_values = np.concatenate([split.validation[-history:], split.test])
    test = WindowDataset(scaler.transform(test_values), history)
    return next(iter(DataLoader(test, batch_size=len(test), shuffle=False)))


def run_isolated_scale_shift(
    config: IsolatedScaleShiftConfig | None = None,
    training_config: TrainingConfig | None = None,
    *,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fit RNN and CA-RNN once per independent DGP and evaluate paired shifts.

    Only post-split innovation scale changes: transition, correlation, innovation
    type, and pre-shift observations remain fixed within a realization. A single
    neural seed is used per realization; distinct DGP seeds identify independent
    data realizations. Failures are returned as rows, not silently discarded.
    """
    config = config or IsolatedScaleShiftConfig()
    base = training_config or TrainingConfig()
    shift_start = int(0.8 * config.n_observations)
    projections = make_projection_matrix(config.dimension, seed=config.projection_seed)
    rows: list[dict[str, float | int | str]] = []
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None and checkpoint.exists():
        prior = pd.read_csv(checkpoint).fillna({"failure": ""})
        expected = 2 * len(config.severities)
        completed = prior.groupby("dgp_seed").size()
        if not set(completed.index).issubset(config.dgp_seeds) or not completed.eq(expected).all():
            raise ValueError("checkpoint is incomplete or incompatible with DGP seeds")
        rows = prior.to_dict("records")
    finished_seeds = {row["dgp_seed"] for row in rows}
    for realization, dgp_seed in enumerate(config.dgp_seeds):
        if dgp_seed in finished_seeds:
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
        neural_seed = config.neural_seed + realization
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
                    context, target_tensor = _test_tensors(shifted, scaler, config.history)
                    target = target_tensor.numpy()
                    mean, scale = predict_distribution(model, context)
                    analytic_pits = gaussian_projected_pits(
                        target,
                        mean[0].cpu().numpy(),
                        scale[0].cpu().numpy(),
                        projections.cpu().numpy(),
                    )
                    torch.manual_seed(config.forecast_seed)
                    samples = sample_forecasts(
                        model, context, config.n_forecast_samples
                    ).cpu().numpy()
                    rows.append(
                        {
                            "realization": realization,
                            "dgp_seed": dgp_seed,
                            "neural_seed": neural_seed,
                            "model": name,
                            "severity": severity,
                            "failure": "",
                            "training_seconds": training_seconds,
                            "best_epoch": fitted.best_epoch,
                            "test_nll": predictive_nll(model, context, target_tensor),
                            **{
                                f"analytic_{key}": value
                                for key, value in pit_diagnostics(analytic_pits).items()
                            },
                            **evaluate_samples(
                                target, samples, projections=projections.cpu().numpy()
                            ),
                        }
                    )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                for severity in config.severities:
                    rows.append(
                        {
                            "realization": realization,
                            "dgp_seed": dgp_seed,
                            "neural_seed": neural_seed,
                            "model": name,
                            "severity": severity,
                            "failure": f"{type(exc).__name__}: {exc}",
                            "training_seconds": perf_counter() - started,
                        }
                    )
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
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
    """Mean paired differences and DGP-level Monte Carlo standard errors."""
    required = {"realization", "model", "severity", "failure", *metrics}
    if missing := required - set(raw.columns):
        raise ValueError(f"missing raw columns: {sorted(missing)}")
    rows = []
    for severity in sorted(raw.severity.unique()):
        at_severity = raw.loc[raw.severity.eq(severity)]
        for metric in metrics:
            paired = at_severity.pivot(index="realization", columns="model", values=metric)
            valid = paired[["CA-RNN", "RNN"]].dropna()
            difference = valid["CA-RNN"] - valid["RNN"]
            count = len(difference)
            rows.append(
                {
                    "severity": severity,
                    "metric": metric,
                    "n_valid_pairs": count,
                    "n_dgp_realizations": raw.realization.nunique(),
                    "failure_frequency": 1 - count / raw.realization.nunique(),
                    "mean_paired_difference": float(difference.mean()) if count else np.nan,
                    "mc_standard_error": (
                        float(difference.std(ddof=1) / np.sqrt(count))
                        if count >= 2 else np.nan
                    ),
                    "mean_rnn": float(valid["RNN"].mean()) if count else np.nan,
                    "mean_ca_rnn": float(valid["CA-RNN"].mean()) if count else np.nan,
                }
            )
    return pd.DataFrame(rows)
