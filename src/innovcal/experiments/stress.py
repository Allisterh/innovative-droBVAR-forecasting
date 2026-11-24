"""Frozen-model distribution-shift experiments for CA-RNN variants."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from innovcal.ca_rnn import TrainingConfig, predictive_nll, sample_forecasts
from innovcal.ca_rnn.config import CARNNConfig
from innovcal.data.simulation import simulate_multivariate_series
from innovcal.data.windows import WindowDataset, chronological_split
from innovcal.evaluation import evaluate_samples
from innovcal.experiments.comparison import run_four_model_comparison


@dataclass(frozen=True)
class ShiftConfig:
    n_observations: int = 1200
    shift_fraction: float = 0.8
    dimension: int = 4
    regime: str = "gaussian"
    seed: int = 123


def run_shift_curve(
    severities: list[float],
    config: ShiftConfig | None = None,
    training_config: TrainingConfig | None = None,
    history: int = 20,
    lambda_cal: float = 100.0,
    sequential_weight: float | None = 2000.0,
    n_forecast_samples: int = 500,
    device: str = "cpu",
    bayesian_kl_grid: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Fit once on the reference law and evaluate frozen models under shift.

    The DGP uses common pre-shift observations for every severity. Models and
    standardization are estimated once from severity zero; only the untouched
    test realization changes along the severity curve.
    """
    config = config or ShiftConfig()
    if not severities or any(severity < 0 for severity in severities):
        raise ValueError("severities must be a non-empty nonnegative list")
    training_config = training_config or TrainingConfig()
    shift_start = int(config.n_observations * config.shift_fraction)
    reference = simulate_multivariate_series(
        config.n_observations,
        config.dimension,
        config.regime,
        config.seed,
        shift_start,
        0.0,
    )
    fitted = run_four_model_comparison(
        reference,
        history=history,
        model_config=CARNNConfig(config.dimension),
        training_config=training_config,
        lambda_cal=lambda_cal,
        lambda_seq=sequential_weight or 0.0,
        n_forecast_samples=n_forecast_samples,
        device=device,
        bayesian_kl_grid=bayesian_kl_grid,
    )
    rows = []
    for severity in severities:
        values = simulate_multivariate_series(
            config.n_observations,
            config.dimension,
            config.regime,
            config.seed,
            shift_start,
            severity,
        )
        shifted_split = chronological_split(values)
        test_values = fitted.standardizer.transform(
            np.concatenate(
                [shifted_split.validation[-history:], shifted_split.test], axis=0
            )
        )
        test = WindowDataset(test_values, history)
        context, target = next(
            iter(DataLoader(test, batch_size=len(test), shuffle=False))
        )
        for model_index, (name, model) in enumerate(fitted.models.items()):
            torch.manual_seed(training_config.seed + 10_000)
            nll = predictive_nll(model, context, target, n_weight_samples=50)
            torch.manual_seed(training_config.seed + 20_000)
            samples = sample_forecasts(model, context, n_forecast_samples).cpu().numpy()
            metrics = evaluate_samples(
                target.numpy(),
                samples,
                projections=fitted.projections.cpu().numpy(),
            )
            rows.append(
                {
                    "severity": severity,
                    "model": name,
                    "model_index": model_index,
                    "lambda_cal": lambda_cal if name.startswith("CA-") else 0.0,
                    "lambda_seq": (
                        sequential_weight if name in {"CA-RNN", "CA-BRNN"} else 0.0
                    ),
                    "kl_weight": (
                        fitted.selected_kl_weight
                        if name in {"BRNN", "CA-BRNN"}
                        and fitted.selected_kl_weight is not None
                        else training_config.kl_weight
                        if name in {"BRNN", "CA-BRNN"}
                        else np.nan
                    ),
                    "nll": nll,
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["severity"].eq(0)].set_index("model")
    for metric in (
        "nll",
        "energy_score",
        "interval_score",
        "pit_calibration_error",
        "pit_mean_absolute_autocorrelation",
    ):
        reference_values = frame["model"].map(baseline[metric])
        frame[f"delta_{metric}"] = frame[metric] - reference_values
    return frame
