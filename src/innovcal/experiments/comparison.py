"""Matched RNN, BRNN, CA-RNN, and CA-BRNN experiments."""

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from innovcal.ca_rnn import (
    CARNN,
    BayesianCARNN,
    CARNNConfig,
    TrainingConfig,
    fit_bayesian_model_with_kl_selection,
    fit_model,
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
from innovcal.evaluation import evaluate_samples


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    histories: dict[str, pd.DataFrame]
    models: dict[str, torch.nn.Module]
    standardizer: Standardizer
    projections: torch.Tensor
    target: np.ndarray
    forecast_samples: dict[str, np.ndarray]
    kl_selection: pd.DataFrame | None = None
    selected_kl_weight: float | None = None


def _test_arrays(dataset: WindowDataset) -> tuple[torch.Tensor, np.ndarray]:
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    context, target = next(iter(loader))
    return context, target.numpy()


def run_four_model_comparison(
    values: np.ndarray,
    history: int = 20,
    model_config: CARNNConfig | None = None,
    training_config: TrainingConfig | None = None,
    lambda_cal: float = 10.0,
    lambda_seq: float = 2000.0,
    n_forecast_samples: int = 500,
    device: str = "cpu",
    bayesian_kl_grid: Iterable[float] | None = None,
) -> ExperimentResult:
    """Fit the four variants used in the reference paper's comparison logic.

    When ``bayesian_kl_grid`` is provided, one KL weight is selected using the
    likelihood-only BRNN's chronological posterior-predictive validation NLL.
    That same weight is then held fixed for CA-BRNN, preserving the controlled
    comparison of the calibration penalty. Test observations are accessed only
    after selection. Omitting the grid preserves the fixed-weight behavior in
    ``training_config``.
    """
    split = chronological_split(values)
    standardizer = Standardizer.fit(split.train)
    train, validation, test = partition_window_datasets(
        split, standardizer, history
    )
    dimension = values.shape[1]
    model_config = model_config or CARNNConfig(input_dim=dimension)
    if model_config.input_dim != dimension:
        raise ValueError("model input dimension does not match data")
    training_config = training_config or TrainingConfig()
    projections = make_projection_matrix(dimension, seed=training_config.seed)
    specifications = {
        "RNN": (CARNN, 0.0, 0.0),
        "BRNN": (BayesianCARNN, 0.0, 0.0),
        "CA-RNN": (CARNN, lambda_cal, lambda_seq),
        "CA-BRNN": (BayesianCARNN, lambda_cal, lambda_seq),
    }
    kl_selection = None
    selected_kl_weight = None
    selected_brnn = None
    if bayesian_kl_grid is not None:
        selected_brnn = fit_bayesian_model_with_kl_selection(
            lambda: BayesianCARNN(model_config),
            train,
            validation,
            replace(training_config, lambda_cal=0.0, lambda_seq=0.0),
            bayesian_kl_grid,
            projections,
            device,
        )
        selected_kl_weight = selected_brnn.selected_weight
        kl_selection = pd.DataFrame(
            {
                "kl_weight": list(selected_brnn.validation_nll),
                "validation_nll": list(selected_brnn.validation_nll.values()),
            }
        )
        kl_selection["selected"] = kl_selection["kl_weight"].eq(selected_kl_weight)

    context, target = _test_arrays(test)
    rows, histories, models, forecasts = [], {}, {}, {}
    for name, (model_class, cal_weight, seq_weight) in specifications.items():
        config = replace(
            training_config,
            lambda_cal=cal_weight,
            lambda_seq=seq_weight,
        )
        if name == "BRNN" and selected_brnn is not None:
            model, result = selected_brnn.model, selected_brnn.fit
        else:
            # The same initialization seed is reused within matched model pairs.
            torch.manual_seed(training_config.seed)
            model = model_class(model_config)
            if name == "CA-BRNN" and selected_kl_weight is not None:
                config = replace(config, kl_weight=selected_kl_weight)
            result = fit_model(model, train, validation, config, projections, device)
        torch.manual_seed(training_config.seed + 10_000)
        test_nll = predictive_nll(model, context, torch.as_tensor(target), 50)
        torch.manual_seed(training_config.seed + 20_000)
        samples = sample_forecasts(model, context, n_forecast_samples).cpu().numpy()
        # Evaluation remains standardized so all assets contribute comparably.
        metrics = evaluate_samples(target, samples, projections=projections.cpu().numpy())
        rows.append(
            {"model": name, "best_epoch": result.best_epoch, "nll": test_nll, **metrics}
        )
        histories[name] = pd.DataFrame(result.history)
        models[name] = model
        forecasts[name] = samples
    return ExperimentResult(
        pd.DataFrame(rows),
        histories,
        models,
        standardizer,
        projections,
        target,
        forecasts,
        kl_selection,
        selected_kl_weight,
    )


def lambda_sweep(
    values: np.ndarray,
    lambdas: list[float],
    history: int = 20,
    training_config: TrainingConfig | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Paper-style calibration-weight sweep for the frequentist CA-RNN."""
    split = chronological_split(values)
    standardizer = Standardizer.fit(split.train)
    train, validation, test = partition_window_datasets(
        split, standardizer, history
    )
    dimension = values.shape[1]
    base = training_config or TrainingConfig()
    projections = make_projection_matrix(dimension, seed=base.seed)
    context, target = _test_arrays(test)
    rows = []
    for value in lambdas:
        torch.manual_seed(base.seed)
        model = CARNN(CARNNConfig(input_dim=dimension))
        fit = fit_model(
            model,
            train,
            validation,
            replace(base, lambda_cal=value, lambda_seq=0.0),
            projections,
            device,
        )
        torch.manual_seed(base.seed + 20_000)
        samples = sample_forecasts(model, context, 500).cpu().numpy()
        rows.append(
            {
                "lambda_cal": value,
                "best_epoch": fit.best_epoch,
                "nll": predictive_nll(model, context, torch.as_tensor(target)),
                **evaluate_samples(
                    target, samples, projections=projections.cpu().numpy()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_temporal_penalty_ablation(
    values: np.ndarray,
    sequential_weights: list[float],
    calibration_weight: float = 100.0,
    history: int = 20,
    model_config: CARNNConfig | None = None,
    training_config: TrainingConfig | None = None,
    n_forecast_samples: int = 500,
    device: str = "cpu",
) -> ExperimentResult:
    """Fit a matched fixed-epoch temporal-loss ablation for CA-RNN.

    All specifications start from identical parameters, see observations in the
    same order, train for the same number of epochs, and use common predictive
    random numbers. This isolates the training-objective change.
    """
    if any(weight < 0 for weight in sequential_weights):
        raise ValueError("sequential weights must be nonnegative")
    split = chronological_split(values)
    standardizer = Standardizer.fit(split.train)
    train, validation, test = partition_window_datasets(
        split, standardizer, history
    )
    dimension = values.shape[1]
    model_config = model_config or CARNNConfig(input_dim=dimension)
    if model_config.input_dim != dimension:
        raise ValueError("model input dimension does not match data")
    base = training_config or TrainingConfig()
    base = replace(base, early_stopping=False, selection_metric="nll")
    projections = make_projection_matrix(dimension, seed=base.seed)
    context, target = _test_arrays(test)
    specifications = [
        ("RNN", 0.0, 0.0),
        ("CA-RNN (temporal penalty removed)", calibration_weight, 0.0),
    ]
    specifications.extend(
        (
            "CA-RNN" if weight == 2000.0 else f"CA-RNN (temporal weight {weight:g})",
            calibration_weight,
            weight,
        )
        for weight in sequential_weights
        if weight > 0
    )
    rows, histories, models, forecasts = [], {}, {}, {}
    for name, cal_weight, seq_weight in specifications:
        torch.manual_seed(base.seed)
        model = CARNN(model_config)
        fit = fit_model(
            model,
            train,
            validation,
            replace(base, lambda_cal=cal_weight, lambda_seq=seq_weight),
            projections,
            device,
        )
        torch.manual_seed(base.seed + 10_000)
        test_nll = predictive_nll(model, context, torch.as_tensor(target))
        torch.manual_seed(base.seed + 20_000)
        samples = sample_forecasts(model, context, n_forecast_samples).cpu().numpy()
        metrics = evaluate_samples(
            target, samples, projections=projections.cpu().numpy()
        )
        rows.append(
            {
                "model": name,
                "lambda_cal": cal_weight,
                "lambda_seq": seq_weight,
                "epochs": len(fit.history),
                "nll": test_nll,
                **metrics,
            }
        )
        histories[name] = pd.DataFrame(fit.history)
        models[name] = model
        forecasts[name] = samples
    return ExperimentResult(
        pd.DataFrame(rows),
        histories,
        models,
        standardizer,
        projections,
        target,
        forecasts,
    )
