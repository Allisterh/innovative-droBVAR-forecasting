"""Leakage-aware training for the four paper-style model variants."""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from innovcal.ca_rnn.config import TrainingConfig
from innovcal.ca_rnn.losses import (
    calibration_loss,
    gaussian_nll,
    make_projection_matrix,
    projected_pits,
    sequential_calibration_loss,
)
from innovcal.ca_rnn.predict import predictive_nll


@dataclass
class FitResult:
    history: list[dict[str, float]]
    best_epoch: int
    stopped_early: bool


@dataclass
class KLWeightSelectionResult:
    """Validation-only selection result for a Bayesian KL weight."""

    model: torch.nn.Module
    fit: FitResult
    selected_weight: float
    validation_nll: dict[float, float]


DEFAULT_KL_WEIGHT_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)


def _batch_objective(model, context, target, projections, config, n_train):
    draws = config.bayesian_samples if getattr(model, "is_bayesian", False) else 1
    nll_terms, cal_terms, seq_terms = [], [], []
    for _ in range(draws):
        mean, scale = model(context)
        pits = projected_pits(target, mean, scale, projections)
        nll_terms.append(gaussian_nll(target, mean, scale))
        cal_terms.append(
            calibration_loss(pits, config.calibration_grid_size, config.calibration_temperature)
        )
        seq_terms.append(sequential_calibration_loss(pits, config.sequential_lags))
    nll = torch.stack(nll_terms).mean()
    cal = torch.stack(cal_terms).mean()
    seq = torch.stack(seq_terms).mean()
    kl = model.kl_divergence() / max(1, n_train)
    total = nll + config.lambda_cal * cal + config.lambda_seq * seq + config.kl_weight * kl
    return total, nll, cal, seq, kl


def _run_epoch(model, loader, projections, config, n_train, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(5)
    count = 0
    context_manager = torch.enable_grad() if training else torch.no_grad()
    with context_manager:
        for context, target in loader:
            device = next(model.parameters()).device
            context, target = context.to(device), target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            values = _batch_objective(model, context, target, projections, config, n_train)
            if training:
                values[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            totals += np.array([value.detach().item() for value in values])
            count += 1
    return totals / max(1, count)


def fit_model(
    model,
    train_dataset,
    validation_dataset,
    config: TrainingConfig,
    projections: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> FitResult:
    """Fit one RNN variant and restore the best chronological validation checkpoint."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model.to(device)
    if projections is None:
        projections = make_projection_matrix(model.config.input_dim, seed=config.seed)
    projections = projections.to(device)
    # Ordering is retained because the sequential PIT loss is order dependent.
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history, best_state, best_loss, best_epoch, stale = [], None, float("inf"), 0, 0
    for epoch in range(1, config.epochs + 1):
        train_values = _run_epoch(
            model, train_loader, projections, config, len(train_dataset), optimizer
        )
        validation_values = _run_epoch(
            model, validation_loader, projections, config, len(train_dataset)
        )
        row = {"epoch": float(epoch)}
        for prefix, values in (("train", train_values), ("validation", validation_values)):
            for name, value in zip(
                ("loss", "nll", "calibration", "sequential", "kl"),
                values,
                strict=True,
            ):
                row[f"{prefix}_{name}"] = float(value)
        history.append(row)
        selection_index = 1 if config.selection_metric == "nll" else 0
        selection_value = float(validation_values[selection_index])
        if selection_value < best_loss - config.min_delta:
            best_loss, best_epoch = selection_value, epoch
            best_state, stale = deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if config.early_stopping and stale >= config.patience:
            break
    if config.early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    elif not config.early_stopping:
        best_epoch = len(history)
    return FitResult(history, best_epoch, len(history) < config.epochs)


def fit_bayesian_model_with_kl_selection(
    model_factory: Callable[[], torch.nn.Module],
    train_dataset,
    validation_dataset,
    config: TrainingConfig,
    kl_weights: Iterable[float] = DEFAULT_KL_WEIGHT_GRID,
    projections: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    n_validation_weight_samples: int = 50,
) -> KLWeightSelectionResult:
    """Select a Bayesian KL weight using chronological validation NLL only.

    Each candidate starts from the same random initialization and sees the same
    ordered mini-batches. Within each candidate, deterministic validation NLL
    at posterior-mean weights selects the epoch checkpoint. After that
    checkpoint is restored, Monte Carlo posterior-predictive mixture validation
    NLL selects the KL weight. The function deliberately accepts no test data,
    preventing test-set leakage. Ties favor the smaller KL weight.
    """
    weights = tuple(float(weight) for weight in kl_weights)
    if not weights:
        raise ValueError("kl_weights must contain at least one candidate")
    if len(set(weights)) != len(weights):
        raise ValueError("kl_weights must not contain duplicate candidates")
    if any(not np.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("kl_weights must be finite and nonnegative")
    if n_validation_weight_samples < 1:
        raise ValueError("n_validation_weight_samples must be positive")

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=len(validation_dataset),
        shuffle=False,
    )
    validation_context, validation_target = next(iter(validation_loader))

    candidates: dict[float, tuple[torch.nn.Module, FitResult]] = {}
    validation_nll: dict[float, float] = {}
    for weight in weights:
        # Common random numbers make the comparison depend on the KL weight,
        # rather than on different parameter initializations or batch draws.
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        model = model_factory()
        fit = fit_model(
            model,
            train_dataset,
            validation_dataset,
            replace(config, kl_weight=weight, selection_metric="nll"),
            projections,
            device,
        )
        if fit.best_epoch < 1 or fit.best_epoch > len(fit.history):
            raise RuntimeError("fit returned an invalid best epoch")
        # Common posterior-sampling random numbers reduce selection noise.
        torch.manual_seed(config.seed + 30_000)
        score = predictive_nll(
            model,
            validation_context,
            validation_target,
            n_weight_samples=n_validation_weight_samples,
        )
        if not np.isfinite(score):
            raise RuntimeError(f"validation NLL is not finite for KL weight {weight:g}")
        candidates[weight] = (model, fit)
        validation_nll[weight] = score

    selected_weight = min(weights, key=lambda weight: (validation_nll[weight], weight))
    selected_model, selected_fit = candidates[selected_weight]
    return KLWeightSelectionResult(
        model=selected_model,
        fit=selected_fit,
        selected_weight=selected_weight,
        validation_nll=validation_nll,
    )
