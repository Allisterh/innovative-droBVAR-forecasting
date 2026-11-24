import numpy as np
import torch

from innovcal.ca_rnn import (
    DEFAULT_KL_WEIGHT_GRID,
    CARNNConfig,
    FitResult,
    KLWeightSelectionResult,
    TrainingConfig,
)
from innovcal.experiments import comparison


def test_four_model_comparison_selects_one_common_bayesian_kl_weight(monkeypatch):
    selected_model = None
    fit_calls = []
    selection_configs = []

    def fake_selection(
        model_factory,
        train_dataset,
        validation_dataset,
        config,
        kl_weights,
        projections=None,
        device="cpu",
    ):
        nonlocal selected_model
        del train_dataset, validation_dataset, projections, device
        selection_configs.append(config)
        selected_model = model_factory()
        scores = {float(weight): abs(float(weight) - 0.1) for weight in kl_weights}
        return KLWeightSelectionResult(
            selected_model,
            FitResult([{"epoch": 1.0, "validation_nll": 1.0}], 1, False),
            0.1,
            scores,
        )

    def fake_fit(
        model,
        train_dataset,
        validation_dataset,
        config,
        projections=None,
        device="cpu",
    ):
        del train_dataset, validation_dataset, projections, device
        fit_calls.append(
            (type(model).__name__, config.lambda_cal, config.lambda_seq, config.kl_weight)
        )
        return FitResult([{"epoch": 1.0, "validation_nll": 1.0}], 1, False)

    def fake_samples(model, context, n_samples=500):
        del model
        return torch.zeros(n_samples, len(context), context.shape[-1])

    monkeypatch.setattr(
        comparison, "fit_bayesian_model_with_kl_selection", fake_selection
    )
    monkeypatch.setattr(comparison, "fit_model", fake_fit)
    monkeypatch.setattr(comparison, "predictive_nll", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(comparison, "sample_forecasts", fake_samples)
    monkeypatch.setattr(comparison, "evaluate_samples", lambda *args, **kwargs: {"rmse": 0.0})

    result = comparison.run_four_model_comparison(
        np.zeros((80, 2), dtype=np.float32),
        history=5,
        model_config=CARNNConfig(2, hidden_dim=4),
        training_config=TrainingConfig(epochs=1),
        lambda_cal=10.0,
        lambda_seq=20.0,
        n_forecast_samples=4,
        bayesian_kl_grid=DEFAULT_KL_WEIGHT_GRID,
    )

    assert DEFAULT_KL_WEIGHT_GRID == (0.0001, 0.001, 0.01, 0.1, 1.0)
    assert result.selected_kl_weight == 0.1
    assert result.models["BRNN"] is selected_model
    assert result.kl_selection is not None
    assert result.kl_selection["selected"].sum() == 1
    assert selection_configs[0].lambda_cal == 0.0
    assert selection_configs[0].lambda_seq == 0.0

    bayesian_calibration_fits = [
        call for call in fit_calls if call[0] == "BayesianCARNN"
    ]
    assert bayesian_calibration_fits == [("BayesianCARNN", 10.0, 20.0, 0.1)]

