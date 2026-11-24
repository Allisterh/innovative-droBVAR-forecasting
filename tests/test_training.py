import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from innovcal.ca_rnn import (
    CARNN,
    CARNNConfig,
    FitResult,
    TrainingConfig,
    fit_bayesian_model_with_kl_selection,
    fit_model,
)
from innovcal.data import WindowDataset


def test_training_smoke_run_updates_and_restores_model():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(100, 2)).astype(np.float32)
    train = WindowDataset(values[:70], history=8)
    validation = WindowDataset(values[70:], history=8)
    model = CARNN(CARNNConfig(input_dim=2, hidden_dim=6))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = fit_model(
        model,
        train,
        validation,
        TrainingConfig(epochs=2, batch_size=16, patience=2, lambda_cal=1.0),
    )
    assert len(result.history) == 2
    assert 1 <= result.best_epoch <= 2
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_fixed_epoch_mode_runs_full_budget_and_selects_final_model():
    rng = np.random.default_rng(12)
    values = rng.normal(size=(80, 2)).astype(np.float32)
    model = CARNN(CARNNConfig(input_dim=2, hidden_dim=4))
    result = fit_model(
        model,
        WindowDataset(values[:55], history=6),
        WindowDataset(values[55:], history=6),
        TrainingConfig(epochs=3, batch_size=16, patience=1, early_stopping=False),
    )
    assert len(result.history) == 3
    assert result.best_epoch == 3
    assert not result.stopped_early


def test_kl_weight_selection_uses_only_validation_nll_and_common_initialization(
    monkeypatch,
):
    validation_scores = {0.001: 2.0, 0.01: 1.5, 0.1: 1.7}
    seen_configs = []
    initializations = []

    class DummyBayesianModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.randn(()))
            initializations.append(float(self.value.detach()))

    def fake_fit_model(
        model,
        train_dataset,
        validation_dataset,
        config,
        projections=None,
        device="cpu",
    ):
        del train_dataset, validation_dataset, projections, device
        seen_configs.append(config)
        model.validation_score = validation_scores[config.kl_weight]
        return FitResult(
            history=[
                {
                    "epoch": 1.0,
                    "validation_nll": validation_scores[config.kl_weight],
                    # A contradictory objective verifies that selection uses NLL.
                    "validation_loss": -validation_scores[config.kl_weight],
                }
            ],
            best_epoch=1,
            stopped_early=False,
        )

    def fake_predictive_nll(model, context, target, n_weight_samples=50):
        assert context.shape == (3, 2, 1)
        assert target.shape == (3, 1)
        assert n_weight_samples == 50
        return model.validation_score

    monkeypatch.setattr("innovcal.ca_rnn.trainer.fit_model", fake_fit_model)
    monkeypatch.setattr("innovcal.ca_rnn.trainer.predictive_nll", fake_predictive_nll)
    validation = TensorDataset(torch.zeros(3, 2, 1), torch.zeros(3, 1))
    result = fit_bayesian_model_with_kl_selection(
        DummyBayesianModel,
        object(),
        validation,
        TrainingConfig(epochs=1, seed=91, selection_metric="objective"),
        kl_weights=(0.001, 0.01, 0.1),
    )

    assert result.selected_weight == 0.01
    assert result.validation_nll == validation_scores
    assert all(config.selection_metric == "nll" for config in seen_configs)
    assert [config.kl_weight for config in seen_configs] == [0.001, 0.01, 0.1]
    assert initializations == [initializations[0]] * 3


@pytest.mark.parametrize("grid", [(), (0.01, 0.01), (-0.1,), (float("nan"),)])
def test_kl_weight_selection_rejects_invalid_grids(grid):
    with pytest.raises(ValueError):
        fit_bayesian_model_with_kl_selection(
            lambda: torch.nn.Linear(1, 1),
            object(),
            TensorDataset(torch.zeros(1, 1, 1), torch.zeros(1, 1)),
            TrainingConfig(epochs=1),
            kl_weights=grid,
        )
