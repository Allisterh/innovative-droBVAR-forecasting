"""Models and objectives for calibration-aware recurrent forecasting."""

from innovcal.ca_rnn.config import CARNNConfig, TrainingConfig
from innovcal.ca_rnn.losses import (
    calibration_loss,
    gaussian_nll,
    projected_pits,
    sequential_calibration_loss,
)
from innovcal.ca_rnn.model import CARNN, BayesianCARNN
from innovcal.ca_rnn.predict import (
    predict_distribution,
    predictive_nll,
    sample_forecasts,
)
from innovcal.ca_rnn.trainer import (
    DEFAULT_KL_WEIGHT_GRID,
    FitResult,
    KLWeightSelectionResult,
    fit_bayesian_model_with_kl_selection,
    fit_model,
)

__all__ = [
    "BayesianCARNN",
    "CARNN",
    "CARNNConfig",
    "DEFAULT_KL_WEIGHT_GRID",
    "FitResult",
    "KLWeightSelectionResult",
    "TrainingConfig",
    "calibration_loss",
    "fit_bayesian_model_with_kl_selection",
    "fit_model",
    "gaussian_nll",
    "predict_distribution",
    "predictive_nll",
    "projected_pits",
    "sample_forecasts",
    "sequential_calibration_loss",
]
