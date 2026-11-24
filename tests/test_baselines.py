import numpy as np

from innovcal.baselines import (
    MarginalGARCH,
    VARModel,
    diagnose_var_garch,
    forecast_baselines,
    forecast_gaussian_var,
    select_var_lag,
    var_spectral_radius,
)


def _var_data(seed: int = 10, length: int = 240, dimension: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.zeros((length, dimension))
    for t in range(1, length):
        values[t] = 0.35 * values[t - 1] + rng.normal(scale=0.7, size=dimension)
    return values


def test_var_fit_and_prediction_are_finite():
    values = _var_data()
    model = VARModel.fit(values, lag=2)
    assert model.residuals.shape == (238, 2)
    assert np.isfinite(model.predict(values)).all()


def test_var_coefficient_orientation_recovers_cross_effects():
    rng = np.random.default_rng(4)
    transition = np.array([[0.20, 0.55], [-0.35, 0.10]])
    values = np.zeros((2_000, 2))
    for t in range(1, len(values)):
        values[t] = transition @ values[t - 1] + rng.normal(scale=0.05, size=2)
    fitted = VARModel.fit(values, lag=1)
    assert np.allclose(fitted.coefficients[0], transition, atol=0.03)


def test_garch_filter_is_positive_and_predetermined():
    residuals = _var_data(length=180)
    model = MarginalGARCH.fit(residuals)
    first = model.filter(residuals)
    altered = residuals.copy()
    altered[-1] *= 100
    second = model.filter(altered)
    assert np.all(first > 0)
    # The variance paired with the final residual cannot depend on that residual.
    assert np.allclose(first[-1], second[-1])


def test_baseline_forecasts_have_matched_shapes_and_are_reproducible():
    values = _var_data(length=300)
    train, validation, test = values[:180], values[180:240], values[240:]
    first = forecast_baselines(train, validation, test, n_samples=40, seed=8)
    second = forecast_baselines(train, validation, test, n_samples=40, seed=8)
    assert first.selected_lag in {1, 5, 10, 20}
    assert first.target.shape == test.shape
    for name, samples in first.samples.items():
        assert samples.shape == (40, 60, 2)
        assert np.isfinite(samples).all()
        assert np.allclose(samples, second.samples[name])


def test_gaussian_var_returns_exact_distribution_parameters():
    values = _var_data(length=300)
    train, validation, test = values[:180], values[180:240], values[240:]
    forecast = forecast_gaussian_var(train, validation, test)
    assert forecast.selected_lag in {1, 5, 10, 20}
    assert forecast.mean.shape == test.shape
    assert forecast.covariance.shape == (2, 2)
    assert np.allclose(forecast.covariance, forecast.covariance.T)
    assert np.linalg.eigvalsh(forecast.covariance).min() > 0.0
    assert np.isfinite(forecast.mean).all()


def test_var_lag_selection_uses_validation_scores():
    values = _var_data(length=220)
    lag, scores = select_var_lag(values[:140], values[140:], (1, 3))
    assert lag == min(scores, key=scores.get)
    assert set(scores) == {1, 3}


def test_var_garch_diagnostics_are_finite_and_well_shaped():
    values = _var_data(length=300)
    model = VARModel.fit(values, lag=1)
    diagnostics = diagnose_var_garch(values, lag=1, ljung_box_lags=(5, 10))
    assert np.isclose(diagnostics.var_spectral_radius, var_spectral_radius(model))
    assert diagnostics.var_stable
    assert diagnostics.residual_ljung_box_pvalues.shape == (2, 2)
    assert diagnostics.squared_residual_ljung_box_pvalues.shape == (2, 2)
    assert diagnostics.standardized_residual_correlation.shape == (2, 2)
    assert np.isfinite(diagnostics.standardized_residuals).all()
    assert np.all((diagnostics.variance_clipping_rates >= 0.0) & (diagnostics.variance_clipping_rates <= 1.0))
