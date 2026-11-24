"""Leakage-safe conventional baselines for multivariate return forecasts.

The strongest baseline combines a linear VAR conditional mean, independently
filtered marginal GARCH(1,1) scales, and joint resampling of standardized residual
vectors. Joint resampling preserves the empirical contemporaneous dependence and
tail shape without imposing a Gaussian copula.
"""

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, jarque_bera, kurtosis, skew


def _series(values: np.ndarray, minimum_rows: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < minimum_rows:
        raise ValueError("values must be a sufficiently long time-by-dimension array")
    if not np.isfinite(values).all():
        raise ValueError("values contain non-finite entries")
    return values


@dataclass(frozen=True)
class VARModel:
    """OLS VAR with an intercept and lag order ``p``."""

    intercept: np.ndarray
    coefficients: np.ndarray  # (p, response dimension, predictor dimension)
    residuals: np.ndarray

    @property
    def lag(self) -> int:
        return int(self.coefficients.shape[0])

    @classmethod
    def fit(cls, values: np.ndarray, lag: int = 1) -> "VARModel":
        values = _series(values)
        if lag < 1 or len(values) <= lag + 1:
            raise ValueError("lag must be positive with enough observations")
        rows = []
        for t in range(lag, len(values)):
            rows.append(np.concatenate(([1.0], values[t - lag : t][::-1].ravel())))
        design = np.asarray(rows)
        target = values[lag:]
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted = np.einsum("ni,ij->nj", design, beta)
        coefficients = beta[1:].reshape(
            lag, values.shape[1], values.shape[1]
        ).transpose(0, 2, 1)
        return cls(beta[0], coefficients, target - fitted)

    def predict(self, history: np.ndarray) -> np.ndarray:
        history = _series(history, minimum_rows=self.lag)
        if history.shape[1] != len(self.intercept):
            raise ValueError("history dimension does not match fitted VAR")
        lags = history[-self.lag :][::-1]
        return self.intercept + np.einsum("pij,pj->i", self.coefficients, lags)


@dataclass(frozen=True)
class MarginalGARCH:
    """Gaussian quasi-maximum-likelihood marginal GARCH(1,1) filters."""

    omega: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    initial_variance: np.ndarray
    converged: np.ndarray

    @classmethod
    def fit(cls, residuals: np.ndarray) -> "MarginalGARCH":
        residuals = _series(residuals, minimum_rows=10)
        parameters, convergence, starts = [], [], []
        for marginal in residuals.T:
            column = marginal.copy()
            variance = max(float(np.var(column, ddof=1)), 1e-8)

            def objective(
                parameter: np.ndarray,
                observations: np.ndarray = column,
                starting_variance: float = variance,
            ) -> float:
                omega, alpha, beta = parameter
                conditional = np.empty(len(observations))
                conditional[0] = starting_variance
                for t in range(1, len(observations)):
                    conditional[t] = omega + alpha * observations[t - 1] ** 2 + beta * conditional[t - 1]
                conditional = np.maximum(conditional, 1e-10)
                return float(
                    0.5
                    * np.sum(np.log(conditional) + observations**2 / conditional)
                )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Values in x were outside bounds during a minimize step",
                    category=RuntimeWarning,
                    module=r"scipy\.optimize\._slsqp_py",
                )
                result = minimize(
                    objective,
                    x0=np.array([0.05 * variance, 0.05, 0.90]),
                    method="SLSQP",
                    bounds=((1e-10, 10.0 * variance), (0.0, 0.999), (0.0, 0.999)),
                    constraints={
                        "type": "ineq",
                        "fun": lambda x: 0.999 - x[1] - x[2],
                    },
                    options={"maxiter": 1_000, "ftol": 1e-10},
                )
            if not result.success or not np.isfinite(result.fun):
                # A transparent, stable fallback rather than dropping a run.
                fitted = np.array([0.05 * variance, 0.05, 0.90])
                convergence.append(False)
            else:
                fitted = result.x
                convergence.append(True)
            parameters.append(fitted)
            starts.append(variance)
        array = np.asarray(parameters)
        return cls(array[:, 0], array[:, 1], array[:, 2], np.asarray(starts), np.asarray(convergence))

    def filter(self, residuals: np.ndarray) -> np.ndarray:
        """Return variances measurable immediately before each residual."""
        residuals = _series(residuals)
        if residuals.shape[1] != len(self.omega):
            raise ValueError("residual dimension does not match fitted GARCH")
        variance = np.empty_like(residuals)
        variance[0] = self.initial_variance
        for t in range(1, len(residuals)):
            variance[t] = (
                self.omega
                + self.alpha * residuals[t - 1] ** 2
                + self.beta * variance[t - 1]
            )
        return np.maximum(variance, 1e-10)


@dataclass(frozen=True)
class BaselineForecasts:
    selected_lag: int
    validation_mse: dict[int, float]
    samples: dict[str, np.ndarray]
    target: np.ndarray
    garch_converged: np.ndarray


@dataclass(frozen=True)
class GaussianVARForecast:
    """Exact one-step Gaussian VAR distribution on a held-out block."""

    model: VARModel
    validation_mse: dict[int, float]
    mean: np.ndarray
    covariance: np.ndarray
    target: np.ndarray

    @property
    def selected_lag(self) -> int:
        return self.model.lag


@dataclass(frozen=True)
class BaselineDiagnostics:
    """Training-sample specification checks for the fitted VAR--GARCH model."""

    var_spectral_radius: float
    var_stable: bool
    garch: MarginalGARCH
    standardized_residuals: np.ndarray
    standardized_residual_correlation: np.ndarray
    ljung_box_lags: np.ndarray
    residual_ljung_box_pvalues: np.ndarray
    squared_residual_ljung_box_pvalues: np.ndarray
    residual_skewness: np.ndarray
    residual_kurtosis: np.ndarray
    jarque_bera_pvalues: np.ndarray
    variance_clipping_rates: np.ndarray


def _ljung_box_pvalues(values: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Ljung--Box p-values for one series without an optional dependency."""
    centered = np.asarray(values, dtype=float) - np.mean(values)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denominator = float(centered @ centered)
    output = np.full(len(lags), np.nan)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return output
    n = len(centered)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        correlations = np.asarray(
            [
                centered[k:] @ centered[:-k] / denominator
                for k in range(1, int(lags.max()) + 1)
            ]
        )
    for index, lag in enumerate(lags):
        available = np.arange(1, min(int(lag), n - 1) + 1)
        if len(available) == 0:
            continue
        statistic = n * (n + 2) * np.sum(
            correlations[available - 1] ** 2 / (n - available)
        )
        output[index] = float(chi2.sf(statistic, df=len(available)))
    return output


def var_spectral_radius(model: VARModel) -> float:
    """Maximum modulus of the eigenvalues of a VAR companion matrix."""
    lag, dimension, _ = model.coefficients.shape
    companion = np.zeros((lag * dimension, lag * dimension))
    companion[:dimension] = np.concatenate(model.coefficients, axis=1)
    if lag > 1:
        companion[dimension:, :-dimension] = np.eye((lag - 1) * dimension)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def diagnose_var_garch(
    train: np.ndarray,
    lag: int,
    *,
    ljung_box_lags: Iterable[int] = (5, 10, 20),
) -> BaselineDiagnostics:
    """Fit the benchmark on training data and return specification diagnostics.

    The Ljung--Box checks are descriptive in-sample diagnostics. Their p-values
    do not adjust for VAR or GARCH parameter estimation and should not be read as
    exact finite-sample tests.
    """
    train = _series(train, minimum_rows=max(10, lag + 2))
    lags = np.asarray(sorted(set(int(value) for value in ljung_box_lags)))
    if len(lags) == 0 or np.any(lags < 1):
        raise ValueError("ljung_box_lags must contain positive integers")

    model = VARModel.fit(train, lag)
    garch = MarginalGARCH.fit(model.residuals)
    raw_variances = np.empty_like(model.residuals)
    raw_variances[0] = garch.initial_variance
    for t in range(1, len(model.residuals)):
        raw_variances[t] = (
            garch.omega
            + garch.alpha * model.residuals[t - 1] ** 2
            + garch.beta * raw_variances[t - 1]
        )
    clipping_rates = np.mean(raw_variances <= 1e-10, axis=0)
    variances = np.maximum(raw_variances, 1e-10)
    standardized = model.residuals / np.sqrt(variances)
    standardized -= standardized.mean(axis=0)

    residual_pvalues = np.column_stack(
        [_ljung_box_pvalues(standardized[:, j], lags) for j in range(train.shape[1])]
    )
    squared_pvalues = np.column_stack(
        [
            _ljung_box_pvalues(standardized[:, j] ** 2, lags)
            for j in range(train.shape[1])
        ]
    )
    jb_pvalues = np.asarray(
        [jarque_bera(standardized[:, j]).pvalue for j in range(train.shape[1])]
    )
    radius = var_spectral_radius(model)
    return BaselineDiagnostics(
        var_spectral_radius=radius,
        var_stable=bool(radius < 1.0),
        garch=garch,
        standardized_residuals=standardized,
        standardized_residual_correlation=np.corrcoef(standardized, rowvar=False),
        ljung_box_lags=lags,
        residual_ljung_box_pvalues=residual_pvalues,
        squared_residual_ljung_box_pvalues=squared_pvalues,
        residual_skewness=skew(standardized, axis=0, bias=False),
        residual_kurtosis=kurtosis(standardized, axis=0, fisher=False, bias=False),
        jarque_bera_pvalues=jb_pvalues,
        variance_clipping_rates=clipping_rates,
    )


def _one_step_means(model: VARModel, prefix: np.ndarray, targets: np.ndarray) -> np.ndarray:
    observed = np.concatenate([prefix, targets], axis=0)
    offset = len(prefix)
    return np.asarray([model.predict(observed[:t]) for t in range(offset, len(observed))])


def select_var_lag(
    train: np.ndarray,
    validation: np.ndarray,
    candidates: Iterable[int] = (1, 5, 10, 20),
) -> tuple[int, dict[int, float]]:
    """Select lag by validation one-step MSE without refitting on validation."""
    train, validation = _series(train), _series(validation)
    scores: dict[int, float] = {}
    for lag in dict.fromkeys(int(value) for value in candidates):
        if lag < 1 or lag >= len(train) - 1:
            continue
        model = VARModel.fit(train, lag)
        means = _one_step_means(model, train[-lag:], validation)
        scores[lag] = float(np.mean((means - validation) ** 2))
    if not scores:
        raise ValueError("no valid VAR lag candidates")
    return min(scores, key=scores.get), scores


def forecast_gaussian_var(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    *,
    lag_candidates: Iterable[int] = (1, 5, 10, 20),
) -> GaussianVARForecast:
    """Return the exact fixed-covariance Gaussian VAR predictive law.

    Lag selection uses validation one-step MSE. The selected model and its
    innovation covariance are fitted on the training block only. Test means are
    one-step forecasts that may condition on earlier observed test outcomes.
    """
    train, validation, test = _series(train), _series(validation), _series(test)
    if train.shape[1] != validation.shape[1] or train.shape[1] != test.shape[1]:
        raise ValueError("all partitions must have the same dimension")
    lag, scores = select_var_lag(train, validation, lag_candidates)
    model = VARModel.fit(train, lag)
    covariance = np.cov(model.residuals, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance) + np.eye(train.shape[1]) * 1e-8
    prefix = np.concatenate([train, validation], axis=0)
    means = _one_step_means(model, prefix[-lag:], test)
    return GaussianVARForecast(model, scores, means, covariance, test.copy())


def forecast_baselines(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    *,
    lag_candidates: Iterable[int] = (1, 5, 10, 20),
    n_samples: int = 500,
    seed: int = 123,
) -> BaselineForecasts:
    """Produce matched one-step samples for three increasingly dynamic baselines."""
    train, validation, test = _series(train), _series(validation), _series(test)
    if train.shape[1] != validation.shape[1] or train.shape[1] != test.shape[1]:
        raise ValueError("all partitions must have the same dimension")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")

    gaussian = forecast_gaussian_var(
        train, validation, test, lag_candidates=lag_candidates
    )
    lag = gaussian.selected_lag
    scores = gaussian.validation_mse
    model = gaussian.model
    rng = np.random.default_rng(seed)
    cases, dimension = test.shape

    # Unconditional historical bootstrap: no dynamic mean or volatility model.
    centered_train = train - train.mean(axis=0)
    indices = rng.integers(0, len(centered_train), size=(n_samples, cases))
    historical = centered_train[indices]

    # Fixed-covariance Gaussian VAR.
    covariance = gaussian.covariance
    means = gaussian.mean
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-10)))
    gaussian_noise = np.einsum(
        "scd,ed->sce",
        rng.normal(size=(n_samples, cases, dimension)),
        covariance_root,
    )
    var_gaussian = means[None] + gaussian_noise

    # Fit marginal scales on training residuals and filter all subsequent observed
    # residuals. Variance at target t depends only on residuals through t-1.
    garch = MarginalGARCH.fit(model.residuals)
    full = np.concatenate([train, validation, test], axis=0)
    full_means = np.full_like(full, np.nan)
    for t in range(lag, len(full)):
        full_means[t] = model.predict(full[:t])
    full_residuals = full[lag:] - full_means[lag:]
    variances = garch.filter(full_residuals)
    train_count = len(train) - lag
    standardized_train = model.residuals / np.sqrt(variances[:train_count])
    standardized_train -= standardized_train.mean(axis=0)
    test_start = train_count + len(validation)
    test_scales = np.sqrt(variances[test_start : test_start + cases])
    bootstrap_indices = rng.integers(0, len(standardized_train), size=(n_samples, cases))
    joint_innovations = standardized_train[bootstrap_indices]
    var_garch = means[None] + test_scales[None] * joint_innovations

    return BaselineForecasts(
        selected_lag=lag,
        validation_mse=scores,
        samples={
            "Historical bootstrap": historical,
            "VAR Gaussian": var_gaussian,
            "VAR-GARCH bootstrap": var_garch,
        },
        target=test.copy(),
        garch_converged=garch.converged,
    )
