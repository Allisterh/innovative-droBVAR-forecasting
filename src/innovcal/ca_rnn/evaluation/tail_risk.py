"""Sample-based portfolio Value at Risk and Expected Shortfall evaluation."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from innovcal.evaluation.uncertainty import moving_block_indices


def _validate_inputs(
    target: np.ndarray,
    samples: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if target.ndim != 2 or samples.ndim != 3 or samples.shape[1:] != target.shape:
        raise ValueError("target and samples require shapes (origin, asset) and (draw, origin, asset)")
    if not np.isfinite(target).all() or not np.isfinite(samples).all():
        raise ValueError("target and samples must be finite")
    if weights is None:
        weights = np.full(target.shape[1], 1.0 / target.shape[1])
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (target.shape[1],) or not np.isfinite(weights).all():
        raise ValueError("weights must be a finite vector with one entry per asset")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("portfolio weights must sum to one")
    return target, samples, weights


def portfolio_tail_forecasts(
    target: np.ndarray,
    samples: np.ndarray,
    *,
    tail_probability: float = 0.05,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return origin-level loss, VaR, ES, exceedance, and proper score values.

    Returns are converted to losses using ``loss = -weights @ return``. VaR is
    the upper ``1-tail_probability`` loss quantile and ES averages simulated
    losses at or above that forecast VaR. ``joint_var_es_score`` is the FZ0
    joint score; lower quantile and joint scores are better.
    """
    if not 0 < tail_probability < 0.5:
        raise ValueError("tail_probability must lie between zero and 0.5")
    target, samples, weights = _validate_inputs(target, samples, weights)
    realized_loss = -(target @ weights)
    simulated_loss = -np.einsum("sod,d->so", samples, weights)
    confidence = 1.0 - tail_probability
    value_at_risk = np.quantile(simulated_loss, confidence, axis=0)
    tail_mask = simulated_loss >= value_at_risk[None, :]
    expected_shortfall = np.sum(simulated_loss * tail_mask, axis=0) / np.maximum(
        tail_mask.sum(axis=0), 1
    )
    exceedance = realized_loss > value_at_risk
    error = realized_loss - value_at_risk
    quantile_loss = (confidence - (error < 0).astype(float)) * error

    # FZ0 is defined for a positive upper-tail ES. This holds for the reported
    # standardized-return application; fail explicitly rather than silently
    # shifting the loss scale if it does not.
    if np.any(expected_shortfall <= 0):
        raise ValueError("FZ0 requires positive upper-tail Expected Shortfall")
    joint_score = (
        np.log(expected_shortfall)
        + value_at_risk / expected_shortfall
        + exceedance * (realized_loss - value_at_risk)
        / (tail_probability * expected_shortfall)
        - 1.0
    )
    return pd.DataFrame(
        {
            "realized_loss": realized_loss,
            "value_at_risk": value_at_risk,
            "expected_shortfall": expected_shortfall,
            "exceedance": exceedance.astype(int),
            "quantile_loss": quantile_loss,
            "joint_var_es_score": joint_score,
        }
    )


def _bernoulli_log_likelihood(n0: int, n1: int, probability: float) -> float:
    probability = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
    return n0 * np.log1p(-probability) + n1 * np.log(probability)


def _coverage_tests(exceedance: np.ndarray, tail_probability: float) -> dict[str, float]:
    hit = np.asarray(exceedance, dtype=int)
    n1, n0 = int(hit.sum()), int(len(hit) - hit.sum())
    observed = n1 / len(hit)
    lr_uc = max(
        0.0,
        2.0
        * (
            _bernoulli_log_likelihood(n0, n1, observed)
            - _bernoulli_log_likelihood(n0, n1, tail_probability)
        ),
    )
    transitions = np.zeros((2, 2), dtype=int)
    for previous, current in zip(hit[:-1], hit[1:], strict=True):
        transitions[previous, current] += 1
    n00, n01 = transitions[0]
    n10, n11 = transitions[1]
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    pooled = (n01 + n11) / max(transitions.sum(), 1)
    independent = _bernoulli_log_likelihood(n00 + n10, n01 + n11, pooled)
    markov = _bernoulli_log_likelihood(n00, n01, p01) + _bernoulli_log_likelihood(
        n10, n11, p11
    )
    lr_ind = max(0.0, 2.0 * (markov - independent))
    return {
        "kupiec_p_value": float(stats.chi2.sf(lr_uc, 1)),
        "independence_p_value": float(stats.chi2.sf(lr_ind, 1)),
        "conditional_coverage_p_value": float(stats.chi2.sf(lr_uc + lr_ind, 2)),
    }


def summarize_tail_risk(
    target: np.ndarray,
    samples: np.ndarray,
    *,
    tail_probability: float = 0.05,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize portfolio tail forecasts and descriptive coverage tests."""
    frame = portfolio_tail_forecasts(
        target, samples, tail_probability=tail_probability, weights=weights
    )
    hit = frame["exceedance"].to_numpy()
    tail_losses = frame.loc[frame["exceedance"].astype(bool), "realized_loss"]
    result = {
        "n_origins": float(len(frame)),
        "n_exceedances": float(hit.sum()),
        "exceedance_rate": float(hit.mean()),
        "exceedance_rate_error": float(abs(hit.mean() - tail_probability)),
        "mean_value_at_risk": float(frame["value_at_risk"].mean()),
        "mean_expected_shortfall": float(frame["expected_shortfall"].mean()),
        "mean_exceedance_loss": float(tail_losses.mean()) if len(tail_losses) else np.nan,
        "mean_quantile_loss": float(frame["quantile_loss"].mean()),
        "mean_joint_var_es_score": float(frame["joint_var_es_score"].mean()),
    }
    result.update(_coverage_tests(hit, tail_probability))
    return result


def paired_tail_risk_bootstrap(
    target: np.ndarray,
    forecast_samples: Mapping[str, np.ndarray],
    comparisons: Sequence[tuple[str, str]],
    *,
    tail_probability: float = 0.05,
    weights: np.ndarray | None = None,
    block_length: int = 20,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 1313,
) -> pd.DataFrame:
    """Paired moving-block intervals for additive VaR and joint VaR--ES scores."""
    if n_bootstrap < 2 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    missing = {name for pair in comparisons for name in pair if name not in forecast_samples}
    if missing:
        raise ValueError(f"missing forecasts for models: {sorted(missing)}")
    frames = {
        name: portfolio_tail_forecasts(
            target, samples, tail_probability=tail_probability, weights=weights
        )
        for name, samples in forecast_samples.items()
    }
    rng = np.random.default_rng(seed)
    indices = [
        moving_block_indices(len(target), block_length, rng) for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - confidence
    rows: list[dict[str, float | str]] = []
    for candidate, comparator in comparisons:
        for metric in ("quantile_loss", "joint_var_es_score"):
            difference = frames[candidate][metric].to_numpy() - frames[comparator][
                metric
            ].to_numpy()
            draws = np.array([difference[index].mean() for index in indices])
            lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
            rows.append(
                {
                    "candidate": candidate,
                    "comparator": comparator,
                    "metric": metric,
                    "difference": float(difference.mean()),
                    "bootstrap_se": float(draws.std(ddof=1)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "probability_difference_below_zero": float(np.mean(draws < 0)),
                }
            )
    frame = pd.DataFrame(rows)
    frame.insert(0, "n_origins", len(target))
    frame.insert(1, "block_length", block_length)
    frame.insert(2, "n_bootstrap", n_bootstrap)
    return frame


def paired_seed_tail_risk_bootstrap(
    target: np.ndarray,
    candidate_samples: np.ndarray,
    comparator_samples: np.ndarray,
    *,
    candidate_name: str = "CA-RNN",
    comparator_name: str = "VAR-GARCH bootstrap",
    tail_probability: float = 0.05,
    weights: np.ndarray | None = None,
    block_length: int = 20,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 1313,
) -> pd.DataFrame:
    """Hierarchical paired intervals over neural seeds and forecast origins.

    ``candidate_samples`` has shape ``(seed, draw, origin, asset)``. The fixed
    comparator has shape ``(draw, origin, asset)`` and is compared with every
    neural seed, matching the thesis' conditional-on-baseline estimand.
    """
    candidate_samples = np.asarray(candidate_samples, dtype=float)
    comparator_samples = np.asarray(comparator_samples, dtype=float)
    if candidate_samples.ndim != 4 or comparator_samples.ndim != 3:
        raise ValueError("candidate and comparator require four and three dimensions")
    if candidate_samples.shape[2:] != comparator_samples.shape[1:]:
        raise ValueError("candidate and comparator forecast origins must match")
    if candidate_samples.shape[2:] != np.asarray(target).shape:
        raise ValueError("forecast origins must match target")
    if candidate_samples.shape[0] < 2:
        raise ValueError("at least two candidate optimization seeds are required")
    if n_bootstrap < 2 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")

    candidate_frames = [
        portfolio_tail_forecasts(
            target, value, tail_probability=tail_probability, weights=weights
        )
        for value in candidate_samples
    ]
    comparator_frame = portfolio_tail_forecasts(
        target, comparator_samples, tail_probability=tail_probability, weights=weights
    )
    n_seeds = len(candidate_frames)
    n_origins = len(target)
    rng = np.random.default_rng(seed)
    seed_indices = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    block_indices = [
        [moving_block_indices(n_origins, block_length, rng) for _ in range(n_seeds)]
        for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - confidence
    rows: list[dict[str, float | str]] = []
    for metric in ("quantile_loss", "joint_var_es_score"):
        differences = [
            frame[metric].to_numpy() - comparator_frame[metric].to_numpy()
            for frame in candidate_frames
        ]
        observed = float(np.mean([value.mean() for value in differences]))
        draws = np.array(
            [
                np.mean(
                    [
                        differences[selected_seed][index].mean()
                        for selected_seed, index in zip(
                            seed_indices[b], block_indices[b], strict=True
                        )
                    ]
                )
                for b in range(n_bootstrap)
            ]
        )
        lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
        rows.append(
            {
                "candidate": candidate_name,
                "comparator": comparator_name,
                "metric": metric,
                "difference": observed,
                "bootstrap_se": float(draws.std(ddof=1)),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "probability_difference_below_zero": float(np.mean(draws < 0)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.insert(0, "n_seeds", n_seeds)
    frame.insert(1, "n_origins", n_origins)
    frame.insert(2, "block_length", block_length)
    frame.insert(3, "n_bootstrap", n_bootstrap)
    return frame
