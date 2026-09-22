import numpy as np

from innovcal.ca_rnn.evaluation import (
    paired_seed_tail_risk_bootstrap,
    paired_tail_risk_bootstrap,
    portfolio_tail_forecasts,
    summarize_tail_risk,
)


def test_portfolio_tail_forecasts_are_finite_and_ordered():
    rng = np.random.default_rng(13)
    target = rng.normal(size=(120, 3))
    samples = rng.normal(size=(400, 120, 3))
    frame = portfolio_tail_forecasts(target, samples)
    assert len(frame) == len(target)
    assert np.isfinite(frame.to_numpy()).all()
    assert np.all(frame["expected_shortfall"] >= frame["value_at_risk"])


def test_tail_summary_and_paired_identity():
    rng = np.random.default_rng(14)
    target = rng.normal(size=(100, 2))
    samples = rng.normal(size=(300, 100, 2))
    summary = summarize_tail_risk(target, samples)
    assert 0 <= summary["exceedance_rate"] <= 1
    result = paired_tail_risk_bootstrap(
        target,
        {"candidate": samples, "comparator": samples.copy()},
        [("candidate", "comparator")],
        n_bootstrap=50,
    )
    assert np.allclose(result["difference"], 0.0)
    assert np.allclose(result["ci_lower"], 0.0)
    assert np.allclose(result["ci_upper"], 0.0)


def test_seed_tail_bootstrap_identity():
    rng = np.random.default_rng(15)
    target = rng.normal(size=(80, 2))
    samples = rng.normal(size=(250, 80, 2))
    candidates = np.repeat(samples[None], 3, axis=0)
    result = paired_seed_tail_risk_bootstrap(
        target, candidates, samples, n_bootstrap=40
    )
    assert np.allclose(result["difference"], 0.0)
    assert np.allclose(result["bootstrap_se"], 0.0)
