import numpy as np

from innovcal.ca_rnn.evaluation import (
    paired_matched_seed_tail_risk_bootstrap,
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
        target,
        candidates,
        samples,
        n_bootstrap=40,
        n_draw_resamples=20,
        seed=16,
        draw_seed=17,
    )
    assert np.allclose(result["difference"], 0.0)
    assert np.allclose(result["bootstrap_se"], 0.0)
    # Candidate and comparator simulations are unrelated in the application,
    # so their draw axes are deliberately resampled independently even when a
    # synthetic identity fixture happens to contain the same values.
    assert np.all(result["draw_mc_se"] > 0)
    assert np.all(result["combined_bootstrap_se"] > 0)
    assert set(result["draw_resampling"]) == {"independent"}
    assert result["combined_familywise_ci_lower"].notna().all()
    assert result["combined_familywise_ci_upper"].notna().all()
    assert np.all(result["combined_familywise_family_size"] == 2)
    assert result["combined_familywise_critical_value"].nunique() == 1


def test_seed_tail_bootstrap_draw_propagation_is_reproducible():
    rng = np.random.default_rng(18)
    target = rng.normal(size=(60, 2))
    candidates = rng.normal(size=(3, 120, 60, 2))
    comparator = rng.normal(size=(120, 60, 2))
    settings = dict(
        n_bootstrap=30,
        n_draw_resamples=16,
        seed=19,
        draw_seed=20,
        familywise_confidence=0.9,
    )
    first = paired_seed_tail_risk_bootstrap(
        target, candidates, comparator, **settings
    )
    second = paired_seed_tail_risk_bootstrap(
        target, candidates, comparator, **settings
    )
    columns = [
        "combined_ci_lower",
        "combined_ci_upper",
        "combined_bootstrap_se",
        "combined_familywise_ci_lower",
        "combined_familywise_ci_upper",
    ]
    assert np.allclose(first[columns], second[columns])
    assert np.all(first["combined_familywise_confidence"] == 0.9)


def test_matched_seed_tail_bootstrap_preserves_pairing():
    rng = np.random.default_rng(21)
    target = rng.normal(size=(60, 2))
    samples = rng.normal(size=(3, 120, 60, 2))
    result = paired_matched_seed_tail_risk_bootstrap(
        target,
        samples,
        samples.copy(),
        n_bootstrap=30,
        n_draw_resamples=16,
        seed=22,
        draw_seed=23,
    )
    assert np.allclose(result["difference"], 0.0)
    assert np.allclose(result["bootstrap_se"], 0.0)
    assert np.allclose(result["draw_mc_se"], 0.0)
    assert np.allclose(result["combined_bootstrap_se"], 0.0)
    assert set(result["draw_resampling"]) == {"paired"}
