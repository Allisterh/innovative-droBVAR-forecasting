import numpy as np
import pandas as pd
import pytest

from innovcal.data.simulation import simulate_multivariate_series
from innovcal.evaluation.metrics import gaussian_projected_pits, variogram_score
from innovcal.experiments.isolated_shift import (
    IsolatedScaleShiftConfig,
    _checkpoint_rows,
    crossed_isolated_shift_inference,
    summarize_isolated_shift,
)


def test_scale_shift_keeps_pre_shift_identical_and_not_correlation():
    reference = simulate_multivariate_series(500, seed=24, shift_start=400, shift_mechanism="scale")
    shifted = simulate_multivariate_series(
        500, seed=24, shift_start=400, shift_severity=0.8, shift_mechanism="scale"
    )
    np.testing.assert_array_equal(reference[:400], shifted[:400])
    assert not np.allclose(reference[400:], shifted[400:])


def test_analytic_gaussian_pit_matches_known_standard_normal():
    target = np.array([[0.0, 0.0], [1.0, -1.0]])
    mean = np.zeros_like(target)
    scale = np.repeat(np.eye(2)[None], len(target), axis=0)
    pit = gaussian_projected_pits(target, mean, scale, np.eye(2))
    assert np.allclose(pit[0], 0.5)
    assert np.allclose(pit[1, 0], 0.841344746, atol=1e-7)


def test_variogram_score_rewards_correct_pairwise_spread():
    target = np.array([[0.0, 2.0]])
    correct = np.repeat(target[None], 5, axis=0)
    wrong = np.zeros_like(correct)
    assert variogram_score(target, correct) == pytest.approx(0.0)
    assert variogram_score(target, wrong) > 0


def test_mc_summary_is_paired_and_counts_failures():
    rows = []
    for realization in range(3):
        for model, value in (("RNN", 2.0), ("CA-RNN", 1.0)):
            rows.append(
                {
                    "realization": realization,
                    "model": model,
                    "severity": 0.5,
                    "failure": "failed" if realization == 2 else "",
                    "energy_score": np.nan if realization == 2 else value,
                }
            )
    result = summarize_isolated_shift(pd.DataFrame(rows), metrics=("energy_score",))
    assert result.loc[0, "n_valid_pairs"] == 2
    assert result.loc[0, "failure_frequency"] == pytest.approx(1 / 3)
    assert result.loc[0, "mean_paired_difference"] == -1


def _crossed_fixture() -> pd.DataFrame:
    rows = []
    for dgp_seed in range(4):
        for neural_seed in (11, 22):
            for forecast_seed in (101, 202):
                for severity in (0.0, 0.5):
                    neural_effect = 0.02 if neural_seed == 22 else 0.0
                    forecast_effect = 0.01 if forecast_seed == 202 else 0.0
                    difference = (
                        -0.4
                        + 0.08 * dgp_seed
                        + neural_effect
                        + forecast_effect
                        + 0.2 * severity
                    )
                    baseline = 2.0 + 0.1 * dgp_seed + severity
                    for model, adjustment in (("RNN", 0.0), ("CA-RNN", difference)):
                        rows.append(
                            {
                                "realization": dgp_seed,
                                "dgp_seed": 1000 + dgp_seed,
                                "neural_seed": neural_seed,
                                "forecast_seed": forecast_seed,
                                "model": model,
                                "severity": severity,
                                "failure": "",
                                "energy_score": baseline + adjustment,
                                "mean_squared_error": 0.5 * baseline + adjustment,
                            }
                        )
    return pd.DataFrame(rows)


def test_crossed_defaults_and_legacy_scalar_seeds():
    default = IsolatedScaleShiftConfig()
    assert len(default.effective_neural_seeds) == 3
    assert len(default.effective_forecast_seeds) == 3
    legacy = IsolatedScaleShiftConfig(neural_seed=17, forecast_seed=29)
    assert legacy.effective_neural_seeds == (17,)
    assert legacy.effective_forecast_seeds == (29,)


def test_crossed_summary_averages_replicates_within_dgp():
    result = summarize_isolated_shift(
        _crossed_fixture(), metrics=("energy_score",)
    )
    assert set(result.n_neural_seeds) == {2}
    assert set(result.n_forecast_replicates) == {2}
    assert set(result.n_valid_pairs) == {4}
    expected_zero_shift = np.mean([-0.385, -0.305, -0.225, -0.145])
    observed = result.loc[result.severity.eq(0.0), "mean_paired_difference"].item()
    assert observed == pytest.approx(expected_zero_shift)


def test_crossed_inference_is_joint_reproducible_and_simultaneous():
    kwargs = {
        "metrics": ("energy_score", "mean_squared_error"),
        "n_bootstrap": 400,
        "seed": 77,
    }
    first = crossed_isolated_shift_inference(_crossed_fixture(), **kwargs)
    second = crossed_isolated_shift_inference(_crossed_fixture(), **kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.family_size) == {4}
    assert set(first.n_dgp_seeds) == {4}
    assert set(first.n_neural_seeds) == {2}
    assert set(first.n_forecast_replicates) == {2}
    assert (first.crossed_bootstrap_se > 0).all()
    assert (first.simultaneous_critical_value > 0).all()
    assert (first.family_failure_frequency == 0).all()
    assert (
        first.simultaneous_ci_upper - first.simultaneous_ci_lower
        >= 0
    ).all()


def test_crossed_inference_rejects_an_incomplete_family():
    raw = _crossed_fixture()
    raw.loc[0, "failure"] = "RuntimeError: failed"
    with pytest.raises(ValueError, match="complete model-by-factor grid"):
        crossed_isolated_shift_inference(
            raw,
            metrics=("energy_score",),
            n_bootstrap=20,
        )


def test_checkpoint_is_keyed_by_complete_dgp_neural_blocks(tmp_path):
    config = IsolatedScaleShiftConfig(
        dgp_seeds=(1, 2),
        neural_seeds=(10, 20),
        forecast_seeds=(30, 40),
        severities=(0.0, 0.5),
    )
    rows = []
    for model in ("RNN", "CA-RNN"):
        for severity in config.severities:
            for forecast_seed in config.forecast_seeds:
                rows.append(
                    {
                        "dgp_seed": 1,
                        "neural_seed": 10,
                        "model": model,
                        "severity": severity,
                        "forecast_seed": forecast_seed,
                        "failure": "",
                    }
                )
    checkpoint = tmp_path / "checkpoint.csv"
    pd.DataFrame(rows).to_csv(checkpoint, index=False)
    assert len(_checkpoint_rows(checkpoint, config)) == len(rows)
    pd.DataFrame(rows[:-1]).to_csv(checkpoint, index=False)
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        _checkpoint_rows(checkpoint, config)
