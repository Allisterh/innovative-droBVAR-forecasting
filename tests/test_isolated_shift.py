import numpy as np
import pandas as pd
import pytest

from innovcal.data.simulation import simulate_multivariate_series
from innovcal.evaluation.metrics import gaussian_projected_pits, variogram_score
from innovcal.experiments.isolated_shift import summarize_isolated_shift


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
