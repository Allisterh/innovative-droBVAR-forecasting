import numpy as np
import pytest

from innovcal.evaluation import (
    combined_seed_draw_bootstrap,
    paired_block_bootstrap,
    paired_draw_mc_sensitivity,
    paired_seed_block_bootstrap,
)
from innovcal.evaluation import uncertainty as uncertainty_module
from innovcal.evaluation.uncertainty import moving_block_indices, moving_block_resample


def test_moving_block_indices_have_valid_length_and_range():
    indices = moving_block_indices(37, 5, np.random.default_rng(2))
    assert indices.shape == (37,)
    assert indices.min() >= 0 and indices.max() < 37


def test_moving_block_resample_marks_every_artificial_join():
    indices, block_ids = moving_block_resample(37, 5, np.random.default_rng(2))
    assert indices.shape == block_ids.shape == (37,)
    within_block = block_ids[1:] == block_ids[:-1]
    assert np.all(np.diff(indices)[within_block] == 1)
    assert np.count_nonzero(~within_block) == 7


def test_identical_forecasts_have_zero_paired_intervals():
    rng = np.random.default_rng(3)
    target = rng.normal(size=(40, 2))
    samples = rng.normal(size=(60, 40, 2))
    result = paired_block_bootstrap(
        target,
        {"candidate": samples, "comparator": samples.copy()},
        [("candidate", "comparator")],
        np.eye(2),
        block_length=5,
        n_bootstrap=20,
        seed=4,
        simultaneous_metrics=("mean_squared_error", "energy_score"),
    )
    assert np.allclose(result["difference"], 0.0)
    assert np.allclose(result["ci_lower"], 0.0)
    assert np.allclose(result["ci_upper"], 0.0)
    assert "coverage_error" in set(result["metric"])
    selected = result[result.metric.isin(["mean_squared_error", "energy_score"])]
    assert np.allclose(selected["familywise_ci_lower"], 0.0)
    assert np.allclose(selected["familywise_ci_upper"], 0.0)
    assert np.all(selected["familywise_family_size"] == 2)


def test_identical_multiseed_forecasts_have_zero_hierarchical_intervals():
    rng = np.random.default_rng(5)
    target = rng.normal(size=(35, 2))
    samples = rng.normal(size=(3, 40, 35, 2))
    result = paired_seed_block_bootstrap(
        target,
        {"candidate": samples, "comparator": samples.copy()},
        [("candidate", "comparator")],
        np.eye(2),
        block_length=5,
        n_bootstrap=20,
        seed=6,
    )
    assert np.allclose(result["difference"], 0.0)
    assert np.allclose(result["ci_lower"], 0.0)
    assert np.allclose(result["ci_upper"], 0.0)


def test_hierarchical_bootstrap_draws_one_shared_date_path_per_replicate(monkeypatch):
    rng = np.random.default_rng(9)
    target = rng.normal(size=(30, 2))
    candidate = rng.normal(size=(3, 40, 30, 2))
    comparator = rng.normal(size=(3, 40, 30, 2))
    calls = 0

    def fake_resample(n_observations, block_length, _rng):
        nonlocal calls
        calls += 1
        indices = np.arange(n_observations)
        block_ids = np.arange(n_observations) // block_length
        return indices, block_ids

    monkeypatch.setattr(uncertainty_module, "moving_block_resample", fake_resample)
    paired_seed_block_bootstrap(
        target,
        {"candidate": candidate, "comparator": comparator},
        [("candidate", "comparator")],
        np.eye(2),
        block_length=5,
        n_bootstrap=12,
        seed=10,
    )
    assert calls == 12


def test_nonlinear_diagnostics_average_seed_specific_statistics_not_pooled_pits(
    monkeypatch,
):
    n_origins = 20
    # Use two response dimensions so that every default diagnostic, including
    # the pairwise variogram score, has a nonempty definition in this test.
    target = np.zeros((n_origins, 2))
    candidate = np.stack(
        [np.full((30, n_origins, 2), marker) for marker in (0.0, 1.0)]
    )
    comparator = np.stack(
        [np.full((30, n_origins, 2), marker) for marker in (2.0, 3.0)]
    )
    pit_values = {
        0: np.linspace(0.02, 0.98, n_origins)[:, None],
        1: np.full((n_origins, 1), 0.99),
        2: np.full((n_origins, 1), 0.25),
        3: np.full((n_origins, 1), 0.75),
    }

    def fake_pits(_target, samples, _projections):
        return pit_values[int(samples[0, 0, 0])]

    monkeypatch.setattr(uncertainty_module, "empirical_pits", fake_pits)
    monkeypatch.setattr(
        uncertainty_module,
        "_pit_dependence_from_contributions",
        lambda _values, _indices=None: 0.0,
    )
    result = paired_seed_block_bootstrap(
        target,
        {"candidate": candidate, "comparator": comparator},
        [("candidate", "comparator")],
        np.eye(2),
        block_length=5,
        n_bootstrap=20,
        seed=11,
    )
    observed = result.loc[
        result["metric"].eq("pit_calibration_error"), "difference"
    ].item()
    expected = np.mean(
        [
            uncertainty_module._calibration_error(pit_values[0])
            - uncertainty_module._calibration_error(pit_values[2]),
            uncertainty_module._calibration_error(pit_values[1])
            - uncertainty_module._calibration_error(pit_values[3]),
        ]
    )
    pooled = uncertainty_module._calibration_error(
        np.concatenate([pit_values[0], pit_values[1]], axis=0)
    ) - uncertainty_module._calibration_error(
        np.concatenate([pit_values[2], pit_values[3]], axis=0)
    )
    assert np.isclose(observed, expected)
    assert not np.isclose(observed, pooled)

    coverage_observed = result.loc[
        result["metric"].eq("coverage_error"), "difference"
    ].item()
    # Candidate seed errors are |1-.9| and |0-.9|; both comparator seeds have
    # error |0-.9|. Averaging seed-specific errors gives -0.4, whereas pooling
    # candidate hits first would give |.5-.9|-.9 = -0.5.
    assert np.isclose(coverage_observed, -0.4)
    assert not np.isclose(coverage_observed, -0.5)


def test_hierarchical_bootstrap_adds_declared_familywise_intervals():
    rng = np.random.default_rng(12)
    target = rng.normal(size=(45, 2))
    candidate = rng.normal(loc=0.15, size=(3, 50, 45, 2))
    comparator = rng.normal(size=(3, 50, 45, 2))
    result = paired_seed_block_bootstrap(
        target,
        {"candidate": candidate, "comparator": comparator},
        [("candidate", "comparator")],
        np.eye(2),
        block_length=5,
        n_bootstrap=80,
        seed=13,
        simultaneous_metrics=("mean_squared_error", "energy_score"),
        familywise_confidence=0.95,
    )
    selected = result[result.metric.isin(["mean_squared_error", "energy_score"])]
    assert selected["familywise_ci_lower"].notna().all()
    assert selected["familywise_ci_upper"].notna().all()
    assert np.all(selected["familywise_family_size"] == 2)
    assert selected["familywise_critical_value"].nunique() == 1
    critical = selected["familywise_critical_value"].iloc[0]
    assert np.allclose(
        selected["familywise_ci_lower"],
        selected["difference"] - critical * selected["bootstrap_se"],
    )
    assert np.allclose(
        selected["familywise_ci_upper"],
        selected["difference"] + critical * selected["bootstrap_se"],
    )
    unselected = result[~result.metric.isin(["mean_squared_error", "energy_score"])]
    assert unselected["familywise_ci_lower"].isna().all()
    assert unselected["familywise_resolved"].isna().all()


def test_hierarchical_bootstrap_rejects_unknown_simultaneous_metric():
    rng = np.random.default_rng(14)
    target = rng.normal(size=(30, 2))
    samples = rng.normal(size=(2, 30, 30, 2))
    with pytest.raises(ValueError, match="unknown simultaneous metrics"):
        paired_seed_block_bootstrap(
            target,
            {"candidate": samples, "comparator": samples.copy()},
            [("candidate", "comparator")],
            np.eye(2),
            block_length=5,
            n_bootstrap=20,
            seed=15,
            simultaneous_metrics=("not_a_metric",),
        )


def test_draw_mc_sensitivity_preserves_paired_common_random_numbers():
    rng = np.random.default_rng(16)
    target = rng.normal(size=(24, 2))
    samples = rng.normal(size=(3, 40, 24, 2))
    result = paired_draw_mc_sensitivity(
        target,
        {"candidate": samples, "comparator": samples.copy()},
        [("candidate", "comparator")],
        np.eye(2),
        draw_counts=(10, 40),
        n_resamples=12,
        seed=17,
        metrics=("mean_squared_error", "energy_score", "pit_calibration_error"),
    )
    assert len(result) == 6
    assert np.allclose(result["full_draw_difference"], 0.0)
    assert np.allclose(result["mean_resampled_difference"], 0.0)
    assert np.allclose(result["mc_bias"], 0.0)
    assert np.allclose(result["mc_standard_error"], 0.0)
    assert np.allclose(result["mc_ci_lower"], 0.0)
    assert np.allclose(result["mc_ci_upper"], 0.0)


def test_draw_mc_sensitivity_validates_requested_draw_counts():
    rng = np.random.default_rng(18)
    target = rng.normal(size=(20, 2))
    samples = rng.normal(size=(2, 25, 20, 2))
    with pytest.raises(ValueError, match="draw_counts"):
        paired_draw_mc_sensitivity(
            target,
            {"candidate": samples, "comparator": samples.copy()},
            [("candidate", "comparator")],
            np.eye(2),
            draw_counts=(10, 30),
            n_resamples=5,
        )


def test_combined_bootstrap_is_zero_for_identical_paired_forecasts():
    rng = np.random.default_rng(19)
    target = rng.normal(size=(28, 2))
    samples = rng.normal(size=(3, 35, 28, 2))
    result = combined_seed_draw_bootstrap(
        target,
        {"candidate": samples, "comparator": samples.copy()},
        [("candidate", "comparator")],
        np.eye(2),
        block_length=4,
        n_bootstrap=30,
        n_draw_resamples=20,
        seed=20,
        draw_seed=21,
        draw_pairing="paired",
        mc_metrics=("mean_squared_error", "energy_score"),
        simultaneous_metrics=("mean_squared_error", "energy_score"),
    )
    selected = result[result.metric.isin(["mean_squared_error", "energy_score"])]
    assert np.allclose(selected["difference"], 0.0)
    assert np.allclose(selected["draw_mc_se"], 0.0)
    assert np.allclose(selected["combined_bootstrap_se"], 0.0)
    assert np.allclose(selected["combined_ci_lower"], 0.0)
    assert np.allclose(selected["combined_ci_upper"], 0.0)
    assert np.allclose(selected["combined_familywise_ci_lower"], 0.0)
    assert np.allclose(selected["combined_familywise_ci_upper"], 0.0)
    assert set(selected["draw_resampling"]) == {"paired"}


def test_combined_bootstrap_accepts_single_fit_forecast_arrays():
    rng = np.random.default_rng(190)
    target = rng.normal(size=(28, 2))
    candidate = rng.normal(size=(35, 28, 2))
    comparator = rng.normal(size=(35, 28, 2))
    result = combined_seed_draw_bootstrap(
        target,
        {"candidate": candidate, "comparator": comparator},
        [("candidate", "comparator")],
        np.eye(2),
        block_length=4,
        n_bootstrap=30,
        n_draw_resamples=20,
        seed=191,
        draw_seed=192,
        draw_pairing="independent",
        mc_metrics=("mean_squared_error", "energy_score"),
        simultaneous_metrics=("mean_squared_error", "energy_score"),
    )
    selected = result[result.metric.isin(["mean_squared_error", "energy_score"])]
    assert np.all(selected["n_seeds"] == 1)
    assert selected["combined_ci_lower"].notna().all()
    assert selected["combined_familywise_ci_lower"].notna().all()
    assert set(selected["draw_resampling"]) == {"independent"}


def test_combined_bootstrap_supports_independent_unrelated_draws():
    rng = np.random.default_rng(22)
    target = rng.normal(size=(30, 2))
    samples = rng.normal(size=(3, 40, 30, 2))
    common = dict(
        target=target,
        forecast_samples={"candidate": samples, "comparator": samples.copy()},
        comparisons=[("candidate", "comparator")],
        projections=np.eye(2),
        block_length=5,
        n_bootstrap=30,
        n_draw_resamples=30,
        seed=23,
        draw_seed=24,
        mc_metrics=("mean_squared_error",),
    )
    paired = combined_seed_draw_bootstrap(**common, draw_pairing="paired")
    independent = combined_seed_draw_bootstrap(**common, draw_pairing="independent")
    paired_row = paired[paired.metric == "mean_squared_error"].iloc[0]
    independent_row = independent[independent.metric == "mean_squared_error"].iloc[0]
    assert paired_row.draw_mc_se == 0.0
    assert independent_row.draw_mc_se > 0.0
    assert independent_row.variance_component_se > paired_row.variance_component_se
    assert independent_row.draw_resampling == "independent"


def test_combined_bootstrap_requires_pairing_mode_for_every_comparison():
    rng = np.random.default_rng(25)
    target = rng.normal(size=(20, 2))
    samples = rng.normal(size=(2, 25, 20, 2))
    with pytest.raises(ValueError, match="missing draw-pairing modes"):
        combined_seed_draw_bootstrap(
            target,
            {"candidate": samples, "comparator": samples.copy()},
            [("candidate", "comparator")],
            np.eye(2),
            block_length=5,
            n_bootstrap=10,
            n_draw_resamples=5,
            draw_pairing={},
            mc_metrics=("mean_squared_error",),
        )
