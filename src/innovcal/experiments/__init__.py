"""Paper-style CA-RNN comparisons."""

from innovcal.experiments.comparison import (
    ExperimentResult,
    run_four_model_comparison,
    run_temporal_penalty_ablation,
)
from innovcal.experiments.isolated_shift import (
    IsolatedScaleShiftConfig,
    run_isolated_scale_shift,
    summarize_isolated_shift,
)
from innovcal.experiments.reproducibility import (
    MultiSeedResult,
    RegimeDefinition,
    run_frequentist_seed_comparison,
    run_regime_seed_comparison,
)

__all__ = [
    "ExperimentResult",
    "run_four_model_comparison",
    "run_temporal_penalty_ablation",
    "MultiSeedResult",
    "RegimeDefinition",
    "run_frequentist_seed_comparison",
    "run_regime_seed_comparison",
    "IsolatedScaleShiftConfig",
    "run_isolated_scale_shift",
    "summarize_isolated_shift",
]
