import numpy as np

from run_recovery_benchmark import _amplitude_metrics, _safe_recovery, _stats


def test_recovery_ratio_interpretation() -> None:
    assert _safe_recovery(0.5, 1.0) == 0.5
    assert _safe_recovery(1.0, 1.0) == 0.0
    assert _safe_recovery(1.5, 1.0) == -0.5


def test_masked_stats_use_only_corrupted_pairs() -> None:
    error = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    mask = np.asarray([[False, True], [True, False]])
    assert _stats(error, mask)["mean"] == 2.5


def test_amplitude_retention_is_one_for_identical_motion() -> None:
    clean = np.arange(20, dtype=float).reshape(5, 2, 2)
    result = _amplitude_metrics(clean.copy(), clean, active_threshold=1e-6)
    assert result["median"] == 1.0
    assert result["within_0p9_1p1"] == 1.0
