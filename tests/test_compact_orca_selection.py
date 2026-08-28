from pathlib import Path

import numpy as np

import run_compact_orca_selection as compact


def _fake_scores() -> list[dict[str, object]]:
    return [
        {"index": index, "finger_group": compact.ACTUATOR_META[index][1], "utility_score": float(index)}
        for index in range(17)
    ]


def test_semantic_subset_has_requested_size_and_finger_coverage() -> None:
    for k in compact.CANDIDATE_K:
        selected = compact.select_semantic_subset(_fake_scores(), k)
        assert len(selected) == k
        assert len(set(selected)) == k
        groups = {compact.ACTUATOR_META[index][1] for index in selected}
        assert set(compact.FINGER_GROUPS) <= groups


def test_one_standard_error_prefers_smallest_eligible_k() -> None:
    rows = []
    for repeat in range(4):
        for k, score in ((5, 0.70), (7, 0.80), (9, 0.805), (11, 0.806), (13, 0.804)):
            for classifier in ("svm", "knn", "rf", "mlp"):
                rows.append({"repeat": repeat, "k": k, "combined_score": score})
    selected, summary = compact.select_k(rows)
    assert selected == 11  # Zero SE means only the exact best candidate is eligible.
    assert len(summary) == len(compact.CANDIDATE_K)


def test_outer_split_is_frozen_and_disjoint(tmp_path: Path) -> None:
    ids = [f"seq-{index}" for index in range(60)]
    labels = [f"class-{index % 6}" for index in range(60)]
    development, final = compact.freeze_outer_split(ids, labels, tmp_path, 0.2, 123)
    assert not (set(development) & set(final))
    assert len(development) + len(final) == len(ids)
    assert (tmp_path / "development_sequences.csv").exists()
    assert (tmp_path / "final_test_sequences.csv").exists()
    assert compact.freeze_outer_split(ids, labels, tmp_path, 0.2, 999) == (development, final)


def test_frozen_compact_indices_are_unchanged() -> None:
    assert compact.FROZEN_K == 7
    assert compact.FROZEN_INDICES == (3, 6, 9, 11, 12, 15, 16)


def test_holm_adjustment_is_monotonic_in_sorted_p_values() -> None:
    adjusted = compact._holm_adjust([0.01, 0.04, 0.02])
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])
