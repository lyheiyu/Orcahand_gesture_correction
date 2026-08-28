import numpy as np

import evaluate_orca_dimension_control as experiment


def test_flex11_is_fixed_semantic_subset() -> None:
    assert len(experiment.FLEX11_INDICES) == 11
    assert len(set(experiment.FLEX11_INDICES)) == 11
    assert set(experiment.FLEX11_INDICES) == {2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16}
    assert set(range(17)) - set(experiment.FLEX11_INDICES) == {0, 1, 4, 7, 10, 14}


def test_resample16_dimension_control() -> None:
    sequence11 = np.arange(7 * 11, dtype=np.float32).reshape(7, 11)
    sequence17 = np.arange(7 * 17, dtype=np.float32).reshape(7, 17)
    assert experiment._encode([sequence11]).shape == (1, 176)
    assert experiment._encode([sequence17]).shape == (1, 272)


def test_mapping_rows_cover_every_actuator_once() -> None:
    rows = experiment._mapping_rows()
    assert len(rows) == 17
    assert [row["original_index"] for row in rows] == list(range(17))
    assert sum(row["keep_in_flex11"] == "yes" for row in rows) == 11
