import csv
from pathlib import Path

import numpy as np

from generate_recovery_corruptions import CorruptionCondition, _corrupt_sequence, generate


def _write_fixture(path: Path) -> None:
    fields = ["label", "sequence_id", "frame_id", "timestamp_sec"] + [f"raw_{i}" for i in range(63)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sequence_index in range(4):
            for frame_index in range(8):
                points = np.arange(63, dtype=float) / 100.0 + frame_index / 100.0
                row = {
                    "label": f"class_{sequence_index % 2}",
                    "sequence_id": f"seq_{sequence_index}",
                    "frame_id": frame_index,
                    "timestamp_sec": frame_index / 30.0,
                }
                row.update({f"raw_{i}": value for i, value in enumerate(points)})
                writer.writerow(row)


def test_dropout_changes_only_requested_distal_landmarks() -> None:
    clean = np.arange(10 * 21 * 3, dtype=float).reshape(10, 21, 3) / 100.0
    condition = CorruptionCondition("dropout", "short", 3, "thumb", 0.0)
    corrupted, changes = _corrupt_sequence(clean, condition, random_seed=42)
    changed_landmarks = {landmark_id for _, landmark_id, _, _ in changes}
    assert changed_landmarks == {2, 3, 4}
    untouched = sorted(set(range(21)) - changed_landmarks)
    np.testing.assert_array_equal(corrupted[:, untouched], clean[:, untouched])


def test_generation_is_deterministic_and_manifest_is_frame_level(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_fixture(source)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate(source, first, seed=7, max_sequences=None)
    generate(source, second, seed=7, max_sequences=None)

    assert (first / "corrupted_landmarks.csv").read_bytes() == (second / "corrupted_landmarks.csv").read_bytes()
    assert (first / "corruption_manifest.csv").read_bytes() == (second / "corruption_manifest.csv").read_bytes()
    with (first / "corruption_manifest.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert {"sequence_id", "frame_index", "landmark_id", "random_seed", "condition_id"} <= set(row)
