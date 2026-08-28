from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


META_FIELDS = ("label", "sequence_id", "frame_id", "timestamp_sec")
FINGER_LANDMARKS = {
    # Distal points only. MCP landmarks remain visible so the existing
    # wrist/palm normalization is not invalidated by a local occlusion.
    "thumb": (2, 3, 4),
    "index": (6, 7, 8),
    "middle": (10, 11, 12),
}


@dataclass(frozen=True)
class CorruptionCondition:
    corruption_type: str
    severity: str
    duration: int
    finger_group: str
    magnitude: float

    @property
    def condition_id(self) -> str:
        parts = [self.corruption_type, self.severity]
        if self.duration:
            parts.append(f"{self.duration}f")
        if self.finger_group != "all":
            parts.append(self.finger_group)
        return "_".join(parts)


def corruption_conditions() -> list[CorruptionCondition]:
    conditions = [
        CorruptionCondition("gaussian", "mild", 0, "all", 0.01),
        CorruptionCondition("gaussian", "medium", 0, "all", 0.03),
        CorruptionCondition("gaussian", "severe", 0, "all", 0.06),
        CorruptionCondition("spike", "fixed", 1, "mixed", 0.75),
        CorruptionCondition("spike", "fixed", 2, "mixed", 0.75),
        CorruptionCondition("spike", "fixed", 3, "mixed", 0.75),
    ]
    for finger in FINGER_LANDMARKS:
        for duration in (3, 5):
            conditions.append(CorruptionCondition("dropout", "short", duration, finger, 0.0))
    return conditions


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _raw_columns(fieldnames: list[str]) -> list[str]:
    names = [name for name in fieldnames if name.startswith("raw_")]
    names.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    if names != [f"raw_{index}" for index in range(63)]:
        raise ValueError("The input must contain raw_0 through raw_62.")
    return names


def _ordered_sequences(rows: list[dict[str, str]]) -> list[tuple[str, str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence_id"]].append(row)

    sequences: list[tuple[str, str, list[dict[str, str]]]] = []
    for sequence_id, sequence_rows in grouped.items():
        sequence_rows.sort(
            key=lambda row: (
                int(float(row.get("frame_id") or 0)),
                float(row.get("timestamp_sec") or 0.0),
            )
        )
        sequences.append((str(sequence_rows[0]["label"]), sequence_id, sequence_rows))
    return sorted(sequences, key=lambda item: (item[0], item[1]))


def _git_hash(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _balanced_assignments(
    sequences: list[tuple[str, str, list[dict[str, str]]]],
    conditions: list[CorruptionCondition],
    seed: int,
) -> list[tuple[str, str, list[dict[str, str]], CorruptionCondition, int]]:
    by_label: dict[str, list[tuple[str, str, list[dict[str, str]]]]] = defaultdict(list)
    for sequence in sequences:
        by_label[sequence[0]].append(sequence)

    assignments = []
    global_index = 0
    for label_index, label in enumerate(sorted(by_label)):
        label_sequences = by_label[label]
        rng = np.random.default_rng(seed + label_index)
        order = rng.permutation(len(label_sequences))
        for local_index, source_index in enumerate(order):
            sequence = label_sequences[int(source_index)]
            condition = conditions[(local_index + label_index) % len(conditions)]
            random_seed = seed + (global_index % 10)
            assignments.append((*sequence, condition, random_seed))
            global_index += 1
    return sorted(assignments, key=lambda item: (item[0], item[1]))


def _event_start(length: int, duration: int, rng: np.random.Generator) -> int:
    if length <= duration:
        return 0
    # Avoid the first frame so freeze dropout has a valid last observation.
    low = 1 if length > duration + 1 else 0
    return int(rng.integers(low, length - duration + 1))


def _corrupt_sequence(
    clean: np.ndarray,
    condition: CorruptionCondition,
    random_seed: int,
) -> tuple[np.ndarray, list[tuple[int, int, str, float]]]:
    rng = np.random.default_rng(random_seed)
    corrupted = clean.copy()
    changes: list[tuple[int, int, str, float]] = []

    if condition.corruption_type == "gaussian":
        noise = rng.normal(0.0, condition.magnitude, size=clean.shape)
        corrupted += noise
        for frame_index in range(len(clean)):
            for landmark_id in range(21):
                changes.append(
                    (frame_index, landmark_id, "all", float(np.linalg.norm(noise[frame_index, landmark_id])))
                )
        return corrupted, changes

    if condition.corruption_type == "spike":
        finger = sorted(FINGER_LANDMARKS)[int(rng.integers(0, len(FINGER_LANDMARKS)))]
        landmarks = FINGER_LANDMARKS[finger]
        start = _event_start(len(clean), condition.duration, rng)
        direction = rng.normal(size=3)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        offset = direction * condition.magnitude
        for frame_index in range(start, min(start + condition.duration, len(clean))):
            for landmark_id in landmarks:
                corrupted[frame_index, landmark_id] += offset
                changes.append((frame_index, landmark_id, finger, float(condition.magnitude)))
        return corrupted, changes

    if condition.corruption_type == "dropout":
        landmarks = FINGER_LANDMARKS[condition.finger_group]
        start = _event_start(len(clean), condition.duration, rng)
        source_index = max(start - 1, 0)
        for frame_index in range(start, min(start + condition.duration, len(clean))):
            for landmark_id in landmarks:
                corrupted[frame_index, landmark_id] = clean[source_index, landmark_id]
                displacement = np.linalg.norm(corrupted[frame_index, landmark_id] - clean[frame_index, landmark_id])
                changes.append((frame_index, landmark_id, condition.finger_group, float(displacement)))
        return corrupted, changes

    raise ValueError(f"Unknown corruption type: {condition.corruption_type}")


def generate(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int,
    max_sequences: int | None,
) -> dict[str, int]:
    fieldnames, rows = _load_rows(input_path)
    raw_columns = _raw_columns(fieldnames)
    sequences = _ordered_sequences(rows)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    conditions = corruption_conditions()
    assignments = _balanced_assignments(sequences, conditions, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    corrupted_path = output_dir / "corrupted_landmarks.csv"
    manifest_path = output_dir / "corruption_manifest.csv"
    scenario_path = output_dir / "scenario_manifest.csv"

    output_fields = list(META_FIELDS) + raw_columns
    manifest_fields = [
        "sequence_id", "label", "frame_index", "frame_id", "landmark_id", "finger_group",
        "corruption_type", "severity", "duration", "random_seed", "condition_id", "displacement",
    ]
    scenario_fields = [
        "sequence_id", "label", "num_frames", "corruption_type", "severity", "duration",
        "finger_group", "magnitude", "random_seed", "condition_id", "num_affected_pairs",
    ]

    affected_pairs = 0
    with (
        corrupted_path.open("w", newline="", encoding="utf-8") as corrupted_handle,
        manifest_path.open("w", newline="", encoding="utf-8") as manifest_handle,
        scenario_path.open("w", newline="", encoding="utf-8") as scenario_handle,
    ):
        corrupted_writer = csv.DictWriter(corrupted_handle, fieldnames=output_fields)
        manifest_writer = csv.DictWriter(manifest_handle, fieldnames=manifest_fields)
        scenario_writer = csv.DictWriter(scenario_handle, fieldnames=scenario_fields)
        corrupted_writer.writeheader()
        manifest_writer.writeheader()
        scenario_writer.writeheader()

        for label, sequence_id, sequence_rows, condition, random_seed in assignments:
            clean = np.asarray(
                [[float(row[name]) for name in raw_columns] for row in sequence_rows], dtype=np.float64
            ).reshape(-1, 21, 3)
            corrupted, changes = _corrupt_sequence(clean, condition, random_seed)
            if not np.all(np.isfinite(corrupted)):
                raise ValueError(f"Non-finite corrupted values for sequence {sequence_id}")

            for local_index, (row, points) in enumerate(zip(sequence_rows, corrupted)):
                output_row = {name: row.get(name, "") for name in META_FIELDS}
                output_row.update(
                    {name: f"{value:.9g}" for name, value in zip(raw_columns, points.reshape(-1))}
                )
                corrupted_writer.writerow(output_row)

            for frame_index, landmark_id, finger_group, displacement in changes:
                manifest_writer.writerow(
                    {
                        "sequence_id": sequence_id,
                        "label": label,
                        "frame_index": frame_index,
                        "frame_id": sequence_rows[frame_index].get("frame_id", frame_index),
                        "landmark_id": landmark_id,
                        "finger_group": finger_group,
                        "corruption_type": condition.corruption_type,
                        "severity": condition.severity,
                        "duration": condition.duration,
                        "random_seed": random_seed,
                        "condition_id": condition.condition_id,
                        "displacement": f"{displacement:.9g}",
                    }
                )
            affected_pairs += len(changes)
            scenario_writer.writerow(
                {
                    "sequence_id": sequence_id,
                    "label": label,
                    "num_frames": len(sequence_rows),
                    "corruption_type": condition.corruption_type,
                    "severity": condition.severity,
                    "duration": condition.duration,
                    "finger_group": condition.finger_group,
                    "magnitude": condition.magnitude,
                    "random_seed": random_seed,
                    "condition_id": condition.condition_id,
                    "num_affected_pairs": len(changes),
                }
            )

    metadata = {
        "benchmark": "controlled clean-reference trajectory recovery",
        "input": str(input_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "git_commit": _git_hash(Path(__file__).resolve().parent),
        "base_seed": seed,
        "seed_pool_size": 10,
        "assignment": "one balanced corruption condition per sequence",
        "coordinate_unit": "existing normalized raw landmark coordinate",
        "gaussian_standard_deviations": {"mild": 0.01, "medium": 0.03, "severe": 0.06},
        "spike_displacement_magnitude": 0.75,
        "dropout_model": "freeze distal finger landmarks at the last visible frame",
        "dropout_landmarks": {name: list(indices) for name, indices in FINGER_LANDMARKS.items()},
        "conditions": [asdict(condition) | {"condition_id": condition.condition_id} for condition in conditions],
        "num_sequences": len(assignments),
        "num_frames": sum(len(item[2]) for item in assignments),
        "num_affected_frame_landmark_pairs": affected_pairs,
    }
    with (output_dir / "corruption_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "num_sequences": len(assignments),
        "num_frames": int(metadata["num_frames"]),
        "num_affected_pairs": affected_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled corruptions for trajectory recovery.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()

    result = generate(
        Path(args.input), Path(args.output_dir), seed=args.seed, max_sequences=args.max_sequences
    )
    for key, value in result.items():
        print(f"{key}={value}")
    print(f"output_dir={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
