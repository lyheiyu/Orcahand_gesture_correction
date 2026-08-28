import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orca_sim.gesture_features import OrcaFeatureProjector


META_FIELDS = ("label", "sequence_id", "frame_id", "timestamp_sec")
FINGER_LANDMARKS = {
    # Keep palm scale anchors intact; distal joints are the common self-occlusion failures.
    "thumb": (2, 3, 4),
    "index": (6, 7, 8),
    "middle": (10, 11, 12),
    "ring": (14, 15, 16),
    "pinky": (18, 19, 20),
}
SEVERITY_CONFIG = {
    "light": {"window_fraction": 0.15, "num_fingers": 1, "noise_std": 0.02},
    "medium": {"window_fraction": 0.30, "num_fingers": 1, "noise_std": 0.05},
    "heavy": {"window_fraction": 0.45, "num_fingers": 2, "noise_std": 0.08},
}


def _numeric_suffix(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def _read_sequences(input_path: Path) -> tuple[list[str], list[dict[str, str]], dict[str, list[int]]]:
    with input_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        raw_names = sorted(
            (name for name in fieldnames if name.startswith("raw_")),
            key=_numeric_suffix,
        )
        if len(raw_names) != 63:
            raise SystemExit(f"Expected 63 raw_* columns, found {len(raw_names)} in {input_path}")
        rows = list(reader)

    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row["sequence_id"], []).append(index)
    for indices in grouped.values():
        indices.sort(
            key=lambda index: (
                int(float(rows[index].get("frame_id", 0) or 0)),
                float(rows[index].get("timestamp_sec", 0.0) or 0.0),
            )
        )
    return raw_names, rows, grouped


def _sequence_scale(points: np.ndarray) -> float:
    across = np.linalg.norm(points[5] - points[17])
    forward = np.linalg.norm(points[9] - points[0])
    return max(float(across), float(forward), 1e-6)


def _smooth_random_walk(length: int, rng: np.random.Generator) -> np.ndarray:
    increments = rng.normal(0.0, 1.0, size=(length, 3))
    walk = np.cumsum(increments, axis=0)
    walk -= walk[0]
    norms = np.linalg.norm(walk, axis=1)
    maximum = float(np.max(norms)) if norms.size else 0.0
    return walk / maximum if maximum > 1e-9 else walk


def _apply_corruption(
    clean: np.ndarray,
    start: int,
    end: int,
    landmark_indices: tuple[int, ...],
    mode: str,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    corrupted = clean.copy()
    length = end - start
    if length <= 0:
        return corrupted

    reference_index = max(0, start - 1)
    reference = clean[reference_index, list(landmark_indices)].copy()
    scale = float(np.median([_sequence_scale(points) for points in clean]))
    palm_centers = clean[:, [0, 5, 9, 13, 17]].mean(axis=1)
    walk = _smooth_random_walk(length, rng) * (noise_std * scale)

    for local_index, frame_index in enumerate(range(start, end)):
        if mode == "freeze":
            values = reference.copy()
        elif mode == "drift":
            values = reference + walk[local_index]
        elif mode == "collapse":
            center = palm_centers[frame_index]
            values = center + 0.20 * (reference - center)
        else:
            raise ValueError(f"Unsupported corruption mode: {mode}")

        noise = rng.normal(
            0.0,
            noise_std * scale,
            size=(len(landmark_indices), 3),
        )
        corrupted[frame_index, list(landmark_indices)] = values + noise
    return corrupted


def _output_headers(projector: OrcaFeatureProjector) -> list[str]:
    groups = projector.all_feature_groups(np.zeros((21, 3), dtype=np.float64))
    headers = list(META_FIELDS)
    for prefix in ("raw", "geom", "corrected"):
        headers.extend(f"{prefix}_{index}" for index in range(len(groups[prefix])))
    return headers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a controlled landmark-level occlusion proxy and recompute raw, geometry, "
            "and corrected features. Run augment_dataset_with_optimization.py afterwards to "
            "regenerate Optimized Action and Optimized Full."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--severity", choices=sorted(SEVERITY_CONFIG), default="medium")
    parser.add_argument("--mode", choices=("freeze", "drift", "collapse", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="v2")
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Process only the first N sequences for a smoke test; 0 processes all sequences.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    raw_names, rows, grouped = _read_sequences(input_path)
    selected_sequence_ids = list(grouped)
    if args.max_sequences > 0:
        selected_sequence_ids = selected_sequence_ids[: args.max_sequences]

    config = SEVERITY_CONFIG[args.severity]
    rng = np.random.default_rng(args.seed)
    output_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    with OrcaFeatureProjector(version=args.version) as projector:
        headers = _output_headers(projector)
        for sequence_id in selected_sequence_ids:
            indices = grouped[sequence_id]
            clean = np.asarray(
                [[float(rows[index][name]) for name in raw_names] for index in indices],
                dtype=np.float64,
            ).reshape(len(indices), 21, 3)
            sequence_length = len(indices)
            window_length = min(
                sequence_length,
                max(2, int(math.ceil(sequence_length * float(config["window_fraction"])))),
            )
            max_start = max(0, sequence_length - window_length)
            start = int(rng.integers(0, max_start + 1)) if max_start else 0
            end = start + window_length
            fingers = tuple(
                str(value)
                for value in rng.choice(
                    list(FINGER_LANDMARKS),
                    size=int(config["num_fingers"]),
                    replace=False,
                )
            )
            landmark_indices = tuple(
                sorted({index for finger in fingers for index in FINGER_LANDMARKS[finger]})
            )
            mode = (
                str(rng.choice(("freeze", "drift", "collapse")))
                if args.mode == "mixed"
                else args.mode
            )
            corrupted = _apply_corruption(
                clean,
                start,
                end,
                landmark_indices,
                mode,
                float(config["noise_std"]),
                rng,
            )

            changed = np.linalg.norm(corrupted - clean, axis=2)
            affected = changed[start:end, list(landmark_indices)]
            manifest_rows.append(
                {
                    "sequence_id": sequence_id,
                    "label": rows[indices[0]]["label"],
                    "severity": args.severity,
                    "mode": mode,
                    "fingers": ";".join(fingers),
                    "landmark_indices": ";".join(str(value) for value in landmark_indices),
                    "start_frame_offset": start,
                    "end_frame_offset_exclusive": end,
                    "sequence_length": sequence_length,
                    "noise_std_scale": config["noise_std"],
                    "mean_affected_displacement": float(np.mean(affected)),
                    "max_affected_displacement": float(np.max(affected)),
                    "seed": args.seed,
                }
            )

            for frame_offset, row_index in enumerate(indices):
                source = rows[row_index]
                groups_for_frame = projector.all_feature_groups(corrupted[frame_offset])
                output_row: dict[str, object] = {
                    "label": source["label"],
                    "sequence_id": sequence_id,
                    "frame_id": source.get("frame_id", ""),
                    "timestamp_sec": source.get("timestamp_sec", ""),
                }
                for prefix in ("raw", "geom", "corrected"):
                    for feature_index, value in enumerate(groups_for_frame[prefix]):
                        output_row[f"{prefix}_{feature_index}"] = float(value)
                output_rows.append(output_row)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)

    manifest_headers = [
        "sequence_id",
        "label",
        "severity",
        "mode",
        "fingers",
        "landmark_indices",
        "start_frame_offset",
        "end_frame_offset_exclusive",
        "sequence_length",
        "noise_std_scale",
        "mean_affected_displacement",
        "max_affected_displacement",
        "seed",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=manifest_headers)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"manifest={manifest_path}")
    print(f"severity={args.severity} mode={args.mode} seed={args.seed}")
    print(f"num_sequences={len(selected_sequence_ids)} num_frames={len(output_rows)}")
    print("next_step=run augment_dataset_with_optimization.py on the generated output")


if __name__ == "__main__":
    main()
