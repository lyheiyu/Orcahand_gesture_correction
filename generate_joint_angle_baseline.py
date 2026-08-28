import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


JOINT_DEFINITIONS = (
    ("thumb_cmc", 0, 1, 2),
    ("thumb_mcp", 1, 2, 3),
    ("thumb_ip", 2, 3, 4),
    ("index_mcp", 0, 5, 6),
    ("index_pip", 5, 6, 7),
    ("middle_mcp", 0, 9, 10),
    ("middle_pip", 9, 10, 11),
    ("ring_mcp", 0, 13, 14),
    ("ring_pip", 13, 14, 15),
    ("little_mcp", 0, 17, 18),
    ("little_pip", 17, 18, 19),
)


def joint_angle_vector(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Expected (21, 3) landmarks, got {points.shape}")

    angles = np.zeros(len(JOINT_DEFINITIONS), dtype=np.float64)
    valid = np.ones(len(JOINT_DEFINITIONS), dtype=bool)
    for index, (_, proximal, joint, distal) in enumerate(JOINT_DEFINITIONS):
        v1 = points[proximal] - points[joint]
        v2 = points[distal] - points[joint]
        denominator = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denominator <= 1e-12 or not np.isfinite(denominator):
            valid[index] = False
            angles[index] = 0.0
            continue
        cosine = float(np.dot(v1, v2) / denominator)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        angle = math.degrees(math.acos(cosine))
        if not np.isfinite(angle):
            valid[index] = False
            angles[index] = 0.0
        else:
            angles[index] = angle
    return angles.astype(np.float32), valid


def _numeric_suffix(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append the conventional 11D absolute 3D joint-angle baseline to a gesture CSV."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    angle_headers = [f"joint_angle_{index}" for index in range(len(JOINT_DEFINITIONS))]
    invalid_by_joint = np.zeros(len(JOINT_DEFINITIONS), dtype=np.int64)
    invalid_frames = 0
    num_frames = 0
    sequence_ids: set[str] = set()

    with input_path.open("r", newline="", encoding="utf-8") as fh_in:
        reader = csv.DictReader(fh_in)
        input_headers = reader.fieldnames or []
        raw_names = sorted(
            (name for name in input_headers if name.startswith("raw_")),
            key=_numeric_suffix,
        )
        if len(raw_names) != 63:
            raise SystemExit(f"Expected 63 raw_* columns, found {len(raw_names)}")
        passthrough = [name for name in input_headers if not name.startswith("joint_angle_")]

        with output_path.open("w", newline="", encoding="utf-8") as fh_out:
            writer = csv.DictWriter(fh_out, fieldnames=passthrough + angle_headers)
            writer.writeheader()
            for row in reader:
                points = np.asarray([float(row[name]) for name in raw_names], dtype=np.float64).reshape(21, 3)
                angles, valid = joint_angle_vector(points)
                invalid_by_joint += (~valid).astype(np.int64)
                invalid_frames += int(not bool(np.all(valid)))
                num_frames += 1
                sequence_ids.add(row.get("sequence_id", ""))
                output_row = {name: row.get(name, "") for name in passthrough}
                for index, value in enumerate(angles):
                    output_row[f"joint_angle_{index}"] = float(value)
                writer.writerow(output_row)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "representation": "JointAngle-11",
        "units": "degrees",
        "num_dimensions": len(JOINT_DEFINITIONS),
        "num_frames": num_frames,
        "num_sequences": len(sequence_ids),
        "invalid_frames": invalid_frames,
        "invalid_values": int(np.sum(invalid_by_joint)),
        "invalid_by_joint": {
            JOINT_DEFINITIONS[index][0]: int(value)
            for index, value in enumerate(invalid_by_joint)
        },
        "joint_definitions": [
            {
                "name": name,
                "proximal": proximal,
                "joint": joint,
                "distal": distal,
            }
            for name, proximal, joint, distal in JOINT_DEFINITIONS
        ],
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"summary_json={summary_path}")
    print(f"num_sequences={len(sequence_ids)} num_frames={num_frames}")
    print(f"invalid_frames={invalid_frames} invalid_values={int(np.sum(invalid_by_joint))}")


if __name__ == "__main__":
    main()
