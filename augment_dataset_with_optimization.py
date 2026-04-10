import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orca_sim.mujoco_optimizer import MujocoHandPoseOptimizer


META_FIELDS = ("label", "sequence_id", "frame_id", "timestamp_sec")


def _optimizer_headers() -> list[str]:
    headers = [f"optimized_action_{i}" for i in range(17)]
    headers.extend(f"optimized_sparse_{i}" for i in range(8 * 3))
    headers.extend(f"optimized_full_{i}" for i in range(21 * 3))
    headers.extend(
        [
            "optimized_loss_total",
            "optimized_loss_landmark",
            "optimized_loss_palm",
            "optimized_loss_prior",
            "optimized_loss_temporal",
            "optimized_loss_default_pose",
            "optimized_loss_boundary",
        ]
    )
    return headers


def _extract_points(row: dict[str, str], raw_names: list[str]) -> np.ndarray:
    raw = np.array([float(row[name]) for name in raw_names], dtype=np.float64)
    return raw.reshape(21, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append MuJoCo-optimized features to an existing gesture CSV.")
    parser.add_argument("--input", required=True, help="Input CSV containing raw_* landmark columns.")
    parser.add_argument("--output", required=True, help="Output CSV path with optimized columns appended.")
    parser.add_argument("--version", default=None, help="ORCA hand version, for example v2.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", newline="", encoding="utf-8") as fh_in:
        reader = csv.DictReader(fh_in)
        fieldnames = reader.fieldnames or []
        raw_names = [name for name in fieldnames if name.startswith("raw_")]
        if len(raw_names) != 63:
            raise SystemExit(f"Expected 63 raw_* values, found {len(raw_names)} in {input_path}")

        existing_optimized = [name for name in fieldnames if name.startswith("optimized_")]
        passthrough_fields = [name for name in fieldnames if name not in existing_optimized]
        output_headers = passthrough_fields + _optimizer_headers()

        with output_path.open("w", newline="", encoding="utf-8") as fh_out:
            writer = csv.DictWriter(fh_out, fieldnames=output_headers)
            writer.writeheader()

            with MujocoHandPoseOptimizer(version=args.version) as optimizer:
                prev_sequence_id: str | None = None
                prev_action: np.ndarray | None = None

                for row in reader:
                    sequence_id = row.get("sequence_id", "")
                    if sequence_id != prev_sequence_id:
                        prev_action = None
                        prev_sequence_id = sequence_id

                    points = _extract_points(row, raw_names)
                    result = optimizer.optimize(points, prev_action=prev_action)
                    prev_action = result.action.astype(np.float64)

                    out_row = {name: row.get(name, "") for name in passthrough_fields}
                    for index, value in enumerate(result.action):
                        out_row[f"optimized_action_{index}"] = float(value)
                    for index, value in enumerate(result.optimized_sparse_points.reshape(-1)):
                        out_row[f"optimized_sparse_{index}"] = float(value)
                    for index, value in enumerate(result.optimized_full_points.reshape(-1)):
                        out_row[f"optimized_full_{index}"] = float(value)
                    out_row["optimized_loss_total"] = float(result.loss_terms["total"])
                    out_row["optimized_loss_landmark"] = float(result.loss_terms["landmark"])
                    out_row["optimized_loss_palm"] = float(result.loss_terms["palm"])
                    out_row["optimized_loss_prior"] = float(result.loss_terms["prior"])
                    out_row["optimized_loss_temporal"] = float(result.loss_terms["temporal"])
                    out_row["optimized_loss_default_pose"] = float(result.loss_terms["default_pose"])
                    out_row["optimized_loss_boundary"] = float(result.loss_terms["boundary"])
                    writer.writerow(out_row)


if __name__ == "__main__":
    main()
