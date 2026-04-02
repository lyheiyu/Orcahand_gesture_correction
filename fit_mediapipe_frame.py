import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orca_sim.gesture_features import OrcaFeatureProjector
from orca_sim.mujoco_optimizer import MujocoHandPoseOptimizer


def _load_first_points_for_label(path: Path, label: str) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        raw_names = [name for name in (reader.fieldnames or []) if name.startswith("raw_")]
        for row in reader:
            if row["label"] != label:
                continue
            raw = np.array([float(row[name]) for name in raw_names], dtype=np.float64)
            return raw.reshape(21, 3)
    raise SystemExit(f"Could not find label `{label}` in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one MediaPipe frame with the MuJoCo ORCA optimizer.")
    parser.add_argument("--dataset", default="gesture_dataset.csv", help="CSV containing raw_* landmark features.")
    parser.add_argument("--label", required=True, help="Which label to extract one sample from.")
    parser.add_argument("--version", default=None, help="ORCA hand version, e.g. v2.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    normalized_points = _load_first_points_for_label(dataset_path, args.label)

    with OrcaFeatureProjector(version=args.version) as projector:
        heuristic = projector.corrected_vector(normalized_points)

    with MujocoHandPoseOptimizer(version=args.version) as optimizer:
        result = optimizer.optimize(normalized_points, initial_action=heuristic)

    print(f"dataset={dataset_path}")
    print(f"label={args.label}")
    print(f"optimizer_method={result.method}")
    print(f"success={result.success}")
    print(f"iterations={result.iterations}")
    print(f"loss={result.loss:.6f}")
    print("loss_terms=")
    for name, value in result.loss_terms.items():
        print(f"  {name}: {value:.6f}")
    print("heuristic_action=")
    print(np.array2string(heuristic, precision=4, suppress_small=True))
    print("optimized_action=")
    print(np.array2string(result.action, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
