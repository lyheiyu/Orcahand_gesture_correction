from __future__ import annotations

import csv
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orca_sim.gesture_features import OrcaFeatureProjector, palm_normal_vector
from orca_sim.mujoco_optimizer import MujocoHandPoseOptimizer


DATASET_PATH = ROOT / "gesture_sequence_dataset_chinese_dance_6class.csv"


def _load_sequences(limit: int = 2) -> tuple[list[str], dict[str, list[np.ndarray]]]:
    with DATASET_PATH.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        raw_names = [name for name in (reader.fieldnames or []) if name.startswith("raw_")]
        grouped: dict[str, list[tuple[int, float, np.ndarray]]] = defaultdict(list)
        for row in reader:
            sequence_id = row.get("sequence_id", "")
            if not sequence_id:
                continue
            frame_id = int(float(row.get("frame_id") or 0))
            timestamp_sec = float(row.get("timestamp_sec") or 0.0)
            points = np.array([float(row[name]) for name in raw_names], dtype=np.float64).reshape(21, 3)
            grouped[sequence_id].append((frame_id, timestamp_sec, points))

    selected_ids = sorted(grouped)[:limit]
    sequences = {
        sequence_id: [points for _, _, points in sorted(grouped[sequence_id], key=lambda item: (item[0], item[1]))]
        for sequence_id in selected_ids
    }
    return selected_ids, sequences


class MujocoOptimizerRegressionTests(unittest.TestCase):
    def test_palm_normal_convention_matches_generated_pose(self) -> None:
        with MujocoHandPoseOptimizer(version="v2") as optimizer:
            action = 0.5 * (
                optimizer.env.action_low.astype(np.float64) + optimizer.env.action_high.astype(np.float64)
            )
            generated_points = optimizer.full_landmarks_from_action(action)
            target_normal = palm_normal_vector(generated_points)
            _, predicted_normal = optimizer._forward_sparse_points(action)

        dot_product = float(np.dot(target_normal, predicted_normal))
        self.assertGreater(
            dot_product,
            0.99,
            msg=f"Palm normals should align for a MuJoCo-generated pose, got dot={dot_product:.6f}",
        )

    def test_optimizer_outputs_remain_within_bounds(self) -> None:
        sequence_ids, sequences = _load_sequences(limit=2)
        self.assertTrue(sequence_ids, "Expected at least one sequence in the diagnostic dataset.")

        with MujocoHandPoseOptimizer(version="v2") as optimizer:
            for sequence_id in sequence_ids:
                prev_action = None
                prev_prev_action = None
                for points in sequences[sequence_id]:
                    result = optimizer.optimize(points, prev_action=prev_action, prev_prev_action=prev_prev_action)
                    self.assertTrue(np.isfinite(result.action).all(), "Non-finite actuator values produced.")
                    self.assertTrue(np.isfinite(result.optimized_full_points).all(), "Non-finite full landmarks produced.")
                    self.assertTrue(np.isfinite(result.solve_time_ms), "Non-finite solve time produced.")
                    self.assertGreaterEqual(result.solve_time_ms, 0.0)
                    self.assertTrue(
                        np.all(result.action >= optimizer.env.action_low - 1e-6),
                        "Actuator fell below lower bound.",
                    )
                    self.assertTrue(
                        np.all(result.action <= optimizer.env.action_high + 1e-6),
                        "Actuator exceeded upper bound.",
                    )
                    old_prev = prev_action
                    prev_action = result.action.astype(np.float64)
                    prev_prev_action = old_prev

    def test_sequence_history_reset_changes_first_frame_behavior(self) -> None:
        sequence_ids, sequences = _load_sequences(limit=2)
        self.assertEqual(len(sequence_ids), 2, "Need at least two sequences for the reset regression test.")
        seq_a, seq_b = sequence_ids

        with MujocoHandPoseOptimizer(version="v2") as optimizer:
            prev_action = None
            prev_prev_action = None
            for points in sequences[seq_a]:
                result = optimizer.optimize(points, prev_action=prev_action, prev_prev_action=prev_prev_action)
                old_prev = prev_action
                prev_action = result.action.astype(np.float64)
                prev_prev_action = old_prev

            first_points_b = sequences[seq_b][0]
            reset_result = optimizer.optimize(first_points_b, prev_action=None, prev_prev_action=None)
            carried_result = optimizer.optimize(first_points_b, prev_action=prev_action, prev_prev_action=prev_prev_action)

        self.assertAlmostEqual(reset_result.loss_terms["temporal"], 0.0, places=10)
        self.assertAlmostEqual(reset_result.loss_terms["acceleration"], 0.0, places=10)
        self.assertGreater(carried_result.loss_terms["temporal"], 0.0)
        self.assertGreater(carried_result.loss_terms["acceleration"], 0.0)
        self.assertGreater(
            float(np.linalg.norm(reset_result.action.astype(np.float64) - carried_result.action.astype(np.float64))),
            1e-6,
            "Resetting history should change the first-frame optimized action when carry-over state differs.",
        )

    def test_corrected_vector_respects_bounds(self) -> None:
        _, sequences = _load_sequences(limit=1)
        first_sequence = next(iter(sequences.values()))

        with OrcaFeatureProjector(version="v2") as projector:
            for points in first_sequence[:10]:
                corrected = projector.corrected_vector(points)
                self.assertTrue(np.isfinite(corrected).all())
                self.assertTrue(np.all(corrected >= projector.env.action_low - 1e-6))
                self.assertTrue(np.all(corrected <= projector.env.action_high + 1e-6))


if __name__ == "__main__":
    unittest.main()
