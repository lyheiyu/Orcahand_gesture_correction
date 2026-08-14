from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train_svm as tsvm
from orca_sim.gesture_features import OrcaFeatureProjector, palm_normal_vector
from orca_sim.mujoco_optimizer import MujocoHandPoseOptimizer, OptimizationWeights


def load_rows(dataset_path: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray]:
    return tsvm._load_dataset(dataset_path)


def save_split_manifest(
    output_path: Path,
    row_meta: list[dict[str, str]],
    labels: list[str],
    features: np.ndarray,
    repeats: int,
    test_size: float,
    random_state: int,
    shots_per_class: int,
) -> None:
    sequence_ids, sequence_labels, sequences = tsvm._group_sequences(row_meta, features)
    rows: list[dict[str, object]] = []
    all_indices = np.arange(len(sequences))

    for repeat_index in range(repeats):
        seed = random_state + repeat_index
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=seed,
            stratify=sequence_labels,
        )
        if shots_per_class > 0:
            rng = np.random.RandomState(seed)
            train_indices = tsvm._few_shot_index_subset(train_indices, sequence_labels, shots_per_class, rng)

        for split_name, indices in (("train", train_indices), ("test", test_indices)):
            for sequence_index in indices:
                rows.append(
                    {
                        "repeat": repeat_index,
                        "split": split_name,
                        "sequence_id": sequence_ids[int(sequence_index)],
                        "label": sequence_labels[int(sequence_index)],
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["repeat", "split", "sequence_id", "label"])
        writer.writeheader()
        writer.writerows(rows)


def load_split_manifest(manifest_path: Path) -> dict[int, dict[str, list[str]]]:
    grouped: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"train": [], "test": []})
    with manifest_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            repeat = int(row["repeat"])
            grouped[repeat][row["split"]].append(row["sequence_id"])
    return grouped


def compute_scores(y_true: list[str], y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def summarize_score_rows(score_rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in score_rows[0]:
        values = np.asarray([row[key] for row in score_rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    return out


def evaluate_classification(
    dataset_path: Path,
    split_manifest: Path,
    classifiers: list[str],
    feature_sets: list[str],
    pca_components: int = 0,
) -> dict[str, dict[str, float]]:
    row_meta, feature_names, features = load_rows(dataset_path)
    manifest = load_split_manifest(split_manifest)
    results: dict[str, dict[str, float]] = {}

    for feature_set in feature_sets:
        _, selected_features = tsvm._select_features(feature_names, features, feature_set)
        sequence_ids, sequence_labels, sequences = tsvm._group_sequences(row_meta, selected_features)
        sequence_lookup = {sequence_id: idx for idx, sequence_id in enumerate(sequence_ids)}

        for classifier in classifiers:
            score_rows: list[dict[str, float]] = []
            for repeat_index in sorted(manifest):
                train_seq_ids = manifest[repeat_index]["train"]
                test_seq_ids = manifest[repeat_index]["test"]
                train_indices = [sequence_lookup[sequence_id] for sequence_id in train_seq_ids]
                test_indices = [sequence_lookup[sequence_id] for sequence_id in test_seq_ids]

                train_sequences = [sequences[index] for index in train_indices]
                test_sequences = [sequences[index] for index in test_indices]
                y_train = [sequence_labels[index] for index in train_indices]
                y_test = [sequence_labels[index] for index in test_indices]

                if pca_components > 0:
                    train_sequences, test_sequences = tsvm._project_sequences_with_pca(
                        train_sequences,
                        test_sequences,
                        pca_components,
                        42 + repeat_index,
                    )

                x_train = np.stack([tsvm._aggregate_sequence_array(sequence) for sequence in train_sequences], axis=0)
                x_test = np.stack([tsvm._aggregate_sequence_array(sequence) for sequence in test_sequences], axis=0)

                args = SimpleNamespace(
                    classifier=classifier,
                    c=5.0,
                    gamma="scale",
                    knn_neighbors=3,
                    knn_weights="distance",
                    rf_estimators=200,
                    rf_max_depth=0,
                    mlp_hidden_sizes="128,64",
                    mlp_alpha=1e-4,
                    mlp_learning_rate=1e-3,
                    mlp_max_iter=1200,
                )
                model = tsvm._build_model(args, 42 + repeat_index)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score_rows.append(compute_scores(y_test, y_pred))

            results[f"{classifier}:{feature_set}"] = summarize_score_rows(score_rows)
    return results


def evaluate_optimizer_sequences(dataset_path: Path, selected_sequence_ids: list[str]) -> dict[str, object]:
    row_meta, feature_names, features = load_rows(dataset_path)
    raw_names = [name for name in feature_names if name.startswith("raw_")]
    # feature_names returned by loader excludes meta fields, so rebuild from csv header
    with dataset_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        raw_names = [name for name in (reader.fieldnames or []) if name.startswith("raw_")]
        all_rows = list(reader)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["sequence_id"]].append(row)
    for sequence_id in grouped:
        grouped[sequence_id].sort(key=lambda row: (int(float(row.get("frame_id") or 0)), float(row.get("timestamp_sec") or 0.0)))

    weights = OptimizationWeights()
    weighted_multipliers = {
        "landmark": weights.landmark,
        "palm": weights.palm,
        "prior": weights.prior,
        "temporal": weights.temporal,
        "acceleration": weights.acceleration,
        "default_pose": weights.default_pose,
        "boundary": weights.boundary,
    }

    summary: dict[str, object] = {
        "selected_sequences": selected_sequence_ids,
        "per_sequence": {},
        "global": {},
    }
    all_unweighted: dict[str, list[float]] = defaultdict(list)
    all_weighted: dict[str, list[float]] = defaultdict(list)
    all_success: list[bool] = []
    all_finite: list[bool] = []
    all_bounds: list[bool] = []
    spike_stats: dict[str, dict[str, float]] = {}
    lag_stats: dict[str, dict[str, object]] = {}

    with MujocoHandPoseOptimizer(version="v2") as optimizer, OrcaFeatureProjector(version="v2") as projector:
        for sequence_id in selected_sequence_ids:
            prev_action = None
            prev_prev_action = None
            sequence_unweighted: dict[str, list[float]] = defaultdict(list)
            sequence_weighted: dict[str, list[float]] = defaultdict(list)
            corrected_actions: list[np.ndarray] = []
            optimized_actions: list[np.ndarray] = []
            successes: list[bool] = []

            for row in grouped[sequence_id]:
                points = np.array([float(row[name]) for name in raw_names], dtype=np.float64).reshape(21, 3)
                corrected = projector.corrected_vector(points).astype(np.float64)
                result = optimizer.optimize(points, prev_action=prev_action, prev_prev_action=prev_prev_action)

                corrected_actions.append(corrected)
                optimized_actions.append(result.action.astype(np.float64))
                successes.append(bool(result.success))
                all_success.append(bool(result.success))

                finite_ok = (
                    np.isfinite(result.action).all()
                    and np.isfinite(result.optimized_full_points).all()
                    and all(np.isfinite(value) for value in result.loss_terms.values())
                )
                all_finite.append(bool(finite_ok))

                bounds_ok = np.all(result.action >= optimizer.env.action_low - 1e-6) and np.all(
                    result.action <= optimizer.env.action_high + 1e-6
                )
                all_bounds.append(bool(bounds_ok))

                for key, value in result.loss_terms.items():
                    if key == "total":
                        continue
                    numeric_value = float(value)
                    sequence_unweighted[key].append(numeric_value)
                    all_unweighted[key].append(numeric_value)
                    weighted_value = numeric_value * weighted_multipliers[key]
                    sequence_weighted[key].append(weighted_value)
                    all_weighted[key].append(weighted_value)

                old_prev = prev_action
                prev_action = result.action.astype(np.float64)
                prev_prev_action = old_prev

            corrected_array = np.stack(corrected_actions)
            optimized_array = np.stack(optimized_actions)
            corrected_velocity = np.linalg.norm(corrected_array[1:] - corrected_array[:-1], axis=1)
            optimized_velocity = np.linalg.norm(optimized_array[1:] - optimized_array[:-1], axis=1)
            corrected_acceleration = (
                np.linalg.norm(corrected_array[2:] - 2 * corrected_array[1:-1] + corrected_array[:-2], axis=1)
                if len(corrected_array) >= 3
                else np.array([], dtype=np.float64)
            )
            optimized_acceleration = (
                np.linalg.norm(optimized_array[2:] - 2 * optimized_array[1:-1] + optimized_array[:-2], axis=1)
                if len(optimized_array) >= 3
                else np.array([], dtype=np.float64)
            )

            spike_stats[sequence_id] = {
                "corrected_velocity_max": float(corrected_velocity.max()) if corrected_velocity.size else 0.0,
                "optimized_velocity_max": float(optimized_velocity.max()) if optimized_velocity.size else 0.0,
                "corrected_acceleration_max": float(corrected_acceleration.max()) if corrected_acceleration.size else 0.0,
                "optimized_acceleration_max": float(optimized_acceleration.max()) if optimized_acceleration.size else 0.0,
                "corrected_velocity_mean": float(corrected_velocity.mean()) if corrected_velocity.size else 0.0,
                "optimized_velocity_mean": float(optimized_velocity.mean()) if optimized_velocity.size else 0.0,
                "corrected_acceleration_mean": float(corrected_acceleration.mean()) if corrected_acceleration.size else 0.0,
                "optimized_acceleration_mean": float(optimized_acceleration.mean()) if optimized_acceleration.size else 0.0,
            }

            lags: list[int] = []
            for dim in range(corrected_array.shape[1]):
                corrected_dim = corrected_array[:, dim]
                optimized_dim = optimized_array[:, dim]
                if np.std(corrected_dim) < 1e-6 or np.std(optimized_dim) < 1e-6:
                    continue
                best_lag = 0
                best_corr = -2.0
                for lag in range(-3, 4):
                    if lag < 0:
                        x = corrected_dim[-lag:]
                        y = optimized_dim[: len(optimized_dim) + lag]
                    elif lag > 0:
                        x = corrected_dim[:-lag]
                        y = optimized_dim[lag:]
                    else:
                        x = corrected_dim
                        y = optimized_dim
                    if len(x) < 4:
                        continue
                    corr = np.corrcoef(x, y)[0, 1]
                    if np.isfinite(corr) and corr > best_corr:
                        best_corr = float(corr)
                        best_lag = lag
                lags.append(best_lag)

            lag_histogram: dict[str, int] = {}
            for lag in lags:
                lag_histogram[str(lag)] = lag_histogram.get(str(lag), 0) + 1
            lag_stats[sequence_id] = {
                "mean_best_lag_frames": float(np.mean(lags)) if lags else 0.0,
                "max_abs_best_lag_frames": int(max(abs(lag) for lag in lags)) if lags else 0,
                "lag_histogram": lag_histogram,
            }

            summary["per_sequence"][sequence_id] = {
                "num_frames": int(len(grouped[sequence_id])),
                "success_rate": float(np.mean(successes)) if successes else 0.0,
                "unweighted_mean": {key: float(np.mean(values)) for key, values in sequence_unweighted.items()},
                "weighted_mean": {key: float(np.mean(values)) for key, values in sequence_weighted.items()},
            }

        summary["global"] = {
            "loss_unweighted_mean": {key: float(np.mean(values)) for key, values in all_unweighted.items()},
            "loss_weighted_mean": {key: float(np.mean(values)) for key, values in all_weighted.items()},
            "success_rate": float(np.mean(all_success)) if all_success else 0.0,
            "finite_all_ok": bool(all(all_finite)),
            "bounds_all_ok": bool(all(all_bounds)),
            "spike_stats": spike_stats,
            "lag_stats": lag_stats,
        }

        low = optimizer.env.action_low.astype(np.float64)
        high = optimizer.env.action_high.astype(np.float64)
        rng = np.random.default_rng(123)
        target_action = low + 0.35 * (high - low)
        target_action += 0.15 * (high - low) * rng.uniform(-1.0, 1.0, size=low.shape[0])
        target_action = np.clip(target_action, low, high)
        observation = optimizer.full_landmarks_from_action(target_action)
        recovery = optimizer.optimize(observation)
        target_normal = palm_normal_vector(observation)
        _, recovered_normal = optimizer._forward_sparse_points(recovery.action.astype(np.float64))
        _, perfect_normal = optimizer._forward_sparse_points(target_action)

        summary["synthetic_recovery"] = {
            "success": bool(recovery.success),
            "iterations": int(recovery.iterations),
            "l2_action_error": float(np.linalg.norm(recovery.action.astype(np.float64) - target_action)),
            "mean_abs_action_error": float(np.mean(np.abs(recovery.action.astype(np.float64) - target_action))),
            "max_abs_action_error": float(np.max(np.abs(recovery.action.astype(np.float64) - target_action))),
            "loss_terms": {key: float(value) for key, value in recovery.loss_terms.items()},
            "target_vs_recovered_palm_dot": float(np.dot(target_normal, recovered_normal)),
        }
        summary["palm_consistency_check"] = {
            "target_normal": target_normal.tolist(),
            "current_normal": perfect_normal.tolist(),
            "dot_product": float(np.dot(target_normal, perfect_normal)),
            "perfect_palm_loss": float(np.sum((perfect_normal - target_normal) ** 2)),
        }

        # Lightweight generation check
        generated_rows = 0
        optimized_full_finite = True
        optimized_action_in_bounds = True
        for sequence_id in selected_sequence_ids:
            prev_action = None
            prev_prev_action = None
            for row in grouped[sequence_id][:10]:
                points = np.array([float(row[name]) for name in raw_names], dtype=np.float64).reshape(21, 3)
                result = optimizer.optimize(points, prev_action=prev_action, prev_prev_action=prev_prev_action)
                generated_rows += 1
                optimized_full_finite = optimized_full_finite and bool(np.isfinite(result.optimized_full_points).all())
                optimized_action_in_bounds = optimized_action_in_bounds and bool(
                    np.all(result.action >= optimizer.env.action_low - 1e-6)
                    and np.all(result.action <= optimizer.env.action_high + 1e-6)
                )
                old_prev = prev_action
                prev_action = result.action.astype(np.float64)
                prev_prev_action = old_prev
        summary["generation_check"] = {
            "checked_rows": generated_rows,
            "optimized_full_finite": optimized_full_finite,
            "optimized_action_in_bounds": optimized_action_in_bounds,
        }

    return summary


def compare_dicts(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    comparison: dict[str, object] = {}

    def fetch(obj: dict[str, object], path: list[str]) -> float:
        current: object = obj
        for key in path:
            current = current[key]  # type: ignore[index]
        return float(current)

    numeric_paths = {
        "palm_dot": ["palm_consistency_check", "dot_product"],
        "perfect_palm_loss": ["palm_consistency_check", "perfect_palm_loss"],
        "synthetic_l2_action_error": ["synthetic_recovery", "l2_action_error"],
        "synthetic_mean_abs_action_error": ["synthetic_recovery", "mean_abs_action_error"],
        "synthetic_landmark_loss": ["synthetic_recovery", "loss_terms", "landmark"],
        "synthetic_palm_loss": ["synthetic_recovery", "loss_terms", "palm"],
        "global_success_rate": ["global", "success_rate"],
    }
    for name, path in numeric_paths.items():
        before_value = fetch(before, path)
        after_value = fetch(after, path)
        comparison[name] = {
            "before": before_value,
            "after": after_value,
            "delta": after_value - before_value,
        }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre/post diagnostics for the ORCA palm-normal fix.")
    parser.add_argument("--dataset", default="gesture_sequence_dataset_chinese_dance_6class.csv")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["create-splits", "run"], required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--shots-per-class", type=int, default=3)
    args = parser.parse_args()

    dataset_path = (ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    split_manifest_path = (ROOT / args.split_manifest).resolve() if not Path(args.split_manifest).is_absolute() else Path(args.split_manifest)
    output_path = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    row_meta, feature_names, features = load_rows(dataset_path)
    labels = [row["label"] for row in row_meta]

    if args.mode == "create-splits":
        save_split_manifest(
            split_manifest_path,
            row_meta,
            labels,
            features,
            repeats=args.repeats,
            test_size=args.test_size,
            random_state=args.random_state,
            shots_per_class=args.shots_per_class,
        )
        print(f"saved_split_manifest={split_manifest_path}")
        return

    sequence_ids = sorted({row["sequence_id"] for row in row_meta if row.get("sequence_id")})[:3]
    diagnostics = evaluate_optimizer_sequences(dataset_path, sequence_ids)
    diagnostics["classification"] = evaluate_classification(
        dataset_path,
        split_manifest_path,
        classifiers=["svm", "knn", "rf", "mlp"],
        feature_sets=["raw", "corrected", "optimized_action", "optimized_full"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"diagnostics_json={output_path}")


if __name__ == "__main__":
    main()
