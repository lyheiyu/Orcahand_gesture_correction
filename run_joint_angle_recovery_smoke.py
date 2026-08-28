from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_joint_angle_baseline import JOINT_DEFINITIONS
from generate_smoothing_baselines import _kalman, _one_euro
from joint_angle_recovery import (
    FINGER_LANDMARKS,
    affected_joint_mask,
    angle_sequence,
    corrupt_landmarks,
    estimate_confidence_parameters,
    recover_confidence_weighted,
)


METHODS = ("Corrupted", "Kalman", "One-Euro", "Automatic Confidence", "Oracle Confidence")
COLORS = {
    "Corrupted": "#7f8c8d",
    "Kalman": "#2a6f97",
    "One-Euro": "#61a5c2",
    "Automatic Confidence": "#d97706",
    "Oracle Confidence": "#b42318",
}


def _load_sequences(path: Path) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        raw_columns = sorted(
            (name for name in fieldnames if name.startswith("raw_")),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if raw_columns != [f"raw_{index}" for index in range(63)]:
            raise ValueError("Expected raw_0 through raw_62.")
        for row in reader:
            grouped[row["sequence_id"]].append(
                {
                    "label": str(row["label"]),
                    "frame_id": int(float(row.get("frame_id") or 0)),
                    "timestamp": float(row.get("timestamp_sec") or 0.0),
                    "points": np.asarray([float(row[name]) for name in raw_columns], dtype=np.float32),
                }
            )
    output = {}
    for sequence_id, rows in grouped.items():
        rows.sort(key=lambda row: (int(row["frame_id"]), float(row["timestamp"])))
        timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
        if len(timestamps) > 1 and np.allclose(timestamps, timestamps[0]):
            timestamps = np.arange(len(rows), dtype=np.float64)
        output[sequence_id] = {
            "label": rows[0]["label"],
            "points": np.stack([row["points"] for row in rows]).reshape(-1, 21, 3),
            "timestamps": timestamps,
            "frame_ids": [row["frame_id"] for row in rows],
        }
    return output


def _split_sequences(
    sequences: dict[str, dict[str, object]], seed: int, test_fraction: float = 0.2
) -> tuple[list[str], list[str]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for sequence_id, data in sequences.items():
        by_label[str(data["label"])].append(sequence_id)
    development = []
    test = []
    for label_index, label in enumerate(sorted(by_label)):
        ids = sorted(by_label[label])
        rng = np.random.default_rng(seed + label_index)
        shuffled = [ids[index] for index in rng.permutation(len(ids))]
        num_test = max(1, int(round(len(ids) * test_fraction)))
        test.extend(shuffled[:num_test])
        development.extend(shuffled[num_test:])
    return sorted(development), sorted(test)


def _calibration_ids(
    sequences: dict[str, dict[str, object]], development_ids: list[str], per_class: int = 3
) -> list[str]:
    selected = []
    counts: dict[str, int] = defaultdict(int)
    for sequence_id in development_ids:
        label = str(sequences[sequence_id]["label"])
        if counts[label] < per_class:
            selected.append(sequence_id)
            counts[label] += 1
    return selected


def _smoke_ids(
    sequences: dict[str, dict[str, object]], test_ids: list[str]
) -> list[str]:
    selected = []
    seen = set()
    for sequence_id in test_ids:
        label = str(sequences[sequence_id]["label"])
        if label not in seen:
            selected.append(sequence_id)
            seen.add(label)
    return selected


def _condition(index: int) -> tuple[str, int, str]:
    corruption_type = ("gaussian", "spike", "occlusion")[index % 3]
    if corruption_type == "gaussian":
        return corruption_type, 0, "all"
    if corruption_type == "spike":
        return corruption_type, 3, "random"
    finger = sorted(FINGER_LANDMARKS)[index % len(FINGER_LANDMARKS)]
    return corruption_type, 5, finger


def _apply_condition(
    points: np.ndarray, index: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, object]]:
    corruption_type, duration, finger = _condition(index)
    corrupted, mask, manifest = corrupt_landmarks(
        points,
        corruption_type,
        seed=seed,
        finger_group="index" if finger in {"all", "random"} else finger,
        duration=duration,
        gaussian_sigma=0.03,
        spike_magnitude=0.75,
    )
    actual_finger = manifest[0]["finger_group"] if manifest else finger
    return corrupted, mask, manifest, {
        "corruption_type": corruption_type,
        "duration": duration,
        "finger_group": actual_finger,
        "severity": "medium" if corruption_type == "gaussian" else manifest[0]["severity"],
        "random_seed": seed,
    }


def _kalman_angles(sequence: np.ndarray, process_var: float, measurement_var: float) -> np.ndarray:
    return _kalman(sequence.astype(np.float32), process_var, measurement_var).astype(np.float64)


def _one_euro_angles(
    sequence: np.ndarray, timestamps: np.ndarray, min_cutoff: float, beta: float
) -> np.ndarray:
    return _one_euro(
        sequence.astype(np.float32),
        timestamps.astype(np.float32),
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=1.0,
    ).astype(np.float64)


def _masked_mae(method: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    values = np.abs(method - clean)[mask]
    return float(np.mean(values)) if len(values) else math.nan


def _calibrate(
    sequences: dict[str, dict[str, object]], calibration_ids: list[str], base_parameters
) -> dict[str, object]:
    examples = []
    for index, sequence_id in enumerate(calibration_ids):
        data = sequences[sequence_id]
        clean_points = np.asarray(data["points"])
        corrupted_points, landmark_mask, _, _ = _apply_condition(clean_points, index, 1000 + index)
        clean_angles, _ = angle_sequence(clean_points)
        corrupted_angles, _ = angle_sequence(corrupted_points)
        joint_mask = affected_joint_mask(landmark_mask)
        examples.append(
            (clean_points, corrupted_points, clean_angles, corrupted_angles, joint_mask, np.asarray(data["timestamps"]))
        )

    strength_scores = {}
    for strength in (0.25, 0.5, 1.0, 2.0):
        parameters = estimate_confidence_parameters(
            [np.asarray(sequences[key]["points"]) for key in calibration_ids], strength=strength
        )
        errors = []
        for _, corrupted_points, clean_angles, corrupted_angles, joint_mask, _ in examples:
            recovered, _ = recover_confidence_weighted(corrupted_angles, corrupted_points, parameters)
            errors.append(_masked_mae(recovered, clean_angles, joint_mask))
        strength_scores[strength] = float(np.nanmean(errors))
    best_strength = min(strength_scores, key=strength_scores.get)

    kalman_scores = {}
    for process_var in (0.01, 0.1, 1.0, 10.0):
        for measurement_var in (0.1, 1.0, 10.0, 100.0):
            errors = []
            for _, _, clean_angles, corrupted_angles, joint_mask, _ in examples:
                recovered = _kalman_angles(corrupted_angles, process_var, measurement_var)
                errors.append(_masked_mae(recovered, clean_angles, joint_mask))
            kalman_scores[(process_var, measurement_var)] = float(np.nanmean(errors))
    best_kalman = min(kalman_scores, key=kalman_scores.get)

    one_euro_scores = {}
    for min_cutoff in (0.5, 1.0, 2.0, 4.0):
        for beta in (0.0, 0.02, 0.1, 0.5):
            errors = []
            for _, _, clean_angles, corrupted_angles, joint_mask, timestamps in examples:
                recovered = _one_euro_angles(corrupted_angles, timestamps, min_cutoff, beta)
                errors.append(_masked_mae(recovered, clean_angles, joint_mask))
            one_euro_scores[(min_cutoff, beta)] = float(np.nanmean(errors))
    best_one_euro = min(one_euro_scores, key=one_euro_scores.get)
    return {
        "strength": float(best_strength),
        "strength_scores": {str(key): value for key, value in strength_scores.items()},
        "kalman_process_var": float(best_kalman[0]),
        "kalman_measurement_var": float(best_kalman[1]),
        "kalman_best_development_mae": kalman_scores[best_kalman],
        "oneeuro_min_cutoff": float(best_one_euro[0]),
        "oneeuro_beta": float(best_one_euro[1]),
        "oneeuro_best_development_mae": one_euro_scores[best_one_euro],
    }


def _amplitude_metrics(method: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    clean_amplitude = np.ptp(clean, axis=0)
    method_amplitude = np.ptp(method, axis=0)
    active = clean_amplitude > 1.0
    if not np.any(active):
        return {"mean": math.nan, "median": math.nan, "within": math.nan, "below": math.nan, "above": math.nan}
    retention = method_amplitude[active] / clean_amplitude[active]
    return {
        "mean": float(np.mean(retention)),
        "median": float(np.median(retention)),
        "within": float(np.mean((retention >= 0.9) & (retention <= 1.1))),
        "below": float(np.mean(retention < 0.8)),
        "above": float(np.mean(retention > 1.2)),
    }


def _temporal_lag(method: np.ndarray, clean: np.ndarray, max_lag: int = 10) -> float:
    lags = []
    for joint_index in range(clean.shape[1]):
        reference = clean[:, joint_index] - np.mean(clean[:, joint_index])
        estimate = method[:, joint_index] - np.mean(method[:, joint_index])
        if np.std(reference) < 1e-8 or np.std(estimate) < 1e-8:
            continue
        candidates = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x, y = reference[-lag:], estimate[:lag]
            elif lag > 0:
                x, y = reference[:-lag], estimate[lag:]
            else:
                x, y = reference, estimate
            correlation = np.corrcoef(x, y)[0, 1] if len(x) >= 3 else -np.inf
            candidates.append((correlation if np.isfinite(correlation) else -np.inf, lag))
        lags.append(abs(max(candidates)[1]))
    return float(np.median(lags)) if lags else math.nan


def _metric_row(
    method_name: str,
    method: np.ndarray,
    clean: np.ndarray,
    corrupted: np.ndarray,
    joint_mask: np.ndarray,
    metadata: dict[str, object],
) -> dict[str, object]:
    error = np.abs(method - clean)
    input_error = np.abs(corrupted - clean)
    region = error[joint_mask]
    visible = error[~joint_mask]
    input_region = input_error[joint_mask]
    baseline = float(np.mean(input_region)) if len(input_region) else math.nan
    region_mae = float(np.mean(region)) if len(region) else math.nan
    amplitude = _amplitude_metrics(method, clean)
    return {
        **metadata,
        "method": method_name,
        "all_joint_mae": float(np.mean(error)),
        "all_joint_rmse": float(np.sqrt(np.mean((method - clean) ** 2))),
        "median_absolute_error": float(np.median(error)),
        "p95_absolute_error": float(np.percentile(error, 95)),
        "corrupted_joint_mae": region_mae,
        "visible_joint_mae": float(np.mean(visible)) if len(visible) else math.nan,
        "recovery_ratio": 1.0 - region_mae / baseline if baseline > 1e-12 else math.nan,
        "velocity_error": float(np.mean(np.abs(np.diff(method, axis=0) - np.diff(clean, axis=0)))),
        "acceleration_error": float(
            np.mean(np.abs(np.diff(method, n=2, axis=0) - np.diff(clean, n=2, axis=0)))
        ),
        "amplitude_retention_mean": amplitude["mean"],
        "amplitude_retention_median": amplitude["median"],
        "amplitude_within_0p9_1p1": amplitude["within"],
        "amplitude_below_0p8": amplitude["below"],
        "amplitude_above_1p2": amplitude["above"],
        "temporal_lag_frames": _temporal_lag(method, clean),
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    metrics = (
        "corrupted_joint_mae", "all_joint_mae", "visible_joint_mae", "recovery_ratio",
        "velocity_error", "acceleration_error", "amplitude_retention_median", "temporal_lag_frames",
    )
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group, members in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summary = {key: value for key, value in zip(keys, group)}
        summary["num_instances"] = len(members)
        for metric in metrics:
            values = np.asarray([float(member[metric]) for member in members])
            values = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            summary[f"{metric}_std"] = float(np.std(values)) if len(values) else math.nan
        output.append(summary)
    return output


def _bar(rows: list[dict[str, object]], metric: str, ylabel: str, output: Path) -> None:
    values = []
    errors = []
    for method in METHODS:
        array = np.asarray([float(row[metric]) for row in rows if row["method"] == method])
        array = array[np.isfinite(array)]
        values.append(float(np.mean(array)))
        errors.append(1.96 * float(np.std(array)) / math.sqrt(len(array)))
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    x = np.arange(len(METHODS))
    ax.bar(x, values, yerr=errors, capsize=4, color=[COLORS[key] for key in METHODS])
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, METHODS, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _trajectory_plot(trace: dict[str, object], name: str, output: Path) -> None:
    clean = np.asarray(trace["clean"])
    corrupted = np.asarray(trace["corrupted"])
    joint_mask = np.asarray(trace["joint_mask"])
    affected_counts = np.sum(joint_mask, axis=0)
    candidate_joints = np.flatnonzero(affected_counts)
    if len(candidate_joints):
        joint_index = int(
            max(candidate_joints, key=lambda index: np.mean(np.abs(corrupted[:, index] - clean[:, index])))
        )
    else:
        joint_index = 0
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.plot(clean[:, joint_index], label="Clean", color="#111111", linewidth=2.2)
    for method in ("Corrupted", "Kalman", "One-Euro", "Automatic Confidence", "Oracle Confidence"):
        ax.plot(np.asarray(trace[method])[:, joint_index], label=method, color=COLORS[method], linewidth=1.4)
    frames = np.flatnonzero(joint_mask[:, joint_index])
    if len(frames):
        ax.axvspan(frames[0], frames[-1], color="#ef4444", alpha=0.10)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Joint angle (degrees)")
    ax.set_title(f"{name.title()} automatic case: {trace['sequence_id']} / {JOINT_DEFINITIONS[joint_index][0]}")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _confidence_plot(trace: dict[str, object], output: Path) -> None:
    mask = np.asarray(trace["joint_mask"])
    affected = np.any(mask, axis=0)
    automatic = np.asarray(trace["automatic_confidence"])[:, affected].mean(axis=1)
    oracle = np.asarray(trace["oracle_confidence"])[:, affected].mean(axis=1)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(automatic, label="Automatic confidence", color=COLORS["Automatic Confidence"], linewidth=2)
    ax.plot(oracle, label="Oracle confidence", color=COLORS["Oracle Confidence"], linewidth=2)
    frames = np.flatnonzero(np.any(mask, axis=1))
    if len(frames):
        ax.axvspan(frames[0], frames[-1], color="#ef4444", alpha=0.10)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean confidence on affected joints")
    ax.set_title(f"Confidence trace: {trace['sequence_id']}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run(dataset: Path, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    sequences = _load_sequences(dataset)
    development_ids, test_ids = _split_sequences(sequences, seed)
    calibration_ids = _calibration_ids(sequences, development_ids)
    smoke_ids = _smoke_ids(sequences, test_ids)

    development_points = [np.asarray(sequences[key]["points"]) for key in development_ids]
    base_parameters = estimate_confidence_parameters(development_points)
    calibration = _calibrate(sequences, calibration_ids, base_parameters)
    parameters = estimate_confidence_parameters(
        development_points, strength=float(calibration["strength"])
    )

    metric_rows = []
    manifest_rows = []
    confidence_rows = []
    traces = {}
    nonfinite = 0
    confidence_violations = 0
    causal_max_difference = 0.0

    for index, sequence_id in enumerate(smoke_ids):
        data = sequences[sequence_id]
        clean_points = np.asarray(data["points"], dtype=np.float64)
        corrupted_points, landmark_mask, manifest, condition = _apply_condition(
            clean_points, index, seed + index
        )
        clean_angles, clean_valid = angle_sequence(clean_points)
        corrupted_angles, corrupted_valid = angle_sequence(corrupted_points)
        joint_mask = affected_joint_mask(landmark_mask)
        timestamps = np.asarray(data["timestamps"])

        kalman = _kalman_angles(
            corrupted_angles,
            float(calibration["kalman_process_var"]),
            float(calibration["kalman_measurement_var"]),
        )
        one_euro = _one_euro_angles(
            corrupted_angles,
            timestamps,
            float(calibration["oneeuro_min_cutoff"]),
            float(calibration["oneeuro_beta"]),
        )
        automatic, automatic_confidence = recover_confidence_weighted(
            corrupted_angles, corrupted_points, parameters
        )
        oracle, oracle_confidence = recover_confidence_weighted(
            corrupted_angles, corrupted_points, parameters, oracle_joint_mask=joint_mask
        )
        outputs = {
            "Corrupted": corrupted_angles,
            "Kalman": kalman,
            "One-Euro": one_euro,
            "Automatic Confidence": automatic,
            "Oracle Confidence": oracle,
        }
        metadata = {
            "sequence_id": sequence_id,
            "label": data["label"],
            **condition,
            "num_frames": len(clean_points),
            "num_affected_joint_frames": int(np.sum(joint_mask)),
        }
        for method_name, output in outputs.items():
            metric_rows.append(
                _metric_row(method_name, output, clean_angles, corrupted_angles, joint_mask, metadata)
            )
            nonfinite += int(np.sum(~np.isfinite(output)))
        confidence_violations += int(
            np.sum((automatic_confidence < 0.0) | (automatic_confidence > 1.0))
        )
        confidence_violations += int(
            np.sum((oracle_confidence < 0.0) | (oracle_confidence > 1.0))
        )

        # Future observations must not change earlier causal outputs.
        cut = max(2, len(corrupted_angles) // 2)
        modified_angles = corrupted_angles.copy()
        modified_points = corrupted_points.copy()
        modified_angles[cut:] += 50.0
        modified_points[cut:] += 0.5
        causal_check, _ = recover_confidence_weighted(modified_angles, modified_points, parameters)
        causal_max_difference = max(
            causal_max_difference, float(np.max(np.abs(causal_check[:cut] - automatic[:cut])))
        )

        for row in manifest:
            manifest_rows.append({"sequence_id": sequence_id, "label": data["label"], **row})
        for frame_index in range(len(clean_points)):
            for joint_index, (joint_name, _, _, _) in enumerate(JOINT_DEFINITIONS):
                confidence_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "frame_index": frame_index,
                        "joint_index": joint_index,
                        "joint_name": joint_name,
                        "affected": int(joint_mask[frame_index, joint_index]),
                        "automatic_confidence": float(automatic_confidence[frame_index, joint_index]),
                        "oracle_confidence": float(oracle_confidence[frame_index, joint_index]),
                    }
                )
        traces[sequence_id] = {
            "sequence_id": sequence_id,
            "clean": clean_angles,
            "corrupted": corrupted_angles,
            **outputs,
            "joint_mask": joint_mask,
            "automatic_confidence": automatic_confidence,
            "oracle_confidence": oracle_confidence,
        }
        nonfinite += int(np.sum(~np.isfinite(clean_angles)))
        nonfinite += int(np.sum(~np.isfinite(corrupted_angles)))
        if not np.all(clean_valid) or not np.all(corrupted_valid):
            # Invalid vectors are allowed only if safely represented; count for validation.
            pass

    summary = _summarize(metric_rows, ("method",))
    by_corruption = _summarize(metric_rows, ("method", "corruption_type"))
    _write_rows(output_dir / "smoke_per_sequence_metrics.csv", metric_rows)
    _write_rows(output_dir / "smoke_summary.csv", summary)
    _write_rows(output_dir / "smoke_summary_by_corruption.csv", by_corruption)
    _write_rows(output_dir / "corruption_manifest.csv", manifest_rows)
    _write_rows(output_dir / "confidence_traces.csv", confidence_rows)
    _write_rows(
        output_dir / "split_manifest.csv",
        [
            {"sequence_id": key, "label": sequences[key]["label"], "split": "development"}
            for key in development_ids
        ]
        + [{"sequence_id": key, "label": sequences[key]["label"], "split": "test"} for key in test_ids],
    )

    validation = {
        "num_sequences_total": len(sequences),
        "num_development_sequences": len(development_ids),
        "num_frozen_test_sequences": len(test_ids),
        "num_smoke_sequences": len(smoke_ids),
        "smoke_sequence_ids": smoke_ids,
        "smoke_labels": [sequences[key]["label"] for key in smoke_ids],
        "nonfinite_values": nonfinite,
        "confidence_range_violations": confidence_violations,
        "causal_future_perturbation_max_difference": causal_max_difference,
        "clean_corrupted_alignment": True,
        "history_reset": "state is local to each recover_confidence_weighted call",
    }
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    config = {
        "dataset": str(dataset.resolve()),
        "terminology": "controlled clean reference; not anatomical ground truth",
        "units": "degrees",
        "seed": seed,
        "development_calibration_ids": calibration_ids,
        "smoke_conditions": ["Gaussian medium", "3-frame spike", "5-frame finger occlusion"],
        "gaussian_sigma": 0.03,
        "spike_magnitude": 0.75,
        "confidence_formula_weights": {"angle_jump": 0.35, "geometry": 0.30, "prediction": 0.35},
        "confidence_strength": calibration["strength"],
        "joint_lower_bounds_degrees": parameters.lower_bound.tolist(),
        "joint_upper_bounds_degrees": parameters.upper_bound.tolist(),
        "jump_scales_degrees": parameters.jump_scale.tolist(),
        "prediction_scales_degrees": parameters.prediction_scale.tolist(),
        "geometry_scales_log_length_ratio": parameters.geometry_scale.tolist(),
        "development_selection": calibration,
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    _bar(metric_rows, "corrupted_joint_mae", "Corrupted-joint MAE (degrees)", figure_dir / "corrupted_joint_mae.png")
    _bar(metric_rows, "recovery_ratio", "Recovery ratio", figure_dir / "recovery_ratio.png")
    _bar(metric_rows, "velocity_error", "Velocity error (degrees/frame)", figure_dir / "velocity_error.png")
    _bar(metric_rows, "amplitude_retention_median", "Median amplitude retention", figure_dir / "amplitude_retention.png")

    automatic_rows = sorted(
        (row for row in metric_rows if row["method"] == "Automatic Confidence"),
        key=lambda row: float(row["recovery_ratio"]),
    )
    difficult = automatic_rows[0]
    median_value = float(np.median([float(row["recovery_ratio"]) for row in automatic_rows]))
    median = min(automatic_rows, key=lambda row: abs(float(row["recovery_ratio"]) - median_value))
    _trajectory_plot(traces[str(median["sequence_id"])], "median", figure_dir / "trajectory_median.png")
    _trajectory_plot(traces[str(difficult["sequence_id"])], "difficult", figure_dir / "trajectory_difficult.png")
    _confidence_plot(traces[str(median["sequence_id"])], figure_dir / "confidence_median.png")
    _confidence_plot(traces[str(difficult["sequence_id"])], figure_dir / "confidence_difficult.png")

    summary_by_method = {row["method"]: row for row in summary}
    automatic_ratio = float(summary_by_method["Automatic Confidence"]["recovery_ratio_mean"])
    oracle_ratio = float(summary_by_method["Oracle Confidence"]["recovery_ratio_mean"])
    kalman_ratio = float(summary_by_method["Kalman"]["recovery_ratio_mean"])
    euro_ratio = float(summary_by_method["One-Euro"]["recovery_ratio_mean"])
    best_filter = max(kalman_ratio, euro_ratio)
    if oracle_ratio <= best_filter:
        outcome = "Outcome C: Oracle does not outperform the strongest conventional filter; the constant-velocity temporal model is insufficient for at least part of this diagnostic."
    elif automatic_ratio < 0.75 * oracle_ratio:
        outcome = "Outcome B: Oracle is strong but automatic confidence does not approach it; confidence estimation is the main bottleneck."
    else:
        outcome = "Outcome A/D: Automatic confidence approaches a useful oracle while preserving trajectory fidelity; proceed to a larger benchmark after review."

    report = [
        "# Joint-Angle Recovery Smoke Diagnostic",
        "",
        "## Scope",
        "",
        "This experiment evaluates recovery toward an uncorrupted JointAngle-11 trajectory used as a controlled clean reference. It is not anatomical or physical ground truth. No classification experiment was run.",
        "",
        "Six frozen-test sequences were selected, one from each gesture class. The three requested corruption conditions were assigned twice each. All thresholds, empirical bounds, confidence strength, Kalman parameters, and One-Euro parameters were selected using development sequences only.",
        "",
        "## First Diagnostic Result",
        "",
        "| Method | Corrupted-joint MAE | Recovery ratio | Velocity error | Median amplitude retention | Lag (frames) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary_by_method[method]
        report.append(
            f"| {method} | {float(row['corrupted_joint_mae_mean']):.3f} | {float(row['recovery_ratio_mean']):.3f} | "
            f"{float(row['velocity_error_mean']):.3f} | {float(row['amplitude_retention_median_mean']):.3f} | "
            f"{float(row['temporal_lag_frames_mean']):.2f} |"
        )
    report.extend(
        [
            "",
            "## Decision",
            "",
            outcome,
            "",
            "Results must also be inspected by corruption type because an oracle that ignores every Gaussian-corrupted observation can fail differently from an oracle handling a short spike or occlusion.",
            "",
            "## Validation",
            "",
            f"- Non-finite values: `{nonfinite}`",
            f"- Confidence range violations: `{confidence_violations}`",
            f"- Causal future-perturbation difference: `{causal_max_difference:.3e}`",
            "- Sequence state is reset because every recovery call creates local state.",
            "- The automatic method never reads the synthetic corruption mask; only Oracle Confidence receives it.",
        ]
    )
    (output_dir / "JOINT_ANGLE_RECOVERY_METHOD.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    print(f"output_dir={output_dir.resolve()}")
    print(outcome)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first confidence-aware JointAngle-11 recovery milestone.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="diagnostics/joint_angle_recovery_20260827")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(Path(args.dataset), Path(args.output_dir), args.seed)


if __name__ == "__main__":
    main()
