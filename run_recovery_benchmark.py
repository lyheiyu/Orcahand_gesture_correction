from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_smoothing_baselines import _kalman, _one_euro
from orca_sim.mujoco_optimizer import MujocoHandPoseOptimizer


METHOD_ORDER = ("Corrupted Input", "Kalman", "One-Euro", "Corrected", "Fixed OA")
METHOD_COLORS = {
    "Corrupted Input": "#7f8c8d",
    "Kalman": "#2a6f97",
    "One-Euro": "#61a5c2",
    "Corrected": "#d97706",
    "Fixed OA": "#b42318",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _raw_columns(fieldnames: list[str]) -> list[str]:
    columns = [name for name in fieldnames if name.startswith("raw_")]
    columns.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    if columns != [f"raw_{index}" for index in range(63)]:
        raise ValueError("Expected raw_0 through raw_62.")
    return columns


def _load_sequences(path: Path) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        raw_columns = _raw_columns(fieldnames)
        corrected_columns = [f"corrected_{index}" for index in range(17)]
        oa_columns = [f"optimized_action_{index}" for index in range(17)]
        full_columns = [f"optimized_full_{index}" for index in range(63)]
        has_corrected = all(name in fieldnames for name in corrected_columns)
        has_oa = all(name in fieldnames for name in oa_columns)
        has_full = all(name in fieldnames for name in full_columns)
        for row in reader:
            reduced: dict[str, object] = {
                "label": str(row["label"]),
                "frame_id": str(row.get("frame_id", "0")),
                "timestamp_sec": float(row.get("timestamp_sec") or 0.0),
                "points": np.asarray([float(row[name]) for name in raw_columns], dtype=np.float32),
            }
            if has_corrected:
                reduced["corrected_action"] = np.asarray(
                    [float(row[name]) for name in corrected_columns], dtype=np.float32
                )
            if has_oa:
                reduced["optimized_action"] = np.asarray(
                    [float(row[name]) for name in oa_columns], dtype=np.float32
                )
            if has_full:
                reduced["optimized_full"] = np.asarray(
                    [float(row[name]) for name in full_columns], dtype=np.float32
                )
            grouped[row["sequence_id"]].append(reduced)

    result: dict[str, dict[str, object]] = {}
    for sequence_id, sequence_rows in grouped.items():
        sequence_rows.sort(
            key=lambda row: (
                int(float(str(row.get("frame_id") or 0))),
                float(row.get("timestamp_sec") or 0.0),
            )
        )
        points = np.stack([np.asarray(row["points"]) for row in sequence_rows]).reshape(-1, 21, 3)
        timestamps = np.asarray(
            [float(row.get("timestamp_sec") or 0.0) for row in sequence_rows], dtype=np.float64
        )
        if len(timestamps) > 1 and np.allclose(timestamps, timestamps[0]):
            timestamps = np.arange(len(timestamps), dtype=np.float64)
        result[sequence_id] = {
            "label": str(sequence_rows[0]["label"]),
            "points": points,
            "timestamps": timestamps,
            "frame_ids": [str(row.get("frame_id", index)) for index, row in enumerate(sequence_rows)],
        }
        if "corrected_action" in sequence_rows[0]:
            result[sequence_id]["corrected_action"] = np.stack(
                [np.asarray(row["corrected_action"]) for row in sequence_rows]
            )
        if "optimized_action" in sequence_rows[0]:
            result[sequence_id]["optimized_action"] = np.stack(
                [np.asarray(row["optimized_action"]) for row in sequence_rows]
            )
        if "optimized_full" in sequence_rows[0]:
            result[sequence_id]["optimized_full"] = np.stack(
                [np.asarray(row["optimized_full"]) for row in sequence_rows]
            ).reshape(-1, 21, 3)
    return result


def _load_scenarios(path: Path) -> dict[str, dict[str, str]]:
    _, rows = _read_csv(path)
    return {row["sequence_id"]: row for row in rows}


def _load_masks(path: Path, lengths: dict[str, int]) -> dict[str, np.ndarray]:
    masks = {sequence_id: np.zeros((length, 21), dtype=bool) for sequence_id, length in lengths.items()}
    _, rows = _read_csv(path)
    for row in rows:
        sequence_id = row["sequence_id"]
        if sequence_id not in masks:
            continue
        frame_index = int(row["frame_index"])
        landmark_id = int(row["landmark_id"])
        masks[sequence_id][frame_index, landmark_id] = True
    return masks


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _filter_sequence(points: np.ndarray, timestamps: np.ndarray, method: str) -> np.ndarray:
    flat = points.reshape(len(points), -1).astype(np.float32)
    if method == "Kalman":
        output = _kalman(flat, process_var=1e-4, measurement_var=1e-2)
    elif method == "One-Euro":
        output = _one_euro(
            flat,
            timestamps.astype(np.float32),
            min_cutoff=1.0,
            beta=0.02,
            d_cutoff=1.0,
        )
    else:
        raise ValueError(method)
    return output.reshape(-1, 21, 3).astype(np.float64)


def _run_corrected(
    optimizer: MujocoHandPoseOptimizer, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    actions = []
    landmarks = []
    for frame in points:
        action = optimizer.projector.corrected_vector(frame).astype(np.float64)
        actions.append(action)
        landmarks.append(optimizer.full_landmarks_from_action(action).astype(np.float64))
    return np.asarray(actions), np.asarray(landmarks)


def _run_oa(
    optimizer: MujocoHandPoseOptimizer, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    actions = []
    landmarks = []
    prev_action = None
    prev_prev_action = None
    successes = 0
    iterations = []
    solve_times = []
    for frame in points:
        result = optimizer.optimize(
            frame,
            prev_action=prev_action,
            prev_prev_action=prev_prev_action,
        )
        old_prev_action = prev_action
        prev_action = result.action.astype(np.float64)
        prev_prev_action = old_prev_action
        actions.append(prev_action)
        landmarks.append(result.optimized_full_points.astype(np.float64))
        successes += int(result.success)
        iterations.append(result.iterations)
        solve_times.append(result.solve_time_ms)
    return (
        np.asarray(actions),
        np.asarray(landmarks),
        {
            "success_rate": successes / max(len(points), 1),
            "iterations_mean": float(np.mean(iterations)),
            "solve_time_ms_mean": float(np.mean(solve_times)),
            "solve_time_ms_p95": float(np.percentile(solve_times, 95)),
        },
    )


def _values(error: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return error.reshape(-1)
    selected = error[mask]
    return selected[np.isfinite(selected)]


def _stats(error: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    values = _values(error, mask)
    if len(values) == 0:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "p95": math.nan}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _safe_recovery(error: float, baseline: float) -> float:
    if not np.isfinite(error) or not np.isfinite(baseline) or baseline <= 1e-12:
        return math.nan
    return float(1.0 - error / baseline)


def _motion_error(method: np.ndarray, clean: np.ndarray, order: int) -> float:
    if len(method) <= order:
        return math.nan
    method_delta = np.diff(method, n=order, axis=0)
    clean_delta = np.diff(clean, n=order, axis=0)
    return float(np.mean(np.linalg.norm(method_delta - clean_delta, axis=2)))


def _amplitude_metrics(method: np.ndarray, clean: np.ndarray, active_threshold: float) -> dict[str, float]:
    method_flat = method.reshape(len(method), -1)
    clean_flat = clean.reshape(len(clean), -1)
    clean_amplitude = np.ptp(clean_flat, axis=0)
    method_amplitude = np.ptp(method_flat, axis=0)
    active = clean_amplitude > active_threshold
    if not np.any(active):
        return {"mean": math.nan, "median": math.nan, "within_0p9_1p1": math.nan, "below_0p8": math.nan}
    retention = method_amplitude[active] / clean_amplitude[active]
    return {
        "mean": float(np.mean(retention)),
        "median": float(np.median(retention)),
        "within_0p9_1p1": float(np.mean((retention >= 0.9) & (retention <= 1.1))),
        "below_0p8": float(np.mean(retention < 0.8)),
    }


def _landmark_metric_row(
    method: str,
    clean_reference: np.ndarray,
    corrupted_input: np.ndarray,
    clean_output: np.ndarray,
    corrupt_output: np.ndarray,
    mask: np.ndarray,
    scenario: dict[str, str],
) -> dict[str, object]:
    input_error = np.linalg.norm(corrupted_input - clean_reference, axis=2)
    absolute_error = np.linalg.norm(corrupt_output - clean_reference, axis=2)
    clean_bias = np.linalg.norm(clean_output - clean_reference, axis=2)
    sensitivity = np.linalg.norm(corrupt_output - clean_output, axis=2)
    visible_mask = ~mask

    input_all = _stats(input_error)
    input_region = _stats(input_error, mask)
    absolute_all = _stats(absolute_error)
    absolute_region = _stats(absolute_error, mask)
    visible = _stats(absolute_error, visible_mask)
    bias = _stats(clean_bias)
    sensitivity_all = _stats(sensitivity)
    sensitivity_region = _stats(sensitivity, mask)
    amplitude = _amplitude_metrics(corrupt_output, clean_reference, active_threshold=1e-3)
    own_amplitude = _amplitude_metrics(corrupt_output, clean_output, active_threshold=1e-3)

    row: dict[str, object] = {
        "sequence_id": scenario["sequence_id"],
        "label": scenario["label"],
        "method": method,
        "corruption_type": scenario["corruption_type"],
        "severity": scenario["severity"],
        "duration": int(scenario["duration"]),
        "finger_group": scenario["finger_group"],
        "condition_id": scenario["condition_id"],
        "num_frames": len(clean_reference),
        "num_corrupted_pairs": int(np.sum(mask)),
        "input_error_mean": input_all["mean"],
        "input_region_error_mean": input_region["mean"],
        "landmark_error_mean": absolute_all["mean"],
        "landmark_error_median": absolute_all["median"],
        "landmark_error_std": absolute_all["std"],
        "landmark_error_p95": absolute_all["p95"],
        "corrupted_region_error_mean": absolute_region["mean"],
        "corrupted_region_error_median": absolute_region["median"],
        "corrupted_region_error_std": absolute_region["std"],
        "corrupted_region_error_p95": absolute_region["p95"],
        "visible_region_error_mean": visible["mean"],
        "clean_baseline_error_mean": bias["mean"],
        "corruption_sensitivity_mean": sensitivity_all["mean"],
        "corruption_sensitivity_region_mean": sensitivity_region["mean"],
        "direct_recovery_ratio_all": _safe_recovery(absolute_all["mean"], input_all["mean"]),
        "direct_recovery_ratio_region": _safe_recovery(absolute_region["mean"], input_region["mean"]),
        "robust_recovery_ratio_all": _safe_recovery(sensitivity_all["mean"], input_all["mean"]),
        "robust_recovery_ratio_region": _safe_recovery(sensitivity_region["mean"], input_region["mean"]),
        "velocity_error_to_reference": _motion_error(corrupt_output, clean_reference, order=1),
        "acceleration_error_to_reference": _motion_error(corrupt_output, clean_reference, order=2),
        "velocity_sensitivity": _motion_error(corrupt_output, clean_output, order=1),
        "acceleration_sensitivity": _motion_error(corrupt_output, clean_output, order=2),
        "amplitude_retention_mean": amplitude["mean"],
        "amplitude_retention_median": amplitude["median"],
        "amplitude_within_0p9_1p1": amplitude["within_0p9_1p1"],
        "amplitude_below_0p8": amplitude["below_0p8"],
        "own_amplitude_retention_mean": own_amplitude["mean"],
        "own_amplitude_retention_median": own_amplitude["median"],
        "own_amplitude_within_0p9_1p1": own_amplitude["within_0p9_1p1"],
        "own_amplitude_below_0p8": own_amplitude["below_0p8"],
    }
    return row


def _actuator_metric_row(
    method: str,
    clean_action: np.ndarray,
    corrupt_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    scenario: dict[str, str],
) -> dict[str, object]:
    scale = np.maximum(action_high - action_low, 1e-9)
    clean = (clean_action - action_low) / scale
    corrupt = (corrupt_action - action_low) / scale
    error = np.abs(corrupt - clean)
    amplitude = _amplitude_metrics(corrupt[:, None, :], clean[:, None, :], active_threshold=0.02)
    return {
        "sequence_id": scenario["sequence_id"],
        "label": scenario["label"],
        "method": method,
        "corruption_type": scenario["corruption_type"],
        "severity": scenario["severity"],
        "duration": int(scenario["duration"]),
        "finger_group": scenario["finger_group"],
        "condition_id": scenario["condition_id"],
        "actuator_mae": float(np.mean(error)),
        "actuator_median_ae": float(np.median(error)),
        "actuator_p95_ae": float(np.percentile(error, 95)),
        "actuator_velocity_error": float(np.mean(np.abs(np.diff(corrupt, axis=0) - np.diff(clean, axis=0)))),
        "actuator_acceleration_error": float(
            np.mean(np.abs(np.diff(corrupt, n=2, axis=0) - np.diff(clean, n=2, axis=0)))
        ),
        "actuator_amplitude_retention_mean": amplitude["mean"],
        "actuator_amplitude_retention_median": amplitude["median"],
        "actuator_amplitude_within_0p9_1p1": amplitude["within_0p9_1p1"],
        "actuator_amplitude_below_0p8": amplitude["below_0p8"],
        "minimum_bound_margin": float(np.min(np.minimum(corrupt, 1.0 - corrupt))),
    }


def _summary_rows(
    rows: list[dict[str, object]], group_keys: tuple[str, ...]
) -> list[dict[str, object]]:
    metric_names = [
        "landmark_error_mean",
        "corrupted_region_error_mean",
        "visible_region_error_mean",
        "direct_recovery_ratio_all",
        "direct_recovery_ratio_region",
        "robust_recovery_ratio_all",
        "robust_recovery_ratio_region",
        "velocity_error_to_reference",
        "acceleration_error_to_reference",
        "own_amplitude_retention_median",
        "own_amplitude_within_0p9_1p1",
        "own_amplitude_below_0p8",
    ]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    output = []
    for group, members in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result = {key: value for key, value in zip(group_keys, group)}
        result["num_instances"] = len(members)
        for metric in metric_names:
            values = np.asarray([float(member[metric]) for member in members], dtype=float)
            values = values[np.isfinite(values)]
            result[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            result[f"{metric}_median"] = float(np.median(values)) if len(values) else math.nan
            result[f"{metric}_std"] = float(np.std(values)) if len(values) else math.nan
            result[f"{metric}_p95"] = float(np.percentile(values, 95)) if len(values) else math.nan
        output.append(result)
    return output


def _paired_tests(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method = {
        method: {str(row["sequence_id"]): row for row in rows if row["method"] == method}
        for method in METHOD_ORDER
    }
    comparisons = []
    for baseline in ("Corrupted Input", "Kalman", "One-Euro", "Corrected"):
        common = sorted(set(by_method["Fixed OA"]) & set(by_method[baseline]))
        for metric in ("corrupted_region_error_mean", "robust_recovery_ratio_region"):
            oa = np.asarray([float(by_method["Fixed OA"][key][metric]) for key in common])
            other = np.asarray([float(by_method[baseline][key][metric]) for key in common])
            valid = np.isfinite(oa) & np.isfinite(other)
            difference = oa[valid] - other[valid]
            try:
                statistic, p_value = wilcoxon(difference) if len(difference) else (math.nan, math.nan)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            comparisons.append(
                {
                    "method": "Fixed OA",
                    "baseline": baseline,
                    "metric": metric,
                    "num_pairs": int(np.sum(valid)),
                    "mean_difference_oa_minus_baseline": float(np.mean(difference)) if len(difference) else math.nan,
                    "median_difference_oa_minus_baseline": float(np.median(difference)) if len(difference) else math.nan,
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    return comparisons


def _actuator_summary_rows(
    rows: list[dict[str, object]], group_keys: tuple[str, ...]
) -> list[dict[str, object]]:
    metric_names = [
        "actuator_mae",
        "actuator_median_ae",
        "actuator_p95_ae",
        "actuator_velocity_error",
        "actuator_acceleration_error",
        "actuator_amplitude_retention_median",
        "actuator_amplitude_within_0p9_1p1",
        "actuator_amplitude_below_0p8",
    ]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, members in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result = {key: value for key, value in zip(group_keys, group)}
        result["num_instances"] = len(members)
        for metric in metric_names:
            values = np.asarray([float(member[metric]) for member in members], dtype=float)
            values = values[np.isfinite(values)]
            result[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            result[f"{metric}_median"] = float(np.median(values)) if len(values) else math.nan
            result[f"{metric}_std"] = float(np.std(values)) if len(values) else math.nan
            result[f"{metric}_p95"] = float(np.percentile(values, 95)) if len(values) else math.nan
        output.append(result)
    return output


def _paired_actuator_tests(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    corruption_groups: list[str | None] = [None, "gaussian", "spike", "dropout"]
    for corruption_type in corruption_groups:
        selected = [
            row for row in rows if corruption_type is None or row["corruption_type"] == corruption_type
        ]
        by_method = {
            method: {str(row["sequence_id"]): row for row in selected if row["method"] == method}
            for method in ("Corrected", "Fixed OA")
        }
        common = sorted(set(by_method["Corrected"]) & set(by_method["Fixed OA"]))
        for metric in ("actuator_mae", "actuator_velocity_error", "actuator_acceleration_error"):
            oa = np.asarray([float(by_method["Fixed OA"][key][metric]) for key in common])
            corrected = np.asarray([float(by_method["Corrected"][key][metric]) for key in common])
            valid = np.isfinite(oa) & np.isfinite(corrected)
            difference = oa[valid] - corrected[valid]
            try:
                statistic, p_value = wilcoxon(difference) if len(difference) else (math.nan, math.nan)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            output.append(
                {
                    "corruption_type": corruption_type or "all",
                    "metric": metric,
                    "num_pairs": int(np.sum(valid)),
                    "fixed_oa_mean": float(np.mean(oa[valid])) if np.any(valid) else math.nan,
                    "corrected_mean": float(np.mean(corrected[valid])) if np.any(valid) else math.nan,
                    "mean_difference_oa_minus_corrected": float(np.mean(difference)) if len(difference) else math.nan,
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    return output


def _bar_plot(
    rows: list[dict[str, object]], metric: str, title: str, ylabel: str, output: Path
) -> None:
    means = []
    errors = []
    for method in METHOD_ORDER:
        values = np.asarray([float(row[metric]) for row in rows if row["method"] == method], dtype=float)
        values = values[np.isfinite(values)]
        means.append(float(np.mean(values)) if len(values) else math.nan)
        errors.append(1.96 * float(np.std(values)) / math.sqrt(len(values)) if len(values) else 0.0)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(METHOD_ORDER))
    ax.bar(x, means, yerr=errors, capsize=4, color=[METHOD_COLORS[m] for m in METHOD_ORDER])
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, METHOD_ORDER, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _actuator_bar_plot(
    rows: list[dict[str, object]], metric: str, title: str, ylabel: str, output: Path
) -> None:
    methods = ("Corrected", "Fixed OA")
    means = []
    errors = []
    for method in methods:
        values = np.asarray([float(row[metric]) for row in rows if row["method"] == method], dtype=float)
        values = values[np.isfinite(values)]
        means.append(float(np.mean(values)) if len(values) else math.nan)
        errors.append(1.96 * float(np.std(values)) / math.sqrt(len(values)) if len(values) else 0.0)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=errors, capsize=4, color=[METHOD_COLORS[m] for m in methods])
    ax.set_xticks(x, methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _condition_plot(
    rows: list[dict[str, object]], corruption_type: str, x_key: str, order: list[object], output: Path
) -> None:
    selected = [row for row in rows if row["corruption_type"] == corruption_type]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(order))
    width = 0.16
    for method_index, method in enumerate(METHOD_ORDER):
        means = []
        for value in order:
            metric_values = np.asarray(
                [
                    float(row["robust_recovery_ratio_region"])
                    for row in selected
                    if row["method"] == method and str(row[x_key]) == str(value)
                ],
                dtype=float,
            )
            metric_values = metric_values[np.isfinite(metric_values)]
            means.append(float(np.mean(metric_values)) if len(metric_values) else math.nan)
        ax.bar(x + (method_index - 2) * width, means, width, label=method, color=METHOD_COLORS[method])
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, [str(value) for value in order])
    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel("Robust recovery ratio")
    ax.set_title(f"Recovery under {corruption_type} corruption")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _trajectory_plot(
    trace: dict[str, object], selection_name: str, output: Path
) -> None:
    mask = np.asarray(trace["mask"], dtype=bool)
    clean = np.asarray(trace["clean"])
    series = {
        "Corrupted Input": np.asarray(trace["corrupted"]),
        "Corrected": np.asarray(trace["corrected_corrupt"]),
        "Fixed OA": np.asarray(trace["oa_corrupt"]),
    }
    baselines = {
        "Corrupted Input": clean,
        "Corrected": np.asarray(trace["corrected_clean"]),
        "Fixed OA": np.asarray(trace["oa_clean"]),
    }
    affected = np.any(mask, axis=0)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(np.zeros(len(clean)), label="Clean reference", color="#111111", linewidth=2)
    for method, values in series.items():
        deviation = np.linalg.norm(values[:, affected] - baselines[method][:, affected], axis=2).mean(axis=1)
        ax.plot(deviation, label=method, color=METHOD_COLORS[method], linewidth=1.8)
    affected_frames = np.any(mask, axis=1)
    indices = np.flatnonzero(affected_frames)
    if len(indices):
        ax.axvspan(indices[0], indices[-1], color="#ef4444", alpha=0.10, label="Corrupted interval")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Corruption-induced deviation")
    ax.set_title(
        f"{selection_name.title()} OA case: {trace['sequence_id']} ({trace['condition_id']})"
    )
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _make_report(
    output_dir: Path,
    rows: list[dict[str, object]],
    actuator_rows: list[dict[str, object]],
    validation: dict[str, object],
    paired: list[dict[str, object]],
) -> None:
    def mean_for(method: str, metric: str, corruption_type: str | None = None, duration: int | None = None) -> float:
        values = []
        for row in rows:
            if row["method"] != method:
                continue
            if corruption_type is not None and row["corruption_type"] != corruption_type:
                continue
            if duration is not None and int(row["duration"]) != duration:
                continue
            value = float(row[metric])
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else math.nan

    oa_direct = mean_for("Fixed OA", "direct_recovery_ratio_region")
    oa_robust = mean_for("Fixed OA", "robust_recovery_ratio_region")
    corrected_gaussian = mean_for("Corrected", "corrupted_region_error_mean", "gaussian")
    oa_gaussian = mean_for("Fixed OA", "corrupted_region_error_mean", "gaussian")
    corrected_spike = mean_for("Corrected", "corrupted_region_error_mean", "spike")
    oa_spike = mean_for("Fixed OA", "corrupted_region_error_mean", "spike")
    dropout_results = {
        duration: (
            mean_for("Fixed OA", "corrupted_region_error_mean", "dropout", duration),
            mean_for("Corrected", "corrupted_region_error_mean", "dropout", duration),
        )
        for duration in (3, 5)
    }
    amplitude = mean_for("Fixed OA", "own_amplitude_retention_median")

    def actuator_mean(method: str, metric: str, corruption_type: str | None = None) -> float:
        values = []
        for row in actuator_rows:
            if row["method"] != method:
                continue
            if corruption_type is not None and row["corruption_type"] != corruption_type:
                continue
            value = float(row[metric])
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else math.nan

    oa_actuator_mae = actuator_mean("Fixed OA", "actuator_mae")
    corrected_actuator_mae = actuator_mean("Corrected", "actuator_mae")
    oa_gaussian_actuator = actuator_mean("Fixed OA", "actuator_mae", "gaussian")
    corrected_gaussian_actuator = actuator_mean("Corrected", "actuator_mae", "gaussian")

    def yes_no(condition: bool) -> str:
        return "Yes" if condition else "No"

    lines = [
        "# Controlled Clean-Reference Recovery Benchmark",
        "",
        "## Scope",
        "",
        "This benchmark treats the recorded six-class MediaPipe trajectories as a controlled clean reference, not as physical ground truth. The current fixed-weight Optimized Action implementation and all filter parameters were left unchanged.",
        "",
        "Each sequence receives one balanced corruption condition. Random seeds rotate through a deterministic pool of ten values. Sequence/corruption instances, rather than frames, are the statistical units.",
        "",
        "Two recovery definitions are reported:",
        "",
        "- `direct_recovery_ratio = 1 - error(method(corrupt), clean_reference) / error(corrupt, clean_reference)` includes each method's clean-domain/model bias.",
        "- `robust_recovery_ratio = 1 - error(method(corrupt), method(clean)) / error(corrupt, clean_reference)` isolates sensitivity to the induced corruption.",
        "",
        "Corrected and Fixed OA actuator errors use each method's clean output as a controlled reference; they are not errors against measured joint-angle ground truth.",
        f"Across all conditions, normalized actuator sensitivity is `{oa_actuator_mae:.4f}` for Fixed OA and `{corrected_actuator_mae:.4f}` for Corrected. For Gaussian noise it is `{oa_gaussian_actuator:.4f}` versus `{corrected_gaussian_actuator:.4f}`.",
        "",
        "## Decision Questions",
        "",
        f"### Q1. Does Fixed OA reduce induced trajectory error? {yes_no(oa_direct > 0)} by the direct corrupted-region criterion.",
        f"Mean direct recovery ratio: `{oa_direct:.3f}`. Mean bias-adjusted robust recovery ratio: `{oa_robust:.3f}`.",
        "",
        f"### Q2. Does Fixed OA outperform Corrected for Gaussian noise? {yes_no(oa_gaussian < corrected_gaussian)}.",
        f"Corrupted-region error: Fixed OA `{oa_gaussian:.4f}`, Corrected `{corrected_gaussian:.4f}`.",
        "",
        f"### Q3. Does Fixed OA outperform Corrected for isolated spikes? {yes_no(oa_spike < corrected_spike)}.",
        f"Corrupted-region error: Fixed OA `{oa_spike:.4f}`, Corrected `{corrected_spike:.4f}`.",
        "",
        f"### Q4. Does Fixed OA outperform Corrected for 3-frame and 5-frame dropout? 3 frames: {yes_no(dropout_results[3][0] < dropout_results[3][1])}; 5 frames: {yes_no(dropout_results[5][0] < dropout_results[5][1])}.",
        f"Errors (OA vs Corrected): 3 frames `{dropout_results[3][0]:.4f}` vs `{dropout_results[3][1]:.4f}`; 5 frames `{dropout_results[5][0]:.4f}` vs `{dropout_results[5][1]:.4f}`.",
        "",
        f"### Q5. Does Fixed OA preserve motion amplitude? Its mean sequence-level median own-baseline retention is `{amplitude:.3f}` (ideal `1.0`).",
        "",
        "### Q6. Is Fixed OA recovering the trajectory or mainly smoothing it?",
        "",
    ]
    if oa_direct > 0 and 0.8 <= amplitude <= 1.2:
        lines.append("The current evidence supports partial clean-reference recovery rather than smoothing alone, because induced error decreases while motion amplitude remains broadly retained.")
    elif oa_robust > 0 and oa_direct <= 0:
        lines.append("Fixed OA is less sensitive to corruption than the unprocessed trajectory, but its MuJoCo reconstruction bias prevents a direct clean-reference recovery claim. It should be described as model-constrained robust refinement, not verified landmark recovery.")
    else:
        lines.append("The current evidence does not support a trajectory-recovery claim. Any reduction in derivative magnitude should be interpreted as smoothing unless the objective or observation model is revised.")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Non-finite values: `{validation['nonfinite_values']}`",
            f"- Actuator bound violations: `{validation['bound_violations']}`",
            f"- Mean OA optimizer success rate: `{validation['oa_success_rate_mean']:.4f}`",
            f"- Sequence-start reset maximum action difference: `{validation['history_reset_max_abs_difference']:.3e}`",
            "",
            "## Output Files",
            "",
            "- `per_sequence_landmark_metrics.csv`: primary paired landmark metrics.",
            "- `per_sequence_actuator_metrics.csv`: within-method actuator corruption sensitivity.",
            "- `actuator_overall_summary.csv` and `actuator_summary_by_corruption.csv`: actuator-space aggregate results.",
            "- `overall_summary.csv`, `summary_by_corruption.csv`, and `summary_by_condition.csv`: aggregate tables.",
            "- `paired_wilcoxon_tests.csv`: paired sequence-level tests.",
            "- `figures/`: recovery, motion-fidelity, and systematically selected trajectory plots.",
            "",
            "## Interpretation Constraint",
            "",
            "This experiment evaluates recovery toward an observed clean MediaPipe trajectory. It does not establish anatomical ground-truth accuracy, long-duration occlusion completion, or cross-subject generalization.",
        ]
    )
    (output_dir / "RECOVERY_BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    clean_path: Path,
    corrupted_path: Path,
    manifest_path: Path,
    scenario_path: Path,
    output_dir: Path,
    *,
    version: str | None,
    max_sequences: int | None,
) -> None:
    clean_sequences = _load_sequences(clean_path)
    corrupted_sequences = _load_sequences(corrupted_path)
    scenarios = _load_scenarios(scenario_path)
    common = sorted(set(clean_sequences) & set(corrupted_sequences) & set(scenarios))
    if max_sequences is not None:
        common = common[:max_sequences]
    if not common:
        raise ValueError("No aligned sequences were found.")

    for sequence_id in common:
        if clean_sequences[sequence_id]["frame_ids"] != corrupted_sequences[sequence_id]["frame_ids"]:
            raise ValueError(f"Frame alignment mismatch for {sequence_id}")
        if np.asarray(clean_sequences[sequence_id]["points"]).shape != np.asarray(
            corrupted_sequences[sequence_id]["points"]
        ).shape:
            raise ValueError(f"Shape mismatch for {sequence_id}")

    masks = _load_masks(manifest_path, {key: len(clean_sequences[key]["points"]) for key in common})
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    landmark_rows: list[dict[str, object]] = []
    actuator_rows: list[dict[str, object]] = []
    traces: dict[str, dict[str, object]] = {}
    oa_diagnostics = []
    bound_violations = 0
    nonfinite_values = 0
    reset_differences = []
    reused_clean_features = 0

    with MujocoHandPoseOptimizer(version=version) as optimizer:
        low = optimizer.env.action_low.astype(np.float64)
        high = optimizer.env.action_high.astype(np.float64)
        for sequence_index, sequence_id in enumerate(common, start=1):
            clean_data = clean_sequences[sequence_id]
            corrupt_data = corrupted_sequences[sequence_id]
            clean = np.asarray(clean_data["points"], dtype=np.float64)
            corrupted = np.asarray(corrupt_data["points"], dtype=np.float64)
            timestamps = np.asarray(clean_data["timestamps"], dtype=np.float64)
            scenario = scenarios[sequence_id]
            mask = masks[sequence_id]
            if not np.any(mask):
                raise ValueError(f"Empty corruption mask for {sequence_id}")

            kalman_clean = _filter_sequence(clean, timestamps, "Kalman")
            kalman_corrupt = _filter_sequence(corrupted, timestamps, "Kalman")
            euro_clean = _filter_sequence(clean, timestamps, "One-Euro")
            euro_corrupt = _filter_sequence(corrupted, timestamps, "One-Euro")
            if "corrected_action" in clean_data:
                corrected_clean_action = np.asarray(clean_data["corrected_action"], dtype=np.float64)
                corrected_clean = np.asarray(
                    [optimizer.full_landmarks_from_action(action) for action in corrected_clean_action],
                    dtype=np.float64,
                )
            else:
                corrected_clean_action, corrected_clean = _run_corrected(optimizer, clean)
            corrected_corrupt_action, corrected_corrupt = _run_corrected(optimizer, corrupted)
            if "optimized_action" in clean_data and "optimized_full" in clean_data:
                oa_clean_action = np.asarray(clean_data["optimized_action"], dtype=np.float64)
                oa_clean = np.asarray(clean_data["optimized_full"], dtype=np.float64)
                reused_clean_features += 1
            else:
                oa_clean_action, oa_clean, clean_diag = _run_oa(optimizer, clean)
                oa_diagnostics.append(clean_diag)
            oa_corrupt_action, oa_corrupt, corrupt_diag = _run_oa(optimizer, corrupted)
            oa_diagnostics.append(corrupt_diag)

            if sequence_index <= 3:
                independent = optimizer.optimize(corrupted[0])
                reset_differences.append(float(np.max(np.abs(independent.action - oa_corrupt_action[0]))))

            for action in (
                corrected_clean_action,
                corrected_corrupt_action,
                oa_clean_action,
                oa_corrupt_action,
            ):
                bound_violations += int(np.sum((action < low - 1e-7) | (action > high + 1e-7)))
                nonfinite_values += int(np.sum(~np.isfinite(action)))
            for points in (
                clean,
                corrupted,
                kalman_clean,
                kalman_corrupt,
                euro_clean,
                euro_corrupt,
                corrected_clean,
                corrected_corrupt,
                oa_clean,
                oa_corrupt,
            ):
                nonfinite_values += int(np.sum(~np.isfinite(points)))

            method_outputs = {
                "Corrupted Input": (clean, corrupted),
                "Kalman": (kalman_clean, kalman_corrupt),
                "One-Euro": (euro_clean, euro_corrupt),
                "Corrected": (corrected_clean, corrected_corrupt),
                "Fixed OA": (oa_clean, oa_corrupt),
            }
            for method, (clean_output, corrupt_output) in method_outputs.items():
                landmark_rows.append(
                    _landmark_metric_row(
                        method,
                        clean,
                        corrupted,
                        clean_output,
                        corrupt_output,
                        mask,
                        scenario,
                    )
                )
            actuator_rows.extend(
                [
                    _actuator_metric_row(
                        "Corrected", corrected_clean_action, corrected_corrupt_action, low, high, scenario
                    ),
                    _actuator_metric_row("Fixed OA", oa_clean_action, oa_corrupt_action, low, high, scenario),
                ]
            )
            traces[sequence_id] = {
                "sequence_id": sequence_id,
                "condition_id": scenario["condition_id"],
                "mask": mask,
                "clean": clean,
                "corrupted": corrupted,
                "corrected_clean": corrected_clean,
                "corrected_corrupt": corrected_corrupt,
                "oa_clean": oa_clean,
                "oa_corrupt": oa_corrupt,
            }
            print(
                f"processed={sequence_index}/{len(common)} sequence_id={sequence_id} "
                f"condition={scenario['condition_id']}",
                flush=True,
            )

    validation = {
        "num_sequences": len(common),
        "num_frames": int(sum(len(clean_sequences[key]["points"]) for key in common)),
        "nonfinite_values": nonfinite_values,
        "bound_violations": bound_violations,
        "oa_success_rate_mean": float(np.mean([item["success_rate"] for item in oa_diagnostics])),
        "oa_iterations_mean": float(np.mean([item["iterations_mean"] for item in oa_diagnostics])),
        "oa_solve_time_ms_mean": float(np.mean([item["solve_time_ms_mean"] for item in oa_diagnostics])),
        "oa_solve_time_ms_p95_mean": float(np.mean([item["solve_time_ms_p95"] for item in oa_diagnostics])),
        "history_reset_max_abs_difference": float(max(reset_differences, default=math.nan)),
        "reused_clean_feature_sequences": reused_clean_features,
    }
    if nonfinite_values or bound_violations:
        raise RuntimeError(f"Validation failed: {validation}")

    overall = _summary_rows(landmark_rows, ("method",))
    by_corruption = _summary_rows(landmark_rows, ("method", "corruption_type"))
    by_condition = _summary_rows(
        landmark_rows, ("method", "corruption_type", "severity", "duration", "finger_group")
    )
    paired = _paired_tests(landmark_rows)
    actuator_paired = _paired_actuator_tests(actuator_rows)
    actuator_overall = _actuator_summary_rows(actuator_rows, ("method",))
    actuator_by_corruption = _actuator_summary_rows(actuator_rows, ("method", "corruption_type"))
    actuator_by_condition = _actuator_summary_rows(
        actuator_rows, ("method", "corruption_type", "severity", "duration", "finger_group")
    )
    _write_dict_rows(output_dir / "per_sequence_landmark_metrics.csv", landmark_rows)
    _write_dict_rows(output_dir / "per_sequence_actuator_metrics.csv", actuator_rows)
    _write_dict_rows(output_dir / "overall_summary.csv", overall)
    _write_dict_rows(output_dir / "summary_by_corruption.csv", by_corruption)
    _write_dict_rows(output_dir / "summary_by_condition.csv", by_condition)
    _write_dict_rows(output_dir / "paired_wilcoxon_tests.csv", paired)
    _write_dict_rows(output_dir / "actuator_paired_wilcoxon_tests.csv", actuator_paired)
    _write_dict_rows(output_dir / "actuator_overall_summary.csv", actuator_overall)
    _write_dict_rows(output_dir / "actuator_summary_by_corruption.csv", actuator_by_corruption)
    _write_dict_rows(output_dir / "actuator_summary_by_condition.csv", actuator_by_condition)
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    _bar_plot(
        landmark_rows,
        "corrupted_region_error_mean",
        "Clean-reference error in corrupted regions",
        "Mean Euclidean landmark error",
        figure_dir / "corrupted_region_error.png",
    )
    _bar_plot(
        landmark_rows,
        "robust_recovery_ratio_region",
        "Bias-adjusted recovery from induced corruption",
        "Robust recovery ratio",
        figure_dir / "recovery_ratio.png",
    )
    _bar_plot(
        landmark_rows,
        "own_amplitude_retention_median",
        "Motion amplitude retention relative to each method's clean output",
        "Median amplitude retention (target 1.0)",
        figure_dir / "amplitude_retention.png",
    )
    _bar_plot(
        landmark_rows,
        "velocity_sensitivity",
        "Velocity error induced by corruption",
        "Velocity sensitivity",
        figure_dir / "velocity_error.png",
    )
    _bar_plot(
        landmark_rows,
        "acceleration_sensitivity",
        "Acceleration error induced by corruption",
        "Acceleration sensitivity",
        figure_dir / "acceleration_error.png",
    )
    _condition_plot(landmark_rows, "gaussian", "severity", ["mild", "medium", "severe"], figure_dir / "recovery_vs_gaussian_severity.png")
    _condition_plot(landmark_rows, "spike", "duration", [1, 2, 3], figure_dir / "recovery_vs_spike_duration.png")
    _condition_plot(landmark_rows, "dropout", "duration", [3, 5], figure_dir / "recovery_vs_dropout_duration.png")
    _actuator_bar_plot(
        actuator_rows,
        "actuator_mae",
        "Actuator sensitivity to induced corruption",
        "Normalized actuator MAE",
        figure_dir / "actuator_sensitivity.png",
    )
    _actuator_bar_plot(
        actuator_rows,
        "actuator_velocity_error",
        "Actuator velocity error induced by corruption",
        "Normalized velocity error",
        figure_dir / "actuator_velocity_error.png",
    )
    _actuator_bar_plot(
        actuator_rows,
        "actuator_acceleration_error",
        "Actuator acceleration error induced by corruption",
        "Normalized acceleration error",
        figure_dir / "actuator_acceleration_error.png",
    )
    _actuator_bar_plot(
        actuator_rows,
        "actuator_amplitude_retention_median",
        "Actuator motion amplitude retention",
        "Median retention (target 1.0)",
        figure_dir / "actuator_amplitude_retention.png",
    )

    oa_rows = [row for row in landmark_rows if row["method"] == "Fixed OA"]
    oa_rows = [row for row in oa_rows if np.isfinite(float(row["robust_recovery_ratio_region"]))]
    oa_rows.sort(key=lambda row: float(row["robust_recovery_ratio_region"]))
    difficult = oa_rows[0]
    median = min(
        oa_rows,
        key=lambda row: abs(
            float(row["robust_recovery_ratio_region"])
            - float(np.median([float(item["robust_recovery_ratio_region"]) for item in oa_rows]))
        ),
    )
    _trajectory_plot(traces[str(median["sequence_id"])], "median", figure_dir / "trajectory_median_case.png")
    _trajectory_plot(traces[str(difficult["sequence_id"])], "difficult", figure_dir / "trajectory_difficult_case.png")
    (output_dir / "trajectory_selection.json").write_text(
        json.dumps(
            {
                "rule": "Median is closest to the median Fixed OA robust corrupted-region recovery ratio; difficult is the minimum ratio.",
                "median_sequence_id": median["sequence_id"],
                "median_ratio": median["robust_recovery_ratio_region"],
                "difficult_sequence_id": difficult["sequence_id"],
                "difficult_ratio": difficult["robust_recovery_ratio_region"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _make_report(output_dir, landmark_rows, actuator_rows, validation, paired)
    print(f"output_dir={output_dir.resolve()}")
    print(json.dumps(validation, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled clean-reference recovery benchmark.")
    parser.add_argument("--clean", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="v2")
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()
    run_benchmark(
        Path(args.clean),
        Path(args.corrupted),
        Path(args.manifest),
        Path(args.scenarios),
        Path(args.output_dir),
        version=args.version,
        max_sequences=args.max_sequences,
    )


if __name__ == "__main__":
    main()
