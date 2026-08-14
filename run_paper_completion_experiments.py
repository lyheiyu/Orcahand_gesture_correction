from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_svm as tsvm
from orca_sim.mujoco_optimizer import MujocoHandPoseOptimizer, OptimizationWeights


CLASSIFIERS = ("svm", "knn", "rf", "mlp")
ABLATION_CLASSIFIERS = ("svm", "rf")
PCA_FEATURES = ("raw", "raw_pca17", "corrected", "optimized_action", "optimized_full")
ABLATION_FEATURES = ("corrected", "full", "no_palm", "no_acceleration", "no_temporal", "l2")
DISPLAY_CLASSIFIERS = {"svm": "SVM", "knn": "KNN", "rf": "RandomForest", "mlp": "MLP"}
DISPLAY_PCA = {
    "raw": "Raw",
    "raw_pca17": "PCA-17",
    "corrected": "Corrected",
    "optimized_action": "Optimized Action",
    "optimized_full": "Optimized Full",
}
DISPLAY_ABLATION = {
    "corrected": "Corrected only",
    "full": "Full",
    "no_palm": "Without palm normal",
    "no_acceleration": "Without acceleration",
    "no_temporal": "Without temporal terms",
    "l2": "L2 instead of Huber",
}


def _model_args(classifier: str) -> SimpleNamespace:
    return SimpleNamespace(
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


def _load_manifest(path: Path) -> dict[int, dict[str, list[str]]]:
    grouped: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"train": [], "test": []})
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            grouped[int(row["repeat"])][row["split"]].append(row["sequence_id"])
    return dict(grouped)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _summarize(rows: list[dict[str, object]], group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in group_keys)].append(row)
    summary: list[dict[str, object]] = []
    for group, group_rows in sorted(grouped.items()):
        result: dict[str, object] = dict(zip(group_keys, group, strict=True))
        for metric in ("accuracy", "macro_f1", "kappa"):
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values))
        result["repeats"] = len(group_rows)
        summary.append(result)
    return summary


def _evaluate_sequences(
    sequence_ids: list[str],
    labels: list[str],
    sequences: list[np.ndarray],
    manifest: dict[int, dict[str, list[str]]],
    classifier: str,
    feature_set: str,
    *,
    pca_components: int = 0,
) -> tuple[list[dict[str, object]], np.ndarray]:
    sequence_lookup = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}
    class_labels = sorted(set(labels))
    aggregate_confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    rows: list[dict[str, object]] = []
    for repeat in sorted(manifest):
        train_indices = [sequence_lookup[value] for value in manifest[repeat]["train"]]
        test_indices = [sequence_lookup[value] for value in manifest[repeat]["test"]]
        train_sequences = [sequences[index] for index in train_indices]
        test_sequences = [sequences[index] for index in test_indices]
        y_train = [labels[index] for index in train_indices]
        y_test = [labels[index] for index in test_indices]
        if pca_components:
            train_sequences, test_sequences = tsvm._project_sequences_with_pca(
                train_sequences, test_sequences, pca_components, 42 + repeat
            )
        x_train = np.stack([tsvm._aggregate_sequence_array(value) for value in train_sequences])
        x_test = np.stack([tsvm._aggregate_sequence_array(value) for value in test_sequences])
        model = tsvm._build_model(_model_args(classifier), 42 + repeat)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        aggregate_confusion += confusion_matrix(y_test, prediction, labels=class_labels)
        rows.append(
            {
                "repeat": repeat,
                "classifier": classifier,
                "feature_set": feature_set,
                "accuracy": float(accuracy_score(y_test, prediction)),
                "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
                "kappa": float(cohen_kappa_score(y_test, prediction)),
            }
        )
    return rows, aggregate_confusion


def evaluate_pca_suite(dataset: Path, manifest_path: Path, output_dir: Path) -> None:
    row_meta, feature_names, features = tsvm._load_dataset(dataset)
    manifest = _load_manifest(manifest_path)
    per_repeat: list[dict[str, object]] = []
    confusions: dict[tuple[str, str], np.ndarray] = {}
    for feature_set in PCA_FEATURES:
        source = "raw" if feature_set == "raw_pca17" else feature_set
        _, selected = tsvm._select_features(feature_names, features, source)
        sequence_ids, labels, sequences = tsvm._group_sequences(row_meta, selected)
        for classifier in CLASSIFIERS:
            rows, confusion = _evaluate_sequences(
                sequence_ids,
                labels,
                sequences,
                manifest,
                classifier,
                feature_set,
                pca_components=17 if feature_set == "raw_pca17" else 0,
            )
            per_repeat.extend(rows)
            confusions[(classifier, feature_set)] = confusion
    _write_rows(output_dir / "pca_all_classifiers_per_repeat_6class.csv", per_repeat)
    _write_rows(
        output_dir / "pca_all_classifiers_summary_6class.csv",
        _summarize(per_repeat, ("classifier", "feature_set")),
    )
    _plot_pca_suite(per_repeat, output_dir)


def _variant_weights(name: str) -> OptimizationWeights:
    weights = OptimizationWeights()
    if name == "no_palm":
        weights.palm = 0.0
    elif name == "no_acceleration":
        weights.acceleration = 0.0
    elif name == "no_temporal":
        weights.temporal = 0.0
        weights.acceleration = 0.0
    elif name == "l2":
        weights.huber_delta = 0.0
    else:
        raise ValueError(f"Unknown ablation variant: {name}")
    return weights


def _load_raw_grouped(dataset: Path) -> tuple[list[str], dict[str, list[dict[str, object]]]]:
    with dataset.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        raw_names = [name for name in (reader.fieldnames or []) if name.startswith("raw_")]
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in reader:
            grouped[row["sequence_id"]].append(
                {
                    "label": row["label"],
                    "sequence_id": row["sequence_id"],
                    "frame_id": int(float(row["frame_id"])),
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "points": np.asarray([float(row[name]) for name in raw_names], dtype=np.float64).reshape(21, 3),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["frame_id"]), float(row["timestamp_sec"])))
    return raw_names, dict(grouped)


def _generate_variant_worker(variant: str, dataset_text: str, output_text: str) -> dict[str, object]:
    dataset = Path(dataset_text)
    output = Path(output_text)
    _, grouped = _load_raw_grouped(dataset)
    weights = _variant_weights(variant)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "sequence_id", "frame_id", "timestamp_sec"]
    fieldnames += [f"{variant}_{index}" for index in range(17)]
    fieldnames += ["success", "iterations", "solve_time_ms", "total_loss"]
    solve_times: list[float] = []
    iterations: list[int] = []
    successes: list[bool] = []
    with output.open("w", newline="", encoding="utf-8") as fh, MujocoHandPoseOptimizer(version="v2") as optimizer:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sequence_id in sorted(grouped):
            previous = None
            previous_previous = None
            for row in grouped[sequence_id]:
                result = optimizer.optimize(
                    np.asarray(row["points"]),
                    prev_action=previous,
                    prev_prev_action=previous_previous,
                    weights=weights,
                )
                output_row: dict[str, object] = {
                    "label": row["label"],
                    "sequence_id": sequence_id,
                    "frame_id": row["frame_id"],
                    "timestamp_sec": row["timestamp_sec"],
                    "success": int(result.success),
                    "iterations": result.iterations,
                    "solve_time_ms": result.solve_time_ms,
                    "total_loss": result.loss,
                }
                output_row.update({f"{variant}_{index}": float(value) for index, value in enumerate(result.action)})
                writer.writerow(output_row)
                solve_times.append(result.solve_time_ms)
                iterations.append(result.iterations)
                successes.append(result.success)
                old_previous = previous
                previous = result.action.astype(np.float64)
                previous_previous = old_previous
    solve_array = np.asarray(solve_times, dtype=np.float64)
    return {
        "variant": variant,
        "frames": len(solve_times),
        "success_rate": float(np.mean(successes)),
        "solve_time_mean_ms": float(np.mean(solve_array)),
        "solve_time_median_ms": float(np.median(solve_array)),
        "solve_time_p95_ms": float(np.percentile(solve_array, 95)),
        "iterations_mean": float(np.mean(iterations)),
        "weights": json.dumps(asdict(weights), sort_keys=True),
    }


def generate_ablation_features(dataset: Path, diagnostics_dir: Path, workers: int, force: bool) -> Path:
    variants = ("no_palm", "no_acceleration", "no_temporal", "l2")
    variant_paths = {variant: diagnostics_dir / f"ablation_{variant}_frames.csv" for variant in variants}
    summaries: list[dict[str, object]] = []
    pending = [variant for variant in variants if force or not variant_paths[variant].exists()]
    if pending:
        with ProcessPoolExecutor(max_workers=min(max(1, workers), len(pending))) as executor:
            futures = {
                executor.submit(
                    _generate_variant_worker,
                    variant,
                    str(dataset),
                    str(variant_paths[variant]),
                ): variant
                for variant in pending
            }
            for future in as_completed(futures):
                result = future.result()
                summaries.append(result)
                print(f"generated_ablation={result['variant']} frames={result['frames']}")
    for variant in variants:
        if variant not in pending:
            with variant_paths[variant].open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            times = np.asarray([float(row["solve_time_ms"]) for row in rows], dtype=np.float64)
            iterations = np.asarray([float(row["iterations"]) for row in rows], dtype=np.float64)
            summaries.append(
                {
                    "variant": variant,
                    "frames": len(rows),
                    "success_rate": float(np.mean([int(row["success"]) for row in rows])),
                    "solve_time_mean_ms": float(np.mean(times)),
                    "solve_time_median_ms": float(np.median(times)),
                    "solve_time_p95_ms": float(np.percentile(times, 95)),
                    "iterations_mean": float(np.mean(iterations)),
                    "weights": json.dumps(asdict(_variant_weights(variant)), sort_keys=True),
                }
            )
    _write_rows(diagnostics_dir / "ablation_generation_runtime.csv", summaries)

    original_meta, original_names, original_features = tsvm._load_dataset(dataset)
    corrected_names, corrected = tsvm._select_features(original_names, original_features, "corrected")
    full_names, full = tsvm._select_features(original_names, original_features, "optimized_action")
    compact_path = diagnostics_dir / "ablation_features_6class.csv"
    variant_maps: dict[str, dict[tuple[str, int], list[float]]] = {}
    for variant, path in variant_paths.items():
        mapping: dict[tuple[str, int], list[float]] = {}
        with path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row["sequence_id"], int(float(row["frame_id"])))
                mapping[key] = [float(row[f"{variant}_{index}"]) for index in range(17)]
        variant_maps[variant] = mapping
    fields = ["label", "sequence_id", "frame_id", "timestamp_sec"]
    fields += [f"corrected_{index}" for index in range(17)]
    fields += [f"full_{index}" for index in range(17)]
    for variant in variants:
        fields += [f"{variant}_{index}" for index in range(17)]
    with compact_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row_index, meta in enumerate(original_meta):
            frame_id = int(float(meta["frame_id"]))
            key = (meta["sequence_id"], frame_id)
            out: dict[str, object] = dict(meta)
            out.update({f"corrected_{index}": float(corrected[row_index, index]) for index in range(17)})
            out.update({f"full_{index}": float(full[row_index, index]) for index in range(17)})
            for variant in variants:
                out.update({f"{variant}_{index}": variant_maps[variant][key][index] for index in range(17)})
            writer.writerow(out)
    return compact_path


def evaluate_ablation_suite(dataset: Path, manifest_path: Path, output_dir: Path) -> None:
    row_meta, feature_names, features = tsvm._load_dataset(dataset)
    manifest = _load_manifest(manifest_path)
    per_repeat: list[dict[str, object]] = []
    for feature_set in ABLATION_FEATURES:
        _, selected = tsvm._select_features(feature_names, features, feature_set)
        sequence_ids, labels, sequences = tsvm._group_sequences(row_meta, selected)
        for classifier in ABLATION_CLASSIFIERS:
            rows, _ = _evaluate_sequences(
                sequence_ids, labels, sequences, manifest, classifier, feature_set
            )
            per_repeat.extend(rows)
    _write_rows(output_dir / "loss_ablation_per_repeat_6class.csv", per_repeat)
    _write_rows(
        output_dir / "loss_ablation_summary_6class.csv",
        _summarize(per_repeat, ("classifier", "feature_set")),
    )
    _write_rows(
        output_dir / "loss_ablation_paired_stats_6class.csv",
        _paired_ablation_stats(per_repeat),
    )
    _plot_ablation(per_repeat, output_dir)
    evaluate_ablation_stability(dataset, per_repeat, output_dir)


def _paired_ablation_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    lookup = {
        (str(row["classifier"]), str(row["feature_set"]), int(row["repeat"])): float(row["accuracy"])
        for row in rows
    }
    output: list[dict[str, object]] = []
    for classifier in ABLATION_CLASSIFIERS:
        full = np.asarray([lookup[(classifier, "full", repeat)] for repeat in range(20)])
        for feature_set in ABLATION_FEATURES:
            if feature_set == "full":
                continue
            alternative = np.asarray([lookup[(classifier, feature_set, repeat)] for repeat in range(20)])
            differences = full - alternative
            p_value = (
                float(wilcoxon(differences).pvalue)
                if wilcoxon is not None and np.any(np.abs(differences) > 1e-12)
                else 1.0
            )
            output.append(
                {
                    "classifier": classifier,
                    "comparison": f"full_minus_{feature_set}",
                    "mean_accuracy_difference": float(np.mean(differences)),
                    "positive_repeats": int(np.sum(differences > 0)),
                    "equal_repeats": int(np.sum(differences == 0)),
                    "negative_repeats": int(np.sum(differences < 0)),
                    "wilcoxon_p": p_value,
                }
            )
    return output


def evaluate_ablation_stability(
    dataset: Path,
    classification_rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    import evaluate_jitter
    import matplotlib.pyplot as plt

    row_meta, feature_names, features = evaluate_jitter._load_dataset(dataset)
    stability_rows: list[dict[str, object]] = []
    for feature_set in ABLATION_FEATURES:
        summary = evaluate_jitter.evaluate_feature_set(row_meta, feature_names, features, feature_set)
        stability_rows.append(
            {
                "feature_set": feature_set,
                "num_sequences": int(summary["num_sequences"]),
                "num_frames": int(summary["num_frames"]),
                "velocity_mean": float(summary["velocity_mean_mean"]),
                "velocity_std": float(summary["velocity_mean_std"]),
                "acceleration_mean": float(summary["acceleration_mean_mean"]),
                "acceleration_std": float(summary["acceleration_mean_std"]),
            }
        )
    _write_rows(output_dir / "loss_ablation_stability_6class.csv", stability_rows)

    accuracy_lookup = {
        feature_set: float(
            np.mean(
                [
                    float(row["accuracy"])
                    for row in classification_rows
                    if row["classifier"] == "svm" and row["feature_set"] == feature_set
                ]
            )
        )
        for feature_set in ABLATION_FEATURES
    }
    x = np.arange(len(ABLATION_FEATURES))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    axes[0].bar(
        x - 0.18,
        [float(row["velocity_mean"]) for row in stability_rows],
        0.36,
        label="Velocity",
        color="#4C78A8",
    )
    axes[0].bar(
        x + 0.18,
        [float(row["acceleration_mean"]) for row in stability_rows],
        0.36,
        label="Acceleration",
        color="#F58518",
    )
    axes[0].set_xticks(x, [DISPLAY_ABLATION[value] for value in ABLATION_FEATURES], rotation=25, ha="right")
    axes[0].set_ylabel("Mean actuator-space difference")
    axes[0].set_title("Temporal stability", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for row in stability_rows:
        feature_set = str(row["feature_set"])
        axes[1].scatter(
            float(row["acceleration_mean"]),
            accuracy_lookup[feature_set],
            s=72,
            label=DISPLAY_ABLATION[feature_set],
        )
        annotation_offsets = {
            "corrected": (-82, 4),
            "full": (6, 4),
            "no_palm": (8, 14),
            "no_acceleration": (6, 4),
            "no_temporal": (6, 4),
            "l2": (8, -18),
        }
        axes[1].annotate(
            DISPLAY_ABLATION[feature_set],
            (float(row["acceleration_mean"]), accuracy_lookup[feature_set]),
            xytext=annotation_offsets[feature_set],
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6}
            if feature_set in {"no_palm", "l2"}
            else None,
        )
    axes[1].set_xlabel("Actuator acceleration mean (lower is smoother)")
    axes[1].set_ylabel("SVM accuracy")
    axes[1].set_title("Stability-recognition trade-off", fontweight="bold")
    axes[1].grid(alpha=0.25)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.25, top=0.80, wspace=0.24)
    fig.suptitle("Loss Ablation: Stability and Downstream Recognition", y=0.97, fontweight="bold")
    fig.savefig(output_dir / "loss_ablation_stability_tradeoff_6class.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def benchmark_runtime(dataset: Path, diagnostics_dir: Path, max_frames: int) -> None:
    _, grouped = _load_raw_grouped(dataset)
    rows: list[dict[str, object]] = []
    with MujocoHandPoseOptimizer(version="v2") as optimizer:
        first_sequence = next(iter(grouped.values()))
        first_points = np.asarray(first_sequence[0]["points"])
        optimizer.optimize(first_points)
        for sequence_id in sorted(grouped):
            previous = None
            previous_previous = None
            for row in grouped[sequence_id]:
                result = optimizer.optimize(
                    np.asarray(row["points"]),
                    prev_action=previous,
                    prev_prev_action=previous_previous,
                )
                rows.append(
                    {
                        "sequence_id": sequence_id,
                        "frame_id": row["frame_id"],
                        "solve_time_ms": result.solve_time_ms,
                        "iterations": result.iterations,
                        "success": int(result.success),
                        "finite": int(np.isfinite(result.action).all()),
                    }
                )
                old_previous = previous
                previous = result.action.astype(np.float64)
                previous_previous = old_previous
                if len(rows) >= max_frames:
                    break
            if len(rows) >= max_frames:
                break
    _write_rows(diagnostics_dir / "runtime_frames_6class.csv", rows)
    times = np.asarray([float(row["solve_time_ms"]) for row in rows], dtype=np.float64)
    iterations = np.asarray([int(row["iterations"]) for row in rows], dtype=np.float64)
    summary = {
        "frames": len(rows),
        "solve_time_mean_ms": float(np.mean(times)),
        "solve_time_median_ms": float(np.median(times)),
        "solve_time_p95_ms": float(np.percentile(times, 95)),
        "iterations_mean": float(np.mean(iterations)),
        "iterations_median": float(np.median(iterations)),
        "success_rate": float(np.mean([int(row["success"]) for row in rows])),
        "finite_rate": float(np.mean([int(row["finite"]) for row in rows])),
    }
    (diagnostics_dir / "runtime_summary_6class.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def _plot_pca_suite(rows: list[dict[str, object]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1"), strict=True):
        matrix = np.zeros((len(PCA_FEATURES), len(CLASSIFIERS)), dtype=np.float64)
        for i, feature_set in enumerate(PCA_FEATURES):
            for j, classifier in enumerate(CLASSIFIERS):
                matrix[i, j] = np.mean(
                    [float(row[metric]) for row in rows if row["feature_set"] == feature_set and row["classifier"] == classifier]
                )
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.65, vmax=1.0, aspect="auto")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(len(CLASSIFIERS)), [DISPLAY_CLASSIFIERS[value] for value in CLASSIFIERS], rotation=25, ha="right")
        ax.set_yticks(range(len(PCA_FEATURES)), [DISPLAY_PCA[value] for value in PCA_FEATURES])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white" if matrix[i, j] > 0.88 else "black")
        ax.grid(False)
    fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    fig.suptitle("Dimension-Matched PCA and Structured Representations", fontweight="bold")
    fig.savefig(output_dir / "pca_all_classifiers_6class.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_ablation(rows: list[dict[str, object]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    summary = _summarize(rows, ("classifier", "feature_set"))
    lookup = {(row["classifier"], row["feature_set"]): row for row in summary}
    x = np.arange(len(ABLATION_FEATURES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    for index, classifier in enumerate(ABLATION_CLASSIFIERS):
        means = [float(lookup[(classifier, feature)]["accuracy_mean"]) for feature in ABLATION_FEATURES]
        stds = [float(lookup[(classifier, feature)]["accuracy_std"]) for feature in ABLATION_FEATURES]
        ax.bar(
            x + (index - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=DISPLAY_CLASSIFIERS[classifier],
            color=("#4C78A8", "#F58518")[index],
        )
    ax.set_xticks(x, [DISPLAY_ABLATION[value] for value in ABLATION_FEATURES], rotation=22, ha="right")
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("Accuracy (mean +/- SD)")
    ax.set_title("Loss Ablation Under Identical Few-Shot Splits", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_ablation_6class.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the non-data experiments required by the paper.")
    parser.add_argument("--dataset", default="diagnostics/gesture_sequence_dataset_chinese_dance_6class_after_fix.csv")
    parser.add_argument("--split-manifest", default="diagnostics/palm_fix_split_manifest_6class.csv")
    parser.add_argument("--output-dir", default="figures/paper_rewrite_main")
    parser.add_argument("--diagnostics-dir", default="diagnostics/paper_completion")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--runtime-max-frames", type=int, default=300)
    parser.add_argument("--force-ablation", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="Redraw cached PCA and ablation figures only.")
    args = parser.parse_args()

    dataset = (ROOT / args.dataset).resolve()
    manifest = (ROOT / args.split_manifest).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    diagnostics_dir = (ROOT / args.diagnostics_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        pca_rows = _read_rows(output_dir / "pca_all_classifiers_per_repeat_6class.csv")
        ablation_rows = _read_rows(output_dir / "loss_ablation_per_repeat_6class.csv")
        _plot_pca_suite(pca_rows, output_dir)
        _plot_ablation(ablation_rows, output_dir)
        evaluate_ablation_stability(
            diagnostics_dir / "ablation_features_6class.csv",
            ablation_rows,
            output_dir,
        )
        print(f"paper_figures={output_dir}")
        return

    evaluate_pca_suite(dataset, manifest, output_dir)
    benchmark_runtime(dataset, diagnostics_dir, args.runtime_max_frames)
    if not args.skip_ablation:
        ablation_dataset = generate_ablation_features(dataset, diagnostics_dir, args.workers, args.force_ablation)
        evaluate_ablation_suite(ablation_dataset, manifest, output_dir)
    print(f"paper_figures={output_dir}")
    print(f"paper_diagnostics={diagnostics_dir}")


if __name__ == "__main__":
    main()
