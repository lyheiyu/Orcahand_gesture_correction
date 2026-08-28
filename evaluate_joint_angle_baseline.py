from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_shot_sweep_figures as sweep
import train_svm as tsvm


CLASSIFIERS = ("svm", "knn", "rf", "mlp")
FEATURE_SETS = (
    "raw",
    "raw_pca11",
    "raw_pca17",
    "joint_angle",
    "corrected",
    "optimized_action",
    "optimized_full",
)
DISPLAY_FEATURES = {
    "raw": "Raw-63",
    "raw_pca11": "PCA-11",
    "raw_pca17": "PCA-17",
    "joint_angle": "JointAngle-11",
    "corrected": "Corrected-17",
    "optimized_action": "Optimized Action-17",
    "optimized_full": "Optimized Full-63",
}
COLORS = {
    "raw": "#4C78A8",
    "raw_pca11": "#9C755F",
    "raw_pca17": "#B279A2",
    "joint_angle": "#ECA82C",
    "corrected": "#54A24B",
    "optimized_action": "#F58518",
    "optimized_full": "#E45756",
}
REPRESENTATION_INFO = {
    "raw": (63, "Cartesian landmark coordinates", "No", "No", "No"),
    "raw_pca11": (11, "Statistical latent representation", "No", "No", "No"),
    "raw_pca17": (17, "Statistical latent representation", "No", "No", "No"),
    "joint_angle": (11, "Geometric joint-angle representation", "No", "No", "No"),
    "corrected": (17, "Embodiment-aware actuator representation", "No", "Yes", "No"),
    "optimized_action": (17, "Kinematic/temporal refined actuator representation", "Yes", "Yes", "Yes"),
    "optimized_full": (63, "Reconstructed Cartesian landmark representation", "Yes", "Yes", "Yes"),
}


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load(dataset: Path) -> tuple[list[str], list[str], dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    row_meta, feature_names, features = tsvm._load_dataset(dataset)
    _, raw_features = tsvm._select_features(feature_names, features, "raw")
    sequence_ids, labels, raw_sequences = tsvm._group_sequences(row_meta, raw_features)
    sequences: dict[str, list[np.ndarray]] = {"raw": raw_sequences}
    descriptors: dict[str, np.ndarray] = {
        "raw": np.stack([tsvm._aggregate_sequence_array(sequence) for sequence in raw_sequences])
    }
    for feature_set in ("joint_angle", "corrected", "optimized_action", "optimized_full"):
        _, selected = tsvm._select_features(feature_names, features, feature_set)
        ids, selected_labels, grouped = tsvm._group_sequences(row_meta, selected)
        if ids != sequence_ids or selected_labels != labels:
            raise RuntimeError(f"Sequence alignment changed for {feature_set}")
        sequences[feature_set] = grouped
        descriptors[feature_set] = np.stack(
            [tsvm._aggregate_sequence_array(sequence) for sequence in grouped]
        )
    return sequence_ids, labels, sequences, descriptors


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["shot"]), str(row["classifier"]), str(row["feature_set"]))].append(row)
    output: list[dict[str, object]] = []
    for (shot, classifier, feature_set), values in sorted(grouped.items()):
        result: dict[str, object] = {
            "shot": shot,
            "num_train": shot * 6,
            "classifier": classifier,
            "feature_set": feature_set,
            "repeats": len(values),
        }
        for metric in ("accuracy", "macro_f1", "kappa"):
            array = np.asarray([float(value[metric]) for value in values], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(array))
            result[f"{metric}_std"] = float(np.std(array))
            result[f"{metric}_ci95"] = float(
                1.96 * np.std(array, ddof=1) / np.sqrt(len(array))
            )
        output.append(result)
    return output


def _paired(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    lookup = {
        (int(row["shot"]), int(row["repeat"]), str(row["classifier"]), str(row["feature_set"])): row
        for row in rows
    }
    comparisons = (
        ("joint_angle", "raw"),
        ("joint_angle", "raw_pca11"),
        ("corrected", "joint_angle"),
        ("optimized_action", "joint_angle"),
        ("optimized_action", "corrected"),
    )
    output: list[dict[str, object]] = []
    for shot in sorted({int(row["shot"]) for row in rows}):
        for classifier in CLASSIFIERS:
            repeats = sorted({int(row["repeat"]) for row in rows})
            for first, second in comparisons:
                for metric in ("accuracy", "macro_f1"):
                    differences = np.asarray(
                        [
                            float(lookup[(shot, repeat, classifier, first)][metric])
                            - float(lookup[(shot, repeat, classifier, second)][metric])
                            for repeat in repeats
                        ],
                        dtype=np.float64,
                    )
                    p_value = float("nan")
                    if wilcoxon is not None and np.any(differences != 0):
                        p_value = float(wilcoxon(differences).pvalue)
                    output.append(
                        {
                            "shot": shot,
                            "classifier": classifier,
                            "metric": metric,
                            "comparison": f"{first}_minus_{second}",
                            "mean_difference": float(np.mean(differences)),
                            "std_difference": float(np.std(differences)),
                            "ci95_difference": float(
                                1.96 * np.std(differences, ddof=1) / np.sqrt(len(differences))
                            ),
                            "positive_repeats": int(np.sum(differences > 0)),
                            "equal_repeats": int(np.sum(differences == 0)),
                            "negative_repeats": int(np.sum(differences < 0)),
                            "wilcoxon_p": p_value,
                        }
                    )
    return output


def _joint_angle_stability(sequences: list[np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, sequence in enumerate(sequences):
        velocity = np.linalg.norm(np.diff(sequence.astype(np.float64), axis=0), axis=1)
        acceleration = (
            np.linalg.norm(sequence[2:] - 2.0 * sequence[1:-1] + sequence[:-2], axis=1)
            if len(sequence) >= 3
            else np.asarray([], dtype=np.float64)
        )
        rows.append(
            {
                "sequence_index": index,
                "num_frames": len(sequence),
                "mean_velocity_deg": float(np.mean(velocity)) if velocity.size else 0.0,
                "max_velocity_deg": float(np.max(velocity)) if velocity.size else 0.0,
                "mean_acceleration_deg": float(np.mean(acceleration)) if acceleration.size else 0.0,
                "max_acceleration_deg": float(np.max(acceleration)) if acceleration.size else 0.0,
            }
        )
    return rows


def evaluate(
    dataset: Path,
    shots: tuple[int, ...],
    repeats: int,
    test_size: float,
    random_state: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[int, str, str], np.ndarray], list[str], list[dict[str, object]]]:
    sequence_ids, labels, sequences, descriptors = _load(dataset)
    splits, manifest = sweep._build_nested_splits(
        sequence_ids, labels, shots, repeats, test_size, random_state
    )
    class_labels = sorted(set(labels))
    confusions: dict[tuple[int, str, str], np.ndarray] = defaultdict(
        lambda: np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    )
    rows: list[dict[str, object]] = []

    for repeat in range(repeats):
        for shot in shots:
            train_indices = splits[(repeat, shot)]["train"]
            test_indices = splits[(repeat, shot)]["test"]
            y_train = [labels[index] for index in train_indices]
            y_test = [labels[index] for index in test_indices]
            matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for feature_set in ("raw", "joint_angle", "corrected", "optimized_action", "optimized_full"):
                matrix = descriptors[feature_set]
                matrices[feature_set] = (matrix[train_indices], matrix[test_indices])

            raw_train = [sequences["raw"][index] for index in train_indices]
            raw_test = [sequences["raw"][index] for index in test_indices]
            for dimensions in (11, 17):
                projected_train, projected_test = tsvm._project_sequences_with_pca(
                    raw_train, raw_test, dimensions, random_state + repeat
                )
                matrices[f"raw_pca{dimensions}"] = (
                    np.stack([tsvm._aggregate_sequence_array(value) for value in projected_train]),
                    np.stack([tsvm._aggregate_sequence_array(value) for value in projected_test]),
                )

            for feature_set in FEATURE_SETS:
                x_train, x_test = matrices[feature_set]
                for classifier in CLASSIFIERS:
                    model = tsvm._build_model(sweep._model_args(classifier), random_state + repeat)
                    model.fit(x_train, y_train)
                    prediction = model.predict(x_test)
                    confusions[(shot, classifier, feature_set)] += confusion_matrix(
                        y_test, prediction, labels=class_labels
                    )
                    rows.append(
                        {
                            "repeat": repeat,
                            "shot": shot,
                            "num_train": len(train_indices),
                            "num_test": len(test_indices),
                            "classifier": classifier,
                            "feature_set": feature_set,
                            "accuracy": float(accuracy_score(y_test, prediction)),
                            "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
                            "kappa": float(cohen_kappa_score(y_test, prediction)),
                        }
                    )
    return rows, manifest, dict(confusions), class_labels, _joint_angle_stability(sequences["joint_angle"])


def _plot_curves(summary: list[dict[str, object]], output_dir: Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    lookup = {
        (int(row["shot"]), str(row["classifier"]), str(row["feature_set"])): row
        for row in summary
    }
    shots = sorted({int(row["shot"]) for row in summary})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True, sharey=True)
    for ax, classifier in zip(axes.flat, CLASSIFIERS, strict=True):
        for feature_set in FEATURE_SETS:
            means = [float(lookup[(shot, classifier, feature_set)][f"{metric}_mean"]) for shot in shots]
            ci95 = [float(lookup[(shot, classifier, feature_set)][f"{metric}_ci95"]) for shot in shots]
            ax.errorbar(
                shots,
                means,
                yerr=ci95,
                marker="o",
                linewidth=1.9,
                capsize=3,
                color=COLORS[feature_set],
                label=DISPLAY_FEATURES[feature_set],
            )
        ax.set_title(sweep.DISPLAY_CLASSIFIERS[classifier], fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(shots, [str(value) for value in shots])
        ax.set_ylim(0.15, 1.02)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Training sequences per class (shot)")
    axes[1, 1].set_xlabel("Training sequences per class (shot)")
    axes[0, 0].set_ylabel(metric.replace("_", " ").title())
    axes[1, 0].set_ylabel(metric.replace("_", " ").title())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False)
    fig.suptitle(f"Joint-Angle Baseline: {metric.replace('_', ' ').title()} (95% CI)", y=0.985, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.96))
    fig.savefig(output_dir / f"joint_angle_shot_sweep_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_cm(ax, matrix: np.ndarray, labels: list[str], title: str):
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    short_labels = [sweep.PAPER_LABELS.get(label, label) for label in labels]
    ax.set_xticks(range(len(labels)), short_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), short_labels, fontsize=8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = normalized[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.55 else "black")
    return image


def _plot_confusions(confusions, labels, output_dir: Path, shot: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    for ax, feature_set in zip(axes, ("joint_angle", "corrected", "optimized_action"), strict=True):
        image = _draw_cm(ax, confusions[(shot, "rf", feature_set)], labels, f"RF: {DISPLAY_FEATURES[feature_set]}")
    fig.colorbar(image, ax=axes, shrink=0.78, label="Recall")
    fig.suptitle(f"Geometric and Actuator Representations ({shot}-shot)", fontweight="bold")
    fig.savefig(output_dir / f"cm_joint_angle_vs_actuator_rf_{shot}shot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), constrained_layout=True)
    for ax, classifier in zip(axes.flat, CLASSIFIERS, strict=True):
        image = _draw_cm(
            ax,
            confusions[(shot, classifier, "joint_angle")],
            labels,
            f"{sweep.DISPLAY_CLASSIFIERS[classifier]}: JointAngle-11",
        )
    fig.colorbar(image, ax=axes, shrink=0.78, label="Recall")
    fig.suptitle(f"JointAngle-11 Confusion Matrices ({shot}-shot)", fontweight="bold")
    fig.savefig(output_dir / f"cm_joint_angle_all_classifiers_{shot}shot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _representation_table(summary: list[dict[str, object]], shot: int) -> list[dict[str, object]]:
    lookup = {
        (str(row["classifier"]), str(row["feature_set"])): row
        for row in summary
        if int(row["shot"]) == shot
    }
    rows: list[dict[str, object]] = []
    for feature_set in FEATURE_SETS:
        dimension, kind, temporal, orca, mujoco = REPRESENTATION_INFO[feature_set]
        row: dict[str, object] = {
            "representation": DISPLAY_FEATURES[feature_set],
            "dimension": dimension,
            "representation_type": kind,
            "uses_temporal_information": temporal,
            "uses_orca_embodiment": orca,
            "uses_mujoco": mujoco,
        }
        for classifier in CLASSIFIERS:
            value = lookup[(classifier, feature_set)]
            row[f"{classifier}_accuracy_mean"] = value["accuracy_mean"]
            row[f"{classifier}_macro_f1_mean"] = value["macro_f1_mean"]
        rows.append(row)
    return rows


def _write_report(path: Path, summary, paired, invalid_summary: dict[str, object], shot: int) -> None:
    lookup = {
        (str(row["classifier"]), str(row["feature_set"])): row
        for row in summary
        if int(row["shot"]) == shot
    }
    lines = [
        "# JointAngle-11 Baseline Results",
        "",
        f"Primary comparison: {shot}-shot, with the same nested sequence splits and preprocessing rules.",
        "",
        f"Invalid angle values: {invalid_summary.get('invalid_values', 'unknown')} across "
        f"{invalid_summary.get('num_frames', 'unknown')} frames.",
        "",
        "## Primary Accuracy",
        "",
        "| Classifier | Raw | PCA-11 | JointAngle-11 | PCA-17 | Corrected | Optimized Action | Optimized Full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for classifier in CLASSIFIERS:
        values = [
            float(lookup[(classifier, feature)]["accuracy_mean"])
            for feature in (
                "raw",
                "raw_pca11",
                "joint_angle",
                "raw_pca17",
                "corrected",
                "optimized_action",
                "optimized_full",
            )
        ]
        lines.append(
            f"| {sweep.DISPLAY_CLASSIFIERS[classifier]} | "
            + " | ".join(f"{value:.4f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired Accuracy Differences",
            "",
            "Positive values favor the first representation. Differences are percentage points.",
            "",
            "| Classifier | Comparison | Difference (pp) | 95% CI (pp) | Wilcoxon p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for classifier in CLASSIFIERS:
        for comparison in (
            "joint_angle_minus_raw",
            "joint_angle_minus_raw_pca11",
            "corrected_minus_joint_angle",
            "optimized_action_minus_joint_angle",
            "optimized_action_minus_corrected",
        ):
            row = next(
                value
                for value in paired
                if int(value["shot"]) == shot
                and value["classifier"] == classifier
                and value["metric"] == "accuracy"
                and value["comparison"] == comparison
            )
            lines.append(
                f"| {sweep.DISPLAY_CLASSIFIERS[classifier]} | {comparison.replace('_', ' ')} | "
                f"{100.0 * float(row['mean_difference']):.2f} | "
                f"{100.0 * float(row['ci95_difference']):.2f} | "
                f"{float(row['wilcoxon_p']):.4g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "JointAngle-11 tests conventional geometric reparameterization without ORCA or MuJoCo. "
            "Corrected tests embodiment-aware actuator mapping, while Optimized Action additionally "
            "uses MuJoCo and causal temporal regularization. Absolute temporal-difference magnitudes "
            "in degrees must not be compared directly with actuator-space magnitudes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JointAngle-11 with matched PCA and actuator baselines.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--invalid-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--primary-shot", type=int, default=3)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shots = tuple(sorted(set(args.shots)))
    if args.primary_shot not in shots:
        raise SystemExit("--primary-shot must be included in --shots")
    invalid_summary = json.loads(Path(args.invalid_summary).read_text(encoding="utf-8"))

    rows, manifest, confusions, labels, stability = evaluate(
        dataset, shots, args.repeats, args.test_size, args.random_state
    )
    summary = _summarize(rows)
    paired = _paired(rows)
    _write_rows(output_dir / "joint_angle_per_repeat.csv", rows)
    _write_rows(output_dir / "joint_angle_summary.csv", summary)
    _write_rows(output_dir / "joint_angle_paired_stats.csv", paired)
    _write_rows(output_dir / "joint_angle_split_manifest.csv", manifest)
    _write_rows(output_dir / "joint_angle_stability_degrees.csv", stability)
    _write_rows(
        output_dir / "joint_angle_representation_table.csv",
        _representation_table(summary, args.primary_shot),
    )

    confusion_rows: list[dict[str, object]] = []
    for (shot, classifier, feature_set), matrix in sorted(confusions.items()):
        for true_index, true_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                confusion_rows.append(
                    {
                        "shot": shot,
                        "classifier": classifier,
                        "feature_set": feature_set,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )
    _write_rows(output_dir / "joint_angle_confusions.csv", confusion_rows)
    _plot_curves(summary, output_dir, "accuracy")
    _plot_curves(summary, output_dir, "macro_f1")
    _plot_confusions(confusions, labels, output_dir, args.primary_shot)
    _write_report(
        output_dir / "JOINT_ANGLE_RESULTS.md",
        summary,
        paired,
        invalid_summary,
        args.primary_shot,
    )
    print(f"dataset={dataset}")
    print(f"output_dir={output_dir}")
    print(f"shots={','.join(str(value) for value in shots)} repeats={args.repeats}")


if __name__ == "__main__":
    main()
