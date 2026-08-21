from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_svm as tsvm


CLASSIFIERS = ("svm", "knn", "rf", "mlp")
FEATURE_SETS = ("raw", "raw_pca17", "corrected", "optimized_action", "optimized_full")
DISPLAY_CLASSIFIERS = {"svm": "SVM", "knn": "KNN", "rf": "RandomForest", "mlp": "MLP"}
DISPLAY_FEATURES = {
    "raw": "Raw",
    "raw_pca17": "PCA-17",
    "corrected": "Corrected",
    "optimized_action": "Optimized Action",
    "optimized_full": "Optimized Full",
}
COLORS = {
    "raw": "#4C78A8",
    "raw_pca17": "#B279A2",
    "corrected": "#54A24B",
    "optimized_action": "#F58518",
    "optimized_full": "#E45756",
}
PAPER_LABELS = {
    "deer_horn": "Deer horn",
    "flower_pinch": "Flower pinch",
    "orchid_finger": "Orchid finger",
    "orchid_palm": "Orchid palm",
    "prayer_beads": "Prayer beads",
    "three_finger_bent": "Three-finger",
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


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_nested_splits(
    sequence_ids: list[str],
    labels: list[str],
    shots: tuple[int, ...],
    repeats: int,
    test_size: float,
    random_state: int,
) -> tuple[dict[tuple[int, int], dict[str, list[int]]], list[dict[str, object]]]:
    all_indices = np.arange(len(sequence_ids))
    classes = sorted(set(labels))
    labels_array = np.asarray(labels)
    maximum_shot = max(shots)
    splits: dict[tuple[int, int], dict[str, list[int]]] = {}
    manifest_rows: list[dict[str, object]] = []

    for repeat in range(repeats):
        seed = random_state + repeat
        train_pool, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        nested_by_class: dict[str, list[int]] = {}
        for class_index, label in enumerate(classes):
            class_pool = np.asarray([index for index in train_pool if labels_array[index] == label], dtype=int)
            if len(class_pool) < maximum_shot:
                raise ValueError(
                    f"Class {label!r} has only {len(class_pool)} train-pool sequences; "
                    f"cannot evaluate {maximum_shot}-shot."
                )
            rng = np.random.RandomState(seed + 1009 * (class_index + 1))
            nested_by_class[label] = rng.permutation(class_pool).tolist()

        for shot in shots:
            train_indices = []
            for label in classes:
                train_indices.extend(nested_by_class[label][:shot])
            train_indices = sorted(train_indices)
            test_list = sorted(int(index) for index in test_indices)
            splits[(repeat, shot)] = {"train": train_indices, "test": test_list}
            for split_name, indices in (("train", train_indices), ("test", test_list)):
                for index in indices:
                    manifest_rows.append(
                        {
                            "repeat": repeat,
                            "shot": shot,
                            "split": split_name,
                            "sequence_id": sequence_ids[index],
                            "label": labels[index],
                        }
                    )
    return splits, manifest_rows


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
            array = np.asarray([float(row[metric]) for row in values], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(array))
            result[f"{metric}_std"] = float(np.std(array))
            result[f"{metric}_ci95"] = float(1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))
        output.append(result)
    return output


def _paired_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    lookup = {
        (int(row["shot"]), int(row["repeat"]), str(row["classifier"]), str(row["feature_set"])): float(row["accuracy"])
        for row in rows
    }
    output: list[dict[str, object]] = []
    shots = sorted({int(row["shot"]) for row in rows})
    repeats = sorted({int(row["repeat"]) for row in rows})
    for shot in shots:
        for classifier in CLASSIFIERS:
            for baseline in ("raw", "raw_pca17", "corrected"):
                differences = np.asarray(
                    [
                        lookup[(shot, repeat, classifier, "optimized_action")]
                        - lookup[(shot, repeat, classifier, baseline)]
                        for repeat in repeats
                    ],
                    dtype=np.float64,
                )
                p_value = float("nan")
                if wilcoxon is not None and np.any(differences != 0):
                    p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
                output.append(
                    {
                        "shot": shot,
                        "classifier": classifier,
                        "comparison": f"optimized_action_minus_{baseline}",
                        "mean_difference": float(np.mean(differences)),
                        "std_difference": float(np.std(differences)),
                        "ci95_difference": float(1.96 * np.std(differences, ddof=1) / np.sqrt(len(differences))),
                        "positive_repeats": int(np.sum(differences > 0)),
                        "equal_repeats": int(np.sum(differences == 0)),
                        "negative_repeats": int(np.sum(differences < 0)),
                        "wilcoxon_p": p_value,
                    }
                )
    return output


def evaluate(
    dataset: Path,
    shots: tuple[int, ...],
    repeats: int,
    test_size: float,
    random_state: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[int, str, str], np.ndarray],
    list[str],
]:
    row_meta, feature_names, features = tsvm._load_dataset(dataset)
    _, raw_features = tsvm._select_features(feature_names, features, "raw")
    sequence_ids, labels, _ = tsvm._group_sequences(row_meta, raw_features)
    splits, manifest_rows = _build_nested_splits(
        sequence_ids, labels, shots, repeats, test_size, random_state
    )
    class_labels = sorted(set(labels))

    sequences_by_feature: dict[str, list[np.ndarray]] = {}
    descriptors_by_feature: dict[str, np.ndarray] = {}
    for feature_set in ("raw", "corrected", "optimized_action", "optimized_full"):
        _, selected = tsvm._select_features(feature_names, features, feature_set)
        ids, selected_labels, sequences = tsvm._group_sequences(row_meta, selected)
        if ids != sequence_ids or selected_labels != labels:
            raise RuntimeError(f"Sequence alignment changed for {feature_set}")
        sequences_by_feature[feature_set] = sequences
        descriptors_by_feature[feature_set] = np.stack(
            [tsvm._aggregate_sequence_array(sequence) for sequence in sequences]
        )

    rows: list[dict[str, object]] = []
    confusions: dict[tuple[int, str, str], np.ndarray] = defaultdict(
        lambda: np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    )
    for repeat in range(repeats):
        for shot in shots:
            train_indices = splits[(repeat, shot)]["train"]
            test_indices = splits[(repeat, shot)]["test"]
            y_train = [labels[index] for index in train_indices]
            y_test = [labels[index] for index in test_indices]

            matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for feature_set in ("raw", "corrected", "optimized_action", "optimized_full"):
                descriptors = descriptors_by_feature[feature_set]
                matrices[feature_set] = (descriptors[train_indices], descriptors[test_indices])

            raw_sequences = sequences_by_feature["raw"]
            pca_train, pca_test = tsvm._project_sequences_with_pca(
                [raw_sequences[index] for index in train_indices],
                [raw_sequences[index] for index in test_indices],
                17,
                random_state + repeat,
            )
            matrices["raw_pca17"] = (
                np.stack([tsvm._aggregate_sequence_array(sequence) for sequence in pca_train]),
                np.stack([tsvm._aggregate_sequence_array(sequence) for sequence in pca_test]),
            )

            for feature_set in FEATURE_SETS:
                x_train, x_test = matrices[feature_set]
                for classifier in CLASSIFIERS:
                    model = tsvm._build_model(_model_args(classifier), random_state + repeat)
                    model.fit(x_train, y_train)
                    prediction = model.predict(x_test)
                    confusions[(shot, classifier, feature_set)] += confusion_matrix(
                        y_test, prediction, labels=class_labels
                    )
                    rows.append(
                        {
                            "repeat": repeat,
                            "shot": shot,
                            "num_train": shot * len(class_labels),
                            "num_test": len(test_indices),
                            "classifier": classifier,
                            "feature_set": feature_set,
                            "accuracy": float(accuracy_score(y_test, prediction)),
                            "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
                            "kappa": float(cohen_kappa_score(y_test, prediction)),
                        }
                    )
    return rows, manifest_rows, dict(confusions), class_labels


def _plot_learning_curves(summary: list[dict[str, object]], output_dir: Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    shots = sorted({int(row["shot"]) for row in summary})
    lookup = {
        (int(row["shot"]), str(row["classifier"]), str(row["feature_set"])): row
        for row in summary
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), sharex=True, sharey=True)
    for ax, classifier in zip(axes.flat, CLASSIFIERS, strict=True):
        for feature_set in FEATURE_SETS:
            means = [float(lookup[(shot, classifier, feature_set)][f"{metric}_mean"]) for shot in shots]
            ci95 = [float(lookup[(shot, classifier, feature_set)][f"{metric}_ci95"]) for shot in shots]
            ax.errorbar(
                shots,
                means,
                yerr=ci95,
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=COLORS[feature_set],
                label=DISPLAY_FEATURES[feature_set],
            )
        ax.set_title(DISPLAY_CLASSIFIERS[classifier], fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(shots, [str(value) for value in shots])
        ax.set_ylim(0.15, 1.02)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Training sequences per class (shot)")
    axes[1, 1].set_xlabel("Training sequences per class (shot)")
    axes[0, 0].set_ylabel(metric.replace("_", " ").title())
    axes[1, 0].set_ylabel(metric.replace("_", " ").title())
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5, frameon=False)
    fig.suptitle(f"Shot-Sweep {metric.replace('_', ' ').title()} (95% CI)", y=0.985, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.96))
    fig.savefig(output_dir / f"shot_sweep_{metric}_6class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_differences(paired: list[dict[str, object]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in paired if row["comparison"] == "optimized_action_minus_corrected"]
    shots = sorted({int(row["shot"]) for row in rows})
    lookup = {(int(row["shot"]), str(row["classifier"])): row for row in rows}
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for classifier, color in zip(CLASSIFIERS, ("#4C78A8", "#72B7B2", "#F58518", "#B279A2"), strict=True):
        means = [float(lookup[(shot, classifier)]["mean_difference"]) for shot in shots]
        ci95 = [float(lookup[(shot, classifier)]["ci95_difference"]) for shot in shots]
        ax.errorbar(shots, means, yerr=ci95, marker="o", linewidth=2, capsize=3, color=color, label=DISPLAY_CLASSIFIERS[classifier])
    ax.axhline(0.0, color="#333333", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xticks(shots, [str(value) for value in shots])
    ax.set_xlabel("Training sequences per class (shot)")
    ax.set_ylabel("Accuracy difference: Optimized Action - Corrected")
    ax.set_title("Does Temporal Refinement Improve Recognition?", fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "shot_sweep_optimized_minus_corrected_6class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_cm(ax, cm: np.ndarray, labels: list[str], title: str) -> object:
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    display_labels = [PAPER_LABELS.get(label, label.replace("_", " ").title()) for label in labels]
    ax.set_xticks(range(len(labels)), display_labels, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), display_labels, fontsize=7)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = normalized[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="white" if value > 0.52 else "black")
    return image


def _plot_confusions(
    confusions: dict[tuple[int, str, str], np.ndarray],
    labels: list[str],
    shots: tuple[int, ...],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    if len(shots) == 4:
        rows, cols = 2, 2
    else:
        cols = min(3, len(shots))
        rows = math.ceil(len(shots) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.1 * rows), squeeze=False, constrained_layout=True)
    flat_axes = list(axes.flat)
    for ax, shot in zip(flat_axes, shots):
        image = _draw_cm(ax, confusions[(shot, "rf", "optimized_action")], labels, f"{shot}-shot")
    for ax in flat_axes[len(shots) :]:
        ax.set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.82, pad=0.01)
    fig.suptitle("RandomForest + Optimized Action Across Shot Counts", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "cm_shot_progression_rf_optimized_action_6class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected_features = ("raw", "raw_pca17", "corrected", "optimized_action")
    fig, axes = plt.subplots(
        len(shots),
        len(selected_features),
        figsize=(15.0, 3.45 * len(shots)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, shot in enumerate(shots):
        for col_index, feature_set in enumerate(selected_features):
            image = _draw_cm(
                axes[row_index, col_index],
                confusions[(shot, "rf", feature_set)],
                labels,
                f"{shot}-shot: {DISPLAY_FEATURES[feature_set]}",
            )
    fig.colorbar(image, ax=axes, shrink=0.82, pad=0.01)
    fig.suptitle("Representation Errors Across Shot Counts", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "cm_low_vs_high_shot_rf_representations_6class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_recall_heatmaps(
    confusions: dict[tuple[int, str, str], np.ndarray],
    labels: list[str],
    shots: tuple[int, ...],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for ax, feature_set in zip(axes, ("corrected", "optimized_action"), strict=True):
        matrix = []
        for shot in shots:
            cm = confusions[(shot, "rf", feature_set)]
            row_sums = cm.sum(axis=1)
            matrix.append(np.divide(np.diag(cm), row_sums, out=np.zeros(len(labels), dtype=float), where=row_sums != 0))
        matrix_array = np.asarray(matrix)
        image = ax.imshow(matrix_array, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(labels)), [PAPER_LABELS.get(label, label) for label in labels], rotation=30, ha="right")
        ax.set_yticks(range(len(shots)), [str(shot) for shot in shots])
        ax.set_xlabel("Gesture class")
        ax.set_ylabel("Shot")
        ax.set_title(f"RandomForest: {DISPLAY_FEATURES[feature_set]}", fontweight="bold")
        for row in range(matrix_array.shape[0]):
            for col in range(matrix_array.shape[1]):
                value = matrix_array[row, col]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.72 else "black")
        ax.grid(False)
    fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02, label="Recall")
    fig.suptitle("Per-Class Recall as Training Data Increase", fontweight="bold")
    fig.savefig(output_dir / "shot_sweep_per_class_recall_rf_6class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate representation robustness across shot counts.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5, 10, 20, 40])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    shots = tuple(sorted(set(args.shots)))
    if len(shots) < 2:
        raise SystemExit("Provide at least two shot values for the sensitivity analysis.")
    dataset = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, manifest_rows, confusions, labels = evaluate(
        dataset, shots, args.repeats, args.test_size, args.random_state
    )
    summary = _summarize(rows)
    paired = _paired_summary(rows)
    _write_rows(output_dir / "shot_sweep_per_repeat_6class.csv", rows)
    _write_rows(output_dir / "shot_sweep_summary_6class.csv", summary)
    _write_rows(output_dir / "shot_sweep_paired_stats_6class.csv", paired)
    _write_rows(output_dir / "shot_sweep_manifest_6class.csv", manifest_rows)
    confusion_rows = []
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
    _write_rows(output_dir / "shot_sweep_confusions_6class.csv", confusion_rows)
    _plot_learning_curves(summary, output_dir, "accuracy")
    _plot_learning_curves(summary, output_dir, "macro_f1")
    _plot_paired_differences(paired, output_dir)
    _plot_confusions(confusions, labels, shots, output_dir)
    _plot_recall_heatmaps(confusions, labels, shots, output_dir)
    print(f"shot_sweep_output={output_dir}")
    print(f"shots={','.join(str(value) for value in shots)}")
    print(f"repeats={args.repeats}")


if __name__ == "__main__":
    main()
