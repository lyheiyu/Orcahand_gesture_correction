from __future__ import annotations

import argparse
import csv
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


FEATURE_SETS = ("raw", "raw_pca17", "corrected", "optimized_action", "optimized_full")
CLASSIFIERS = ("svm", "knn", "rf", "mlp")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_representations(
    dataset: Path,
) -> tuple[list[str], list[str], dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    row_meta, feature_names, features = tsvm._load_dataset(dataset)
    _, raw = tsvm._select_features(feature_names, features, "raw")
    sequence_ids, labels, _ = tsvm._group_sequences(row_meta, raw)
    sequences: dict[str, list[np.ndarray]] = {}
    descriptors: dict[str, np.ndarray] = {}
    for feature_set in ("raw", "corrected", "optimized_action", "optimized_full"):
        _, selected = tsvm._select_features(feature_names, features, feature_set)
        ids, selected_labels, grouped = tsvm._group_sequences(row_meta, selected)
        if ids != sequence_ids or selected_labels != labels:
            raise RuntimeError(f"Sequence alignment changed for {feature_set} in {dataset}")
        sequences[feature_set] = grouped
        descriptors[feature_set] = np.stack(
            [tsvm._aggregate_sequence_array(sequence) for sequence in grouped]
        )
    return sequence_ids, labels, sequences, descriptors


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["shot"]),
            str(row["classifier"]),
            str(row["feature_set"]),
            str(row["condition"]),
        )
        grouped[key].append(row)

    output: list[dict[str, object]] = []
    for (shot, classifier, feature_set, condition), values in sorted(grouped.items()):
        result: dict[str, object] = {
            "shot": shot,
            "classifier": classifier,
            "feature_set": feature_set,
            "condition": condition,
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


def _degradation_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    lookup = {
        (
            int(row["repeat"]),
            int(row["shot"]),
            str(row["classifier"]),
            str(row["feature_set"]),
            str(row["condition"]),
        ): float(row["accuracy"])
        for row in rows
    }
    repeats = sorted({int(row["repeat"]) for row in rows})
    shots = sorted({int(row["shot"]) for row in rows})
    output: list[dict[str, object]] = []
    for shot in shots:
        for classifier in CLASSIFIERS:
            degradation_by_feature: dict[str, np.ndarray] = {}
            for feature_set in FEATURE_SETS:
                degradation = np.asarray(
                    [
                        lookup[(repeat, shot, classifier, feature_set, "occluded")]
                        - lookup[(repeat, shot, classifier, feature_set, "clean")]
                        for repeat in repeats
                    ],
                    dtype=np.float64,
                )
                degradation_by_feature[feature_set] = degradation
                p_value = float("nan")
                if wilcoxon is not None and np.any(degradation != 0):
                    p_value = float(wilcoxon(degradation).pvalue)
                output.append(
                    {
                        "shot": shot,
                        "classifier": classifier,
                        "comparison": f"{feature_set}_occluded_minus_clean",
                        "mean_difference": float(np.mean(degradation)),
                        "std_difference": float(np.std(degradation)),
                        "ci95_difference": float(
                            1.96 * np.std(degradation, ddof=1) / np.sqrt(len(degradation))
                        ),
                        "positive_repeats": int(np.sum(degradation > 0)),
                        "equal_repeats": int(np.sum(degradation == 0)),
                        "negative_repeats": int(np.sum(degradation < 0)),
                        "wilcoxon_p": p_value,
                    }
                )

            relative = (
                degradation_by_feature["optimized_action"]
                - degradation_by_feature["corrected"]
            )
            p_value = float("nan")
            if wilcoxon is not None and np.any(relative != 0):
                p_value = float(wilcoxon(relative).pvalue)
            output.append(
                {
                    "shot": shot,
                    "classifier": classifier,
                    "comparison": "optimized_action_minus_corrected_degradation",
                    "mean_difference": float(np.mean(relative)),
                    "std_difference": float(np.std(relative)),
                    "ci95_difference": float(
                        1.96 * np.std(relative, ddof=1) / np.sqrt(len(relative))
                    ),
                    "positive_repeats": int(np.sum(relative > 0)),
                    "equal_repeats": int(np.sum(relative == 0)),
                    "negative_repeats": int(np.sum(relative < 0)),
                    "wilcoxon_p": p_value,
                }
            )
    return output


def evaluate(
    clean_path: Path,
    occluded_path: Path,
    shots: tuple[int, ...],
    repeats: int,
    test_size: float,
    random_state: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[int, str, str, str], np.ndarray], list[str]]:
    clean_ids, labels, clean_sequences, clean_descriptors = _load_representations(clean_path)
    occluded_ids, occluded_labels, occluded_sequences, occluded_descriptors = _load_representations(
        occluded_path
    )
    if clean_ids != occluded_ids or labels != occluded_labels:
        raise RuntimeError("Clean and occluded datasets do not contain identical ordered sequences")

    splits, manifest = sweep._build_nested_splits(
        clean_ids, labels, shots, repeats, test_size, random_state
    )
    class_labels = sorted(set(labels))
    confusions: dict[tuple[int, str, str, str], np.ndarray] = defaultdict(
        lambda: np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    )
    rows: list[dict[str, object]] = []

    for repeat in range(repeats):
        for shot in shots:
            train_indices = splits[(repeat, shot)]["train"]
            test_indices = splits[(repeat, shot)]["test"]
            y_train = [labels[index] for index in train_indices]
            y_test = [labels[index] for index in test_indices]
            matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for feature_set in ("raw", "corrected", "optimized_action", "optimized_full"):
                clean_matrix = clean_descriptors[feature_set]
                occluded_matrix = occluded_descriptors[feature_set]
                matrices[feature_set] = (
                    clean_matrix[train_indices],
                    clean_matrix[test_indices],
                    occluded_matrix[test_indices],
                )

            raw_train = [clean_sequences["raw"][index] for index in train_indices]
            raw_clean_test = [clean_sequences["raw"][index] for index in test_indices]
            raw_occluded_test = [occluded_sequences["raw"][index] for index in test_indices]
            projected_train, projected_tests = tsvm._project_sequences_with_pca(
                raw_train,
                raw_clean_test + raw_occluded_test,
                17,
                random_state + repeat,
            )
            split_at = len(raw_clean_test)
            matrices["raw_pca17"] = (
                np.stack([tsvm._aggregate_sequence_array(sequence) for sequence in projected_train]),
                np.stack(
                    [tsvm._aggregate_sequence_array(sequence) for sequence in projected_tests[:split_at]]
                ),
                np.stack(
                    [tsvm._aggregate_sequence_array(sequence) for sequence in projected_tests[split_at:]]
                ),
            )

            for feature_set in FEATURE_SETS:
                x_train, x_clean_test, x_occluded_test = matrices[feature_set]
                for classifier in CLASSIFIERS:
                    model = tsvm._build_model(sweep._model_args(classifier), random_state + repeat)
                    model.fit(x_train, y_train)
                    for condition, x_test in (
                        ("clean", x_clean_test),
                        ("occluded", x_occluded_test),
                    ):
                        prediction = model.predict(x_test)
                        confusions[(shot, classifier, feature_set, condition)] += confusion_matrix(
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
                                "condition": condition,
                                "accuracy": float(accuracy_score(y_test, prediction)),
                                "macro_f1": float(
                                    f1_score(y_test, prediction, average="macro", zero_division=0)
                                ),
                                "kappa": float(cohen_kappa_score(y_test, prediction)),
                            }
                        )
    return rows, manifest, dict(confusions), class_labels


def _plot_accuracy(summary: list[dict[str, object]], shots: tuple[int, ...], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    lookup = {
        (
            int(row["shot"]),
            str(row["classifier"]),
            str(row["feature_set"]),
            str(row["condition"]),
        ): row
        for row in summary
    }
    for shot in shots:
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), sharey=True)
        x = np.arange(len(FEATURE_SETS))
        width = 0.36
        for ax, classifier in zip(axes.flat, CLASSIFIERS, strict=True):
            clean = [float(lookup[(shot, classifier, feature, "clean")]["accuracy_mean"]) for feature in FEATURE_SETS]
            occluded = [float(lookup[(shot, classifier, feature, "occluded")]["accuracy_mean"]) for feature in FEATURE_SETS]
            ax.bar(x - width / 2, clean, width, label="Clean", color="#4C78A8")
            ax.bar(x + width / 2, occluded, width, label="Occluded", color="#E45756")
            ax.set_title(sweep.DISPLAY_CLASSIFIERS[classifier], fontweight="bold")
            ax.set_xticks(x, [sweep.DISPLAY_FEATURES[value] for value in FEATURE_SETS], rotation=25, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.25)
        axes[0, 0].set_ylabel("Accuracy")
        axes[1, 0].set_ylabel("Accuracy")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            names,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=2,
            frameon=False,
        )
        fig.suptitle(
            f"Clean-to-Occluded Robustness ({shot}-shot)",
            y=0.985,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
        fig.savefig(output_dir / f"occlusion_accuracy_clean_vs_occluded_{shot}shot.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on clean sequences and test on matched clean/occluded sequences.")
    parser.add_argument("--clean", required=True)
    parser.add_argument("--occluded", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[3])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    clean_path = Path(args.clean).resolve()
    occluded_path = Path(args.occluded).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shots = tuple(sorted(set(args.shots)))

    rows, manifest, confusions, labels = evaluate(
        clean_path,
        occluded_path,
        shots,
        args.repeats,
        args.test_size,
        args.random_state,
    )
    summary = _summarize(rows)
    degradation = _degradation_summary(rows)
    _write_rows(output_dir / "occlusion_per_repeat.csv", rows)
    _write_rows(output_dir / "occlusion_summary.csv", summary)
    _write_rows(output_dir / "occlusion_degradation_paired.csv", degradation)
    _write_rows(output_dir / "occlusion_split_manifest.csv", manifest)

    confusion_rows: list[dict[str, object]] = []
    for (shot, classifier, feature_set, condition), matrix in sorted(confusions.items()):
        for true_index, true_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                confusion_rows.append(
                    {
                        "shot": shot,
                        "classifier": classifier,
                        "feature_set": feature_set,
                        "condition": condition,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )
    _write_rows(output_dir / "occlusion_confusions.csv", confusion_rows)
    _plot_accuracy(summary, shots, output_dir)
    print(f"clean={clean_path}")
    print(f"occluded={occluded_path}")
    print(f"output_dir={output_dir}")
    print(f"shots={','.join(str(value) for value in shots)} repeats={args.repeats}")


if __name__ == "__main__":
    main()
