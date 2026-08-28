from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate_orca_dimension_control as dimension_control
import evaluate_sequence_encodings as sequence_encoding
import generate_shot_sweep_figures as shot_sweep
import train_svm
from run_compact_orca_selection import FROZEN_INDICES


REPRESENTATIONS = ("joint_angle11", "compact_optimized_action7")
DISPLAY_REPRESENTATIONS = {
    "joint_angle11": "JointAngle-11",
    "compact_optimized_action7": "Compact Refined-7",
}
COLORS = {"joint_angle11": "Blues", "compact_optimized_action7": "Oranges"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _choose_training_indices(
    development: list[int], labels: list[str], shot: int, repeat_seed: int
) -> list[int]:
    chosen: list[int] = []
    classes = sorted(set(labels[index] for index in development))
    for class_offset, label in enumerate(classes):
        pool = np.asarray([index for index in development if labels[index] == label], dtype=int)
        rng = np.random.RandomState(repeat_seed + 1009 * (class_offset + 1))
        chosen.extend(rng.permutation(pool)[:shot].tolist())
    return sorted(chosen)


def _annotate_matrix(ax: plt.Axes, counts: np.ndarray, normalized: np.ndarray) -> None:
    threshold = 0.52 * float(np.max(normalized)) if normalized.size else 0.0
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            value = normalized[row, column]
            color = "white" if value > threshold else "#20252B"
            ax.text(
                column,
                row,
                f"{value:.2f}\n({counts[row, column]})",
                ha="center",
                va="center",
                fontsize=7.2,
                color=color,
            )


def _plot_matrix(
    ax: plt.Axes,
    counts: np.ndarray,
    labels: list[str],
    title: str,
    cmap: str,
    *,
    show_y: bool = True,
) -> None:
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap=cmap)
    _annotate_matrix(ax, counts, normalized)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    if show_y:
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.set_ylabel("True label")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("Predicted label")
    return image


def _save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate(
    dataset: Path,
    frozen_dir: Path,
    output_dir: Path,
    main_figure: Path,
    *,
    shot: int,
    repeats: int,
    seed: int,
) -> None:
    development_path = frozen_dir / "development_sequences.csv"
    final_path = frozen_dir / "final_test_sequences.csv"
    if not development_path.exists() or not final_path.exists():
        raise FileNotFoundError("Frozen development/final manifests are required.")

    sequence_ids, labels, sequences = dimension_control._load_base(dataset)
    id_to_index = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}
    development_rows = _read_rows(development_path)
    final_rows = _read_rows(final_path)
    development = [id_to_index[row["sequence_id"]] for row in development_rows]
    final = [id_to_index[row["sequence_id"]] for row in final_rows]
    final_labels = [labels[index] for index in final]
    class_labels = sorted(set(final_labels))
    frozen_metrics_path = frozen_dir / "final_test_per_repeat.csv"
    frozen_metrics = {
        (row["classifier"], row["representation"], int(row["repeat"])): row
        for row in _read_rows(frozen_metrics_path)
    }

    predictions: list[dict[str, object]] = []
    reproduction_rows: list[dict[str, object]] = []
    aggregate: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    )

    for repeat in range(repeats):
        repeat_seed = seed + repeat
        chosen_train = _choose_training_indices(development, labels, shot, repeat_seed)
        train_labels = [labels[index] for index in chosen_train]
        train_by_representation = {
            "joint_angle11": [sequences["joint_angle11"][index] for index in chosen_train],
            "compact_optimized_action7": [
                sequences["optimized_action17"][index][:, FROZEN_INDICES] for index in chosen_train
            ],
        }
        test_by_representation = {
            "joint_angle11": [sequences["joint_angle11"][index] for index in final],
            "compact_optimized_action7": [
                sequences["optimized_action17"][index][:, FROZEN_INDICES] for index in final
            ],
        }

        for representation in REPRESENTATIONS:
            x_train = np.stack(
                [sequence_encoding.encode_sequence(value, "resample16")
                 for value in train_by_representation[representation]]
            )
            x_test = np.stack(
                [sequence_encoding.encode_sequence(value, "resample16")
                 for value in test_by_representation[representation]]
            )
            for classifier in dimension_control.CLASSIFIERS:
                model = train_svm._build_model(shot_sweep._model_args(classifier), repeat_seed)
                model.fit(x_train, train_labels)
                predicted = model.predict(x_test)
                matrix = confusion_matrix(final_labels, predicted, labels=class_labels)
                aggregate[(classifier, representation)] += matrix
                actual = {
                    "accuracy": float(accuracy_score(final_labels, predicted)),
                    "macro_f1": float(f1_score(final_labels, predicted, average="macro", zero_division=0)),
                    "kappa": float(cohen_kappa_score(final_labels, predicted)),
                }
                frozen = frozen_metrics[(classifier, representation, repeat)]
                differences = {
                    metric: abs(actual[metric] - float(frozen[metric]))
                    for metric in ("accuracy", "macro_f1", "kappa")
                }
                reproduction_rows.append({
                    "repeat": repeat,
                    "classifier": classifier,
                    "representation": representation,
                    "accuracy_current": actual["accuracy"],
                    "accuracy_frozen": frozen["accuracy"],
                    "macro_f1_current": actual["macro_f1"],
                    "macro_f1_frozen": frozen["macro_f1"],
                    "kappa_current": actual["kappa"],
                    "kappa_frozen": frozen["kappa"],
                    "max_absolute_difference": max(differences.values()),
                    "exact_within_1e_12": "yes" if max(differences.values()) <= 1e-12 else "no",
                })
                for sequence_index, true_label, predicted_label in zip(final, final_labels, predicted, strict=True):
                    predictions.append({
                        "repeat": repeat,
                        "seed": repeat_seed,
                        "classifier": classifier,
                        "representation": representation,
                        "sequence_id": sequence_ids[sequence_index],
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                    })

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "frozen_final_predictions.csv", predictions)
    _write_rows(output_dir / "reproduction_validation.csv", reproduction_rows)

    matrix_rows: list[dict[str, object]] = []
    for (classifier, representation), matrix in aggregate.items():
        for true_index, true_label in enumerate(class_labels):
            row_total = int(matrix[true_index].sum())
            for predicted_index, predicted_label in enumerate(class_labels):
                count = int(matrix[true_index, predicted_index])
                matrix_rows.append({
                    "classifier": classifier,
                    "representation": representation,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "row_normalized": count / row_total if row_total else 0.0,
                })
    _write_rows(output_dir / "aggregate_confusion_matrices.csv", matrix_rows)

    for classifier in dimension_control.CLASSIFIERS:
        classifier_name = shot_sweep.DISPLAY_CLASSIFIERS[classifier]
        for representation in REPRESENTATIONS:
            fig, ax = plt.subplots(figsize=(7.4, 6.4))
            _plot_matrix(
                ax,
                aggregate[(classifier, representation)],
                class_labels,
                f"{classifier_name} - {DISPLAY_REPRESENTATIONS[representation]}",
                COLORS[representation],
            )
            fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, label="Row-normalized frequency")
            fig.tight_layout()
            _save_figure(fig, output_dir / f"cm_{classifier}_{representation}")

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharey=True)
        for column, representation in enumerate(REPRESENTATIONS):
            _plot_matrix(
                axes[column],
                aggregate[(classifier, representation)],
                class_labels,
                f"{classifier_name} - {DISPLAY_REPRESENTATIONS[representation]}",
                COLORS[representation],
                show_y=column == 0,
            )
        fig.suptitle(f"Frozen Final Test Across {repeats} Few-shot Repeats", weight="bold")
        fig.tight_layout()
        _save_figure(fig, output_dir / f"paired_cm_{classifier}")

    fig, axes = plt.subplots(4, 2, figsize=(15, 22), sharex=True, sharey=True)
    for row, classifier in enumerate(dimension_control.CLASSIFIERS):
        for column, representation in enumerate(REPRESENTATIONS):
            _plot_matrix(
                axes[row, column],
                aggregate[(classifier, representation)],
                class_labels,
                f"{shot_sweep.DISPLAY_CLASSIFIERS[classifier]} - {DISPLAY_REPRESENTATIONS[representation]}",
                COLORS[representation],
                show_y=column == 0,
            )
    fig.suptitle(f"Aggregate Confusion Matrices: {repeats} Repeats x 115 Final Sequences", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save_figure(fig, output_dir / "all_classifiers_confusion_overview")

    main_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharey=True)
    for column, representation in enumerate(REPRESENTATIONS):
        _plot_matrix(
            axes[column],
            aggregate[("svm", representation)],
            class_labels,
            f"SVM - {DISPLAY_REPRESENTATIONS[representation]}",
            COLORS[representation],
            show_y=column == 0,
        )
    fig.suptitle(f"Frozen Final Test Across {repeats} Few-shot Repeats", weight="bold")
    fig.tight_layout()
    _save_figure(fig, main_figure)

    metadata = {
        "dataset": str(dataset),
        "dataset_sha256": _digest(dataset),
        "development_manifest": str(development_path),
        "development_manifest_sha256": _digest(development_path),
        "final_manifest": str(final_path),
        "final_manifest_sha256": _digest(final_path),
        "shot": shot,
        "repeats": repeats,
        "seed": seed,
        "num_development_sequences": len(development),
        "num_final_sequences": len(final),
        "labels": class_labels,
        "representations": list(REPRESENTATIONS),
        "classifiers": list(dimension_control.CLASSIFIERS),
        "aggregation": "sum of confusion counts across all repeat-level final predictions",
        "metric_groups_exact_within_1e_12": sum(
            row["exact_within_1e_12"] == "yes" for row in reproduction_rows
        ),
        "metric_groups_total": len(reproduction_rows),
        "reproduction_note": (
            "CMs are regenerated predictions under the current software environment. "
            "See reproduction_validation.csv for comparison with frozen historical metrics."
        ),
    }
    (output_dir / "confusion_matrix_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"output={output_dir}")
    print(f"predictions={len(predictions)} matrices={len(aggregate)} labels={len(class_labels)}")
    print(f"main_figure={main_figure.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen aggregate confusion matrices.")
    parser.add_argument(
        "--dataset",
        default="diagnostics/updated_6class_20260820/gesture_sequence_dataset_chinese_dance_6class_after_fix.csv",
    )
    parser.add_argument("--frozen-dir", default="diagnostics/orca_compact_selection_20260827")
    parser.add_argument(
        "--output-dir",
        default="paper_final_compact_orca_20260827/figures/supplementary_confusion_matrices",
    )
    parser.add_argument(
        "--main-figure",
        default="paper_final_compact_orca_20260827/figures/figure_10_main_svm_confusion_comparison",
    )
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(
        Path(args.dataset).resolve(),
        Path(args.frozen_dir).resolve(),
        Path(args.output_dir).resolve(),
        Path(args.main_figure).resolve(),
        shot=args.shot,
        repeats=args.repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
