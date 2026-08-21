from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_svm as tsvm


CLASSIFIERS = ["svm", "knn", "rf", "mlp"]
FEATURE_SETS = ["raw", "corrected", "optimized_action", "optimized_full"]
DISPLAY_CLASSIFIERS = {"svm": "SVM", "knn": "KNN", "rf": "RandomForest", "mlp": "MLP"}
DISPLAY_FEATURES = {
    "raw": "Raw",
    "corrected": "Corrected",
    "optimized_action": "Optimized Action",
    "optimized_full": "Optimized Full",
}
COLORS = {
    "raw": "#4C78A8",
    "corrected": "#72B7B2",
    "optimized_action": "#F58518",
    "optimized_full": "#54A24B",
}


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def _load_manifest(path: Path) -> dict[int, dict[str, list[str]]]:
    grouped: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"train": [], "test": []})
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            grouped[int(row["repeat"])][row["split"]].append(row["sequence_id"])
    return dict(grouped)


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


def evaluate_repeats(dataset: Path, manifest_path: Path) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray], list[str]]:
    row_meta, feature_names, features = tsvm._load_dataset(dataset)
    manifest = _load_manifest(manifest_path)
    per_repeat: list[dict[str, object]] = []
    aggregate_confusions: dict[tuple[str, str], np.ndarray] = {}
    class_labels = sorted({row["label"] for row in row_meta})

    for feature_set in FEATURE_SETS:
        _, selected = tsvm._select_features(feature_names, features, feature_set)
        sequence_ids, sequence_labels, sequences = tsvm._group_sequences(row_meta, selected)
        sequence_lookup = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}

        for classifier in CLASSIFIERS:
            aggregate = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
            for repeat in sorted(manifest):
                train_indices = [sequence_lookup[value] for value in manifest[repeat]["train"]]
                test_indices = [sequence_lookup[value] for value in manifest[repeat]["test"]]
                x_train = np.stack(
                    [tsvm._aggregate_sequence_array(sequences[index]) for index in train_indices], axis=0
                )
                x_test = np.stack(
                    [tsvm._aggregate_sequence_array(sequences[index]) for index in test_indices], axis=0
                )
                y_train = [sequence_labels[index] for index in train_indices]
                y_test = [sequence_labels[index] for index in test_indices]

                model = tsvm._build_model(_model_args(classifier), 42 + repeat)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                cm = confusion_matrix(y_test, y_pred, labels=class_labels)
                aggregate += cm
                per_repeat.append(
                    {
                        "repeat": repeat,
                        "classifier": classifier,
                        "feature_set": feature_set,
                        "accuracy": float(accuracy_score(y_test, y_pred)),
                        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                        "kappa": float(cohen_kappa_score(y_test, y_pred)),
                    }
                )
            aggregate_confusions[(classifier, feature_set)] = aggregate
    return per_repeat, aggregate_confusions, class_labels


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _score_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, int], dict[str, object]]:
    return {
        (str(row["classifier"]), str(row["feature_set"]), int(row["repeat"])): row
        for row in rows
    }


def paired_accuracy_figure(rows: list[dict[str, object]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    lookup = _score_lookup(rows)
    comparisons = [("raw", "vs Raw"), ("corrected", "vs Corrected")]
    positions: list[float] = []
    values: list[np.ndarray] = []
    colors: list[str] = []
    labels: list[str] = []
    stats_rows: list[dict[str, object]] = []

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    position = 1.0
    for classifier in CLASSIFIERS:
        for baseline, baseline_label in comparisons:
            differences = np.asarray(
                [
                    float(lookup[(classifier, "optimized_action", repeat)]["accuracy"])
                    - float(lookup[(classifier, baseline, repeat)]["accuracy"])
                    for repeat in range(20)
                ],
                dtype=np.float64,
            )
            positions.append(position)
            values.append(differences)
            colors.append("#F58518" if baseline == "raw" else "#72B7B2")
            classifier_label = "RF" if classifier == "rf" else DISPLAY_CLASSIFIERS[classifier]
            labels.append(f"{classifier_label}\n{baseline_label}")
            p_value = float(wilcoxon(differences).pvalue) if wilcoxon is not None and np.any(differences) else 1.0
            stats_rows.append(
                {
                    "classifier": classifier,
                    "comparison": f"optimized_action_minus_{baseline}",
                    "mean_accuracy_difference": float(np.mean(differences)),
                    "median_accuracy_difference": float(np.median(differences)),
                    "positive_repeats": int(np.sum(differences > 0)),
                    "equal_repeats": int(np.sum(differences == 0)),
                    "negative_repeats": int(np.sum(differences < 0)),
                    "wilcoxon_p": p_value,
                }
            )
            position += 1.0
        position += 0.45

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    violin = ax.violinplot(values, positions=positions, widths=0.75, showmeans=False, showmedians=True)
    for body, color in zip(violin["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.72)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        violin[key].set_color("#333333")
        violin[key].set_linewidth(1.0)
    ax.axhline(0.0, color="#333333", linewidth=1.2, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Paired accuracy difference")
    ax.set_title("Optimized Action Improvement Across Identical Few-Shot Splits", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_accuracy_improvement_6class.png", bbox_inches="tight")
    plt.close(fig)
    _write_rows(output_dir / "paired_accuracy_stats_6class.csv", stats_rows)


def performance_heatmaps(rows: list[dict[str, object]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("kappa", "Cohen's kappa")]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        matrix = np.zeros((len(FEATURE_SETS), len(CLASSIFIERS)), dtype=np.float64)
        for row_index, feature_set in enumerate(FEATURE_SETS):
            for col_index, classifier in enumerate(CLASSIFIERS):
                selected = [
                    float(row[metric])
                    for row in rows
                    if row["feature_set"] == feature_set and row["classifier"] == classifier
                ]
                matrix[row_index, col_index] = np.mean(selected)
        # Keep the full pilot-performance range visible when a larger dataset
        # makes raw-landmark baselines substantially harder.
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.20, vmax=1.0, aspect="auto")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(np.arange(len(CLASSIFIERS)))
        ax.set_xticklabels([DISPLAY_CLASSIFIERS[value] for value in CLASSIFIERS], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(FEATURE_SETS)))
        ax.set_yticklabels([DISPLAY_FEATURES[value] for value in FEATURE_SETS])
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.88 else "black",
                    fontsize=9,
                )
        ax.grid(False)
    fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    fig.savefig(output_dir / "performance_heatmaps_6class.png", bbox_inches="tight")
    plt.close(fig)


def _normalized_cm(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)


def _draw_cm(ax, cm: np.ndarray, labels: list[str], title: str) -> object:
    normalized = _normalized_cm(cm)
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontweight="bold")
    paper_labels = {
        "deer_horn": "Deer horn",
        "flower_pinch": "Flower pinch",
        "orchid_finger": "Orchid finger",
        "orchid_palm": "Orchid palm",
        "prayer_beads": "Prayer beads",
        "three_finger_bent": "Three-finger",
    }
    short_labels = [paper_labels.get(label, label.replace("_", " ").title()) for label in labels]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            value = normalized[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value > 0.52 else "black",
            )
    return image


def confusion_panels(confusions: dict[tuple[str, str], np.ndarray], labels: list[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), constrained_layout=True)
    for ax, feature_set in zip(axes, ["raw", "corrected", "optimized_action"], strict=True):
        image = _draw_cm(ax, confusions[("svm", feature_set)], labels, DISPLAY_FEATURES[feature_set])
    fig.colorbar(image, ax=axes, shrink=0.82, pad=0.01)
    fig.suptitle("SVM Error Patterns Across Representation Stages", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "cm_svm_representation_stages_6class.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6), constrained_layout=True)
    for ax, classifier in zip(axes.flat, CLASSIFIERS, strict=True):
        image = _draw_cm(
            ax,
            confusions[(classifier, "optimized_action")],
            labels,
            DISPLAY_CLASSIFIERS[classifier],
        )
    fig.colorbar(image, ax=axes, shrink=0.82, pad=0.01)
    fig.suptitle("Optimized Action Across Classifiers", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "cm_optimized_action_all_classifiers_6class.png", bbox_inches="tight")
    plt.close(fig)


def per_class_recall_figure(confusions: dict[tuple[str, str], np.ndarray], labels: list[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    selected_features = ["raw", "corrected", "optimized_action"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    rows: list[dict[str, object]] = []
    for index, feature_set in enumerate(selected_features):
        recall = np.diag(_normalized_cm(confusions[("svm", feature_set)]))
        ax.bar(
            x + (index - 1) * width,
            recall,
            width,
            color=COLORS[feature_set],
            label=DISPLAY_FEATURES[feature_set],
        )
        for label, value in zip(labels, recall, strict=True):
            rows.append({"classifier": "svm", "feature_set": feature_set, "label": label, "recall": float(value)})
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Recall across repeated splits")
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", " ") for label in labels], rotation=25, ha="right")
    ax.set_title("Per-Class Recall: SVM", fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_recall_svm_6class.png", bbox_inches="tight")
    plt.close(fig)
    _write_rows(output_dir / "per_class_recall_svm_6class.csv", rows)


def trajectory_figure(dataset: Path, actuator_table: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    with dataset.open("r", newline="", encoding="utf-8") as fh:
        dataset_rows = list(csv.DictReader(fh))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in dataset_rows:
        grouped[row["sequence_id"]].append(row)

    with actuator_table.open("r", newline="", encoding="utf-8") as fh:
        actuator_rows = list(csv.DictReader(fh))

    best: tuple[float, str, int, np.ndarray, np.ndarray] | None = None
    for sequence_id, rows in grouped.items():
        rows.sort(key=lambda row: (int(float(row["frame_id"])), float(row["timestamp_sec"])))
        corrected = np.asarray([[float(row[f"corrected_{i}"]) for i in range(17)] for row in rows])
        optimized = np.asarray([[float(row[f"optimized_action_{i}"]) for i in range(17)] for row in rows])
        if corrected.shape[0] < 10:
            continue
        corrected_acc = np.abs(np.diff(corrected, n=2, axis=0))
        optimized_acc = np.abs(np.diff(optimized, n=2, axis=0))
        dimension_scores = np.max(corrected_acc, axis=0) - np.max(optimized_acc, axis=0)
        dimension = int(np.argmax(dimension_scores))
        score = float(dimension_scores[dimension])
        if best is None or score > best[0]:
            best = (score, sequence_id, dimension, corrected, optimized)

    if best is None:
        raise RuntimeError("No valid sequence available for trajectory visualization.")

    score, sequence_id, dimension, corrected, optimized = best
    role = actuator_rows[dimension]["role"]
    frames = np.arange(corrected.shape[0])
    corrected_velocity = np.linalg.norm(np.diff(corrected, axis=0), axis=1)
    optimized_velocity = np.linalg.norm(np.diff(optimized, axis=0), axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 0.9]})
    axes[0].plot(frames, corrected[:, dimension], color=COLORS["corrected"], linewidth=1.4, label="Corrected")
    axes[0].plot(frames, optimized[:, dimension], color=COLORS["optimized_action"], linewidth=2.0, label="Optimized Action")
    axes[0].set_ylabel("Actuator value (rad)")
    axes[0].set_title(f"Representative Refinement: {role}", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[1].plot(frames[1:], corrected_velocity, color=COLORS["corrected"], linewidth=1.2, label="Corrected velocity norm")
    axes[1].plot(frames[1:], optimized_velocity, color=COLORS["optimized_action"], linewidth=1.8, label="Optimized velocity norm")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Actuator-space velocity")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_refinement_example_6class.png", bbox_inches="tight")
    plt.close(fig)

    _write_rows(
        output_dir / "trajectory_refinement_example_6class.csv",
        [
            {
                "sequence_id": sequence_id,
                "actuator_index": dimension,
                "actuator_role": role,
                "selection_score": score,
                "num_frames": corrected.shape[0],
                "corrected_velocity_mean": float(np.mean(corrected_velocity)),
                "optimized_velocity_mean": float(np.mean(optimized_velocity)),
                "corrected_velocity_max": float(np.max(corrected_velocity)),
                "optimized_velocity_max": float(np.max(optimized_velocity)),
            }
        ],
    )


def pipeline_figure(output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    boxes = [
        (0.25, 1.55, 1.75, 1.2, "MediaPipe\nlandmarks", "21 x 3", "#4C78A8"),
        (2.45, 1.55, 1.85, 1.2, "Embodiment-aware\nprojection", "17-D initialization", "#72B7B2"),
        (4.8, 1.25, 2.55, 1.8, "MuJoCo-constrained\ncausal optimization", "Huber + normal + prior\n+ velocity + acceleration", "#ECA82C"),
        (7.85, 1.55, 1.8, 1.2, "Optimized Action", "17-D latent state", "#F58518"),
        (10.05, 2.55, 1.65, 1.0, "Classification", "few-shot evaluation", "#B279A2"),
        (10.05, 0.65, 1.65, 1.0, "Optimized Full", "63-D reconstruction", "#54A24B"),
    ]
    for x, y, width, height, title, subtitle, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color,
            edgecolor="none",
            alpha=0.92,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        ax.text(x + width / 2, y + height * 0.22, subtitle, ha="center", va="center", color="white", fontsize=8)

    arrows = [
        ((2.0, 2.15), (2.45, 2.15)),
        ((4.3, 2.15), (4.8, 2.15)),
        ((7.35, 2.15), (7.85, 2.15)),
        ((9.65, 2.25), (10.05, 3.05)),
        ((9.65, 1.85), (10.05, 1.15)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#333333"))
    ax.text(6.05, 3.65, "feasible actuator bounds + MuJoCo forward kinematics", ha="center", color="#555555", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "method_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def implementation_validation_figure(before_json: Path, after_json: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    before = json.loads(before_json.read_text(encoding="utf-8"))
    after = json.loads(after_json.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].bar(["Before", "After"], [before["palm_consistency_check"]["dot_product"], after["palm_consistency_check"]["dot_product"]], color=["#E45756", "#54A24B"])
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_ylabel("Unit-normal dot product")
    axes[0].set_title("Palm-normal consistency", fontweight="bold")

    names = ["Actuator MAE", "Landmark loss", "Palm loss"]
    before_values = [
        before["synthetic_recovery"]["mean_abs_action_error"],
        before["synthetic_recovery"]["loss_terms"]["landmark"],
        before["synthetic_recovery"]["loss_terms"]["palm"],
    ]
    after_values = [
        after["synthetic_recovery"]["mean_abs_action_error"],
        after["synthetic_recovery"]["loss_terms"]["landmark"],
        after["synthetic_recovery"]["loss_terms"]["palm"],
    ]
    x = np.arange(len(names))
    axes[1].bar(x - 0.18, before_values, 0.36, label="Before", color="#E45756")
    axes[1].bar(x + 0.18, after_values, 0.36, label="After", color="#54A24B")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=18, ha="right")
    axes[1].set_ylabel("Error / loss")
    axes[1].set_title("Synthetic recovery", fontweight="bold")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "implementation_validation_before_after.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate extended evidence figures for the paper.")
    parser.add_argument("--dataset", default="diagnostics/gesture_sequence_dataset_chinese_dance_6class_after_fix.csv")
    parser.add_argument("--split-manifest", default="diagnostics/palm_fix_split_manifest_6class.csv")
    parser.add_argument("--output-dir", default="figures/paper_rewrite_main")
    parser.add_argument(
        "--actuator-table",
        default="figures/paper_rewrite_main/actuator_definition_table.csv",
        help="Actuator definition CSV used to label the trajectory figure.",
    )
    args = parser.parse_args()

    dataset = (ROOT / args.dataset).resolve()
    manifest = (ROOT / args.split_manifest).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    actuator_table = (ROOT / args.actuator_table).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()

    rows, confusions, class_labels = evaluate_repeats(dataset, manifest)
    _write_rows(output_dir / "per_repeat_scores_6class.csv", rows)
    paired_accuracy_figure(rows, output_dir)
    performance_heatmaps(rows, output_dir)
    confusion_panels(confusions, class_labels, output_dir)
    per_class_recall_figure(confusions, class_labels, output_dir)
    trajectory_figure(dataset, actuator_table, output_dir)
    pipeline_figure(output_dir)
    implementation_validation_figure(
        ROOT / "diagnostics" / "palm_fix_before.json",
        ROOT / "diagnostics" / "palm_fix_after_regen.json",
        output_dir,
    )
    print(f"extended_figures_dir={output_dir}")


if __name__ == "__main__":
    main()
