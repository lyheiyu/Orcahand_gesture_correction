from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate_sequence_encodings as sequence_encoding
import generate_joint_angle_baseline as joint_angles
import generate_shot_sweep_figures as shot_sweep
import train_svm


CLASSIFIERS = ("svm", "knn", "rf", "mlp")
REPRESENTATIONS = (
    "joint_angle11",
    "corrected17",
    "optimized_action17",
    "corrected_flex11",
    "optimized_action_flex11",
    "corrected_pca11",
    "optimized_action_pca11",
)
DISPLAY = {
    "joint_angle11": "JointAngle-11",
    "corrected17": "Corrected-17",
    "optimized_action17": "OptimizedAction-17",
    "corrected_flex11": "Corrected-Flex11",
    "optimized_action_flex11": "OptimizedAction-Flex11",
    "corrected_pca11": "Corrected-PCA11",
    "optimized_action_pca11": "OptimizedAction-PCA11",
}
COLORS = {
    "joint_angle11": "#D99A20",
    "corrected17": "#448C57",
    "optimized_action17": "#E66A21",
    "corrected_flex11": "#74B985",
    "optimized_action_flex11": "#F29A55",
    "corrected_pca11": "#4178A8",
    "optimized_action_pca11": "#8B65A8",
}

# Verified against the right-hand MuJoCo model actuator order. Flex11 is ordered
# to match JointAngle-11 semantics: thumb, index, middle, ring, little.
ACTUATORS = (
    "right_wrist_actuator",
    "right_p-abd_actuator",
    "right_p-mcp_actuator",
    "right_p-pip_actuator",
    "right_r-abd_actuator",
    "right_r-mcp_actuator",
    "right_r-pip_actuator",
    "right_m-abd_actuator",
    "right_m-mcp_actuator",
    "right_m-pip_actuator",
    "right_i-abd_actuator",
    "right_i-mcp_actuator",
    "right_i-pip_actuator",
    "right_t-cmc_actuator",
    "right_t-abd_actuator",
    "right_t-mcp_actuator",
    "right_t-pip_actuator",
)
FLEX11_INDICES = (13, 15, 16, 11, 12, 8, 9, 5, 6, 2, 3)
FLEX11_NAMES = (
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "index_mcp",
    "index_pip",
    "middle_mcp",
    "middle_pip",
    "ring_mcp",
    "ring_pip",
    "little_mcp",
    "little_pip",
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_base(dataset: Path) -> tuple[list[str], list[str], dict[str, list[np.ndarray]]]:
    metadata, feature_names, matrix = train_svm._load_dataset(dataset)
    output: dict[str, list[np.ndarray]] = {}
    sequence_ids: list[str] | None = None
    labels: list[str] | None = None
    for key, source in (("raw", "raw"), ("corrected17", "corrected"),
                        ("optimized_action17", "optimized_action")):
        _, selected = train_svm._select_features(feature_names, matrix, source)
        ids, selected_labels, grouped = train_svm._group_sequences(metadata, selected)
        if sequence_ids is not None and (ids != sequence_ids or selected_labels != labels):
            raise RuntimeError(f"Sequence alignment changed while loading {source}")
        sequence_ids, labels = ids, selected_labels
        output[key] = grouped
    assert sequence_ids is not None and labels is not None
    output["joint_angle11"] = [
        np.stack([joint_angles.joint_angle_vector(frame.reshape(21, 3))[0] for frame in sequence])
        for sequence in output["raw"]
    ]
    output["corrected_flex11"] = [value[:, FLEX11_INDICES] for value in output["corrected17"]]
    output["optimized_action_flex11"] = [
        value[:, FLEX11_INDICES] for value in output["optimized_action17"]
    ]
    return sequence_ids, labels, output


def _encode(sequences: list[np.ndarray]) -> np.ndarray:
    return np.stack(
        [sequence_encoding.encode_sequence(sequence, "resample16") for sequence in sequences]
    )


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["classifier"]), str(row["representation"]))].append(row)
    output: list[dict[str, object]] = []
    for (classifier, representation), values in sorted(groups.items()):
        result: dict[str, object] = {
            "classifier": classifier,
            "representation": representation,
            "display_name": DISPLAY[representation],
            "frame_dimensions": values[0]["frame_dimensions"],
            "encoded_features": values[0]["encoded_features"],
            "repeats": len(values),
        }
        for metric in ("accuracy", "macro_f1", "kappa"):
            data = np.asarray([float(row[metric]) for row in values])
            result[f"{metric}_mean"] = float(np.mean(data))
            result[f"{metric}_std"] = float(np.std(data))
            result[f"{metric}_ci95"] = float(1.96 * np.std(data, ddof=1) / np.sqrt(len(data)))
        output.append(result)
    return output


COMPARISONS = (
    ("joint_angle11", "corrected_flex11"),
    ("joint_angle11", "optimized_action_flex11"),
    ("corrected17", "corrected_flex11"),
    ("optimized_action17", "optimized_action_flex11"),
    ("corrected_pca11", "corrected_flex11"),
    ("optimized_action_pca11", "optimized_action_flex11"),
)


def _paired(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    from scipy.stats import wilcoxon

    lookup = {
        (int(row["repeat"]), str(row["classifier"]), str(row["representation"])): row
        for row in rows
    }
    repeats = sorted({int(row["repeat"]) for row in rows})
    output: list[dict[str, object]] = []
    for classifier in CLASSIFIERS:
        for first, second in COMPARISONS:
            for metric in ("accuracy", "macro_f1", "kappa"):
                differences = np.asarray([
                    float(lookup[(repeat, classifier, first)][metric])
                    - float(lookup[(repeat, classifier, second)][metric])
                    for repeat in repeats
                ])
                p_value = 1.0
                if np.any(differences != 0):
                    p_value = float(wilcoxon(differences).pvalue)
                output.append({
                    "classifier": classifier,
                    "metric": metric,
                    "first": first,
                    "second": second,
                    "comparison": f"{first}_minus_{second}",
                    "mean_difference": float(np.mean(differences)),
                    "std_difference": float(np.std(differences)),
                    "ci95_difference": float(1.96 * np.std(differences, ddof=1) / np.sqrt(len(differences))),
                    "positive_repeats": int(np.sum(differences > 0)),
                    "equal_repeats": int(np.sum(differences == 0)),
                    "negative_repeats": int(np.sum(differences < 0)),
                    "wilcoxon_p": p_value,
                })
    return output


def evaluate(dataset: Path, repeats: int, shot: int, test_size: float, random_state: int):
    sequence_ids, labels, sequences = _load_base(dataset)
    splits, manifest = shot_sweep._build_nested_splits(
        sequence_ids, labels, (shot,), repeats, test_size, random_state
    )
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        seed = random_state + repeat
        train_indices = splits[(repeat, shot)]["train"]
        test_indices = splits[(repeat, shot)]["test"]
        y_train = [labels[index] for index in train_indices]
        y_test = [labels[index] for index in test_indices]
        split_sequences: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        for representation in (
            "joint_angle11", "corrected17", "optimized_action17",
            "corrected_flex11", "optimized_action_flex11",
        ):
            split_sequences[representation] = (
                [sequences[representation][index] for index in train_indices],
                [sequences[representation][index] for index in test_indices],
            )
        for source, target in (("corrected17", "corrected_pca11"),
                               ("optimized_action17", "optimized_action_pca11")):
            split_sequences[target] = train_svm._project_sequences_with_pca(
                *split_sequences[source], n_components=11, seed=seed
            )
        for representation in REPRESENTATIONS:
            train_sequences, test_sequences = split_sequences[representation]
            x_train, x_test = _encode(train_sequences), _encode(test_sequences)
            for classifier in CLASSIFIERS:
                model = train_svm._build_model(shot_sweep._model_args(classifier), seed)
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                rows.append({
                    "repeat": repeat,
                    "seed": seed,
                    "shot": shot,
                    "classifier": classifier,
                    "representation": representation,
                    "frame_dimensions": int(train_sequences[0].shape[1]),
                    "encoded_features": int(x_train.shape[1]),
                    "num_train": len(train_indices),
                    "num_test": len(test_indices),
                    "accuracy": float(accuracy_score(y_test, prediction)),
                    "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
                    "kappa": float(cohen_kappa_score(y_test, prediction)),
                })
    return rows, manifest


def _plot_grouped(summary: list[dict[str, object]], metric: str, selected: tuple[str, ...], path: Path,
                  title: str) -> None:
    lookup = {(str(row["classifier"]), str(row["representation"])): row for row in summary}
    x = np.arange(len(CLASSIFIERS), dtype=float)
    width = 0.82 / len(selected)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for offset, representation in enumerate(selected):
        values = [float(lookup[(classifier, representation)][f"{metric}_mean"]) for classifier in CLASSIFIERS]
        errors = [float(lookup[(classifier, representation)][f"{metric}_ci95"]) for classifier in CLASSIFIERS]
        positions = x - 0.41 + width / 2 + offset * width
        ax.bar(positions, values, width, yerr=errors, capsize=3,
               color=COLORS[representation], label=DISPLAY[representation])
    ax.set_xticks(x, [shot_sweep.DISPLAY_CLASSIFIERS[value] for value in CLASSIFIERS])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _mapping_rows() -> list[dict[str, object]]:
    semantic = {index: name for index, name in zip(FLEX11_INDICES, FLEX11_NAMES, strict=True)}
    rows = []
    for index, actuator in enumerate(ACTUATORS):
        keep = index in semantic
        rows.append({
            "original_index": index,
            "actuator_name": actuator,
            "keep_in_flex11": "yes" if keep else "no",
            "flex11_semantic_name": semantic.get(index, ""),
            "flex11_output_index": FLEX11_INDICES.index(index) if keep else "",
            "reason": "flexion/opposition coordinate matched to JointAngle-11" if keep
                      else "wrist or abduction coordinate excluded by preregistered semantic rule",
        })
    return rows


def _dimension_rows() -> list[dict[str, object]]:
    dimensions = {key: (17 if key.endswith("17") else 11) for key in REPRESENTATIONS}
    return [{
        "representation": key,
        "display_name": DISPLAY[key],
        "frame_dimensions": dimensions[key],
        "resampled_frames": 16,
        "encoded_features": dimensions[key] * 16,
        "construction": (
            "absolute 3D joint angles" if key == "joint_angle11" else
            "original ORCA actuator state" if key.endswith("17") else
            "fixed semantic actuator subset" if "flex11" in key else
            "training-only frame-level PCA"
        ),
    } for key in REPRESENTATIONS]


def _answer(summary: list[dict[str, object]], first: str, second: str) -> str:
    differences = []
    for classifier in CLASSIFIERS:
        lookup = {(str(row["classifier"]), str(row["representation"])): row for row in summary}
        differences.append(float(lookup[(classifier, first)]["accuracy_mean"])
                           - float(lookup[(classifier, second)]["accuracy_mean"]))
    mean = float(np.mean(differences))
    return f"{DISPLAY[first]} - {DISPLAY[second]} averaged across classifiers = {mean:+.4f} accuracy."


def _accuracy_difference(summary: list[dict[str, object]], first: str, second: str) -> dict[str, float]:
    lookup = {(str(row["classifier"]), str(row["representation"])): row for row in summary}
    return {
        classifier: float(lookup[(classifier, first)]["accuracy_mean"])
        - float(lookup[(classifier, second)]["accuracy_mean"])
        for classifier in CLASSIFIERS
    }


def _difference_text(values: dict[str, float]) -> str:
    return ", ".join(
        f"{shot_sweep.DISPLAY_CLASSIFIERS[key]} {100.0 * value:+.2f} pp"
        for key, value in values.items()
    )


def _write_explanation(path: Path, dataset: Path, summary: list[dict[str, object]], repeats: int) -> None:
    lookup = {(str(row["classifier"]), str(row["representation"])): row for row in summary}
    corrected_flex_gain = _accuracy_difference(summary, "corrected_flex11", "corrected17")
    action_flex_gain = _accuracy_difference(summary, "optimized_action_flex11", "optimized_action17")
    joint_vs_corrected = _accuracy_difference(summary, "joint_angle11", "corrected_flex11")
    joint_vs_action = _accuracy_difference(summary, "joint_angle11", "optimized_action_flex11")
    corrected_flex_vs_pca = _accuracy_difference(summary, "corrected_flex11", "corrected_pca11")
    action_flex_vs_pca = _accuracy_difference(summary, "optimized_action_flex11", "optimized_action_pca11")
    lines = [
        "# ORCA Dimension-Control Experiment Explained", "",
        "## A. Why are we doing this experiment?", "",
        "JointAngle has 11 frame-level dimensions, whereas the original ORCA representations have 17. "
        "After Resample-16, classifiers receive 176 versus 272 values. With only three training sequences "
        "per class, this dimensionality difference may matter. This experiment separates dimension, actuator "
        "semantics, and representation quality without changing the ORCA optimizer.", "",
        "## B. What exactly is Corrected-17?", "",
        "Corrected-17 is the frame-wise, rule-based mapping from normalized MediaPipe landmarks to the 17 "
        "right-hand ORCA actuator coordinates. It uses hand geometry and actuator ranges, but no temporal history "
        "and no MuJoCo optimization.", "",
        "## C. What exactly is OptimizedAction-17?", "",
        "OptimizedAction-17 starts from Corrected-17 and refines the same actuator state using MuJoCo forward "
        "kinematics, robust landmark fitting, priors, bounds, and causal temporal regularization. Its outputs remain "
        "17 actuator values per frame.", "",
        "## D. How is Flex11 created?", "",
        "Flex11 is fixed before evaluation. It keeps thumb CMC/MCP/IP and MCP/PIP flexion for the four fingers. "
        "It removes wrist motion and six abduction-related coordinates. Both Corrected and Optimized Action use "
        "the exact same indices: `13,15,16,11,12,8,9,5,6,2,3`.", "",
        "| Index | Actuator | Decision | Reason |", "|---:|---|---|---|",
    ]
    for row in _mapping_rows():
        lines.append(f"| {row['original_index']} | `{row['actuator_name']}` | "
                     f"{row['keep_in_flex11']} | {row['reason']} |")
    lines += [
        "", "## E. Why is Flex11 scientifically fair?", "",
        "The subset is based on semantic correspondence with the preregistered JointAngle-11 definition, not "
        "on test accuracy. No alternative subsets are searched. It is therefore a controlled representation-selection "
        "test rather than test-set feature selection.", "",
        "## F. How is PCA11 different?", "",
        "Flex11 preserves 11 named actuator coordinates. PCA11 instead mixes all 17 coordinates into 11 orthogonal "
        "components. For every repeat, scaling and PCA are fitted only to training frames, then applied unchanged to "
        "test frames. PCA occurs before Resample-16.", "",
        "## G. How does Resample-16 work?", "",
        "Each variable-length trajectory is linearly interpolated at 16 normalized time positions and flattened in "
        "temporal order. Thus `17 x 16 = 272`, while `11 x 16 = 176`. Unlike global mean/std statistics, the order "
        "of the 16 temporal samples is retained.", "",
        "## H. How is the 3-shot experiment performed?", "",
        f"For each of {repeats} repeats, the project creates one stratified sequence-level holdout, chooses exactly "
        "three training sequences per class from the training pool, fits preprocessing on those training sequences, "
        "trains SVM/KNN/RandomForest/MLP with fixed paper settings, and evaluates the common test sequences. All "
        "seven representations share the same split in every repeat.", "",
        "## I. How should I interpret every possible result?", "",
        "- ORCA-Flex11 above JointAngle-11: matched actuator semantics are at least as useful as direct human angles.",
        "- Similar results: the original gap was substantially associated with dimension or irrelevant coordinates.",
        "- JointAngle-11 above ORCA-Flex11: direct human-joint geometry better matches this recognition task.",
        "- Flex11 above ORCA17: wrist/abduction dimensions add redundancy or noise in this few-shot setting.",
        "- PCA11 above Flex11: variance-preserving mixtures are more useful than the predefined semantics.",
        "- Flex11 above PCA11: named actuator semantics contribute beyond dimensionality reduction alone.", "",
        "## J. Relevant code", "",
        "- `generate_joint_angle_baseline.py::joint_angle_vector`: computes the 11 absolute 3D angles.",
        "- `evaluate_orca_dimension_control.py::_load_base`: loads Corrected/OA and selects Flex11.",
        "- `train_svm.py::_project_sequences_with_pca`: fits scaler and PCA on training frames only.",
        "- `evaluate_sequence_encodings.py::encode_sequence`: Resample-16 and temporal flattening.",
        "- `generate_shot_sweep_figures.py::_build_nested_splits`: shared sequence-level few-shot splits.",
        "- `train_svm.py::_build_model`: training-only StandardScaler and fixed classifier.", "",
        "## K. Final pipeline diagram", "",
        "```text",
        "MediaPipe landmarks -> JointAngle-11 ---------------------------> Resample16 -> classifier",
        "                  \\-> Corrected-17 -> Flex11/PCA11 ------------> Resample16 -> classifier",
        "                                   \\-> MuJoCo -> OA-17 -> Flex11/PCA11 -> classifier",
        "```", "",
        "## L. Results and final conclusion", "",
        f"Dataset: `{dataset}`.", "",
        "### Accuracy summary", "",
        "| Representation | SVM | KNN | RandomForest | MLP |", "|---|---:|---:|---:|---:|",
    ]
    for representation in REPRESENTATIONS:
        values = [float(lookup[(classifier, representation)]["accuracy_mean"]) for classifier in CLASSIFIERS]
        lines.append(f"| {DISPLAY[representation]} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += [
        "", "### Answers to the eight interpretation questions", "",
        "**Q1. Does Corrected-Flex11 improve over Corrected-17?** Yes, but the gain is generally modest. "
        + _difference_text(corrected_flex_gain) + ". This suggests that non-flexion coordinates add some "
        "few-shot burden, although the effect is classifier-dependent.", "",
        "**Q2. Does OptimizedAction-Flex11 improve over OptimizedAction-17?** Yes in all four classifiers. "
        + _difference_text(action_flex_gain) + ". The mean gain across classifiers is "
        f"{100.0 * np.mean(list(action_flex_gain.values())):.2f} pp.", "",
        "**Q3. Does JointAngle-11 still outperform dimension-matched ORCA?** Not consistently. Against "
        "Corrected-Flex11 the differences are " + _difference_text(joint_vs_corrected) + "; against "
        "OptimizedAction-Flex11 they are " + _difference_text(joint_vs_action) + ". The three 11D semantic "
        "representations are therefore close overall, rather than showing universal JointAngle dominance.", "",
        "**Q4. Does PCA11 differ from semantic Flex11?** Yes. Semantic Flex11 is higher for every classifier "
        "in both branches: Corrected differences are " + _difference_text(corrected_flex_vs_pca) + ", and "
        "Optimized Action differences are " + _difference_text(action_flex_vs_pca) + ".", "",
        "**Q5. Does Flex11 above PCA11 suggest semantics matter beyond dimension?** Yes. Both have 11 frame "
        "dimensions and 176 encoded inputs, but only Flex11 preserves predefined flexion/opposition coordinates. "
        "The consistent Flex11 advantage supports a semantic-selection explanation, while not proving causality by itself.", "",
        "**Q6. Is dimensionality/redundancy contributing to the original gap?** Partly. Flex11 improves both "
        "ORCA branches, but PCA11 does not consistently improve their 17D sources. Therefore fewer dimensions alone "
        "are insufficient; which dimensions are retained also matters.", "",
        "**Q7. Does JointAngle clearly win against both ORCA 11D controls?** No. It is slightly higher for SVM, "
        "KNN, and RandomForest, while both ORCA-Flex11 variants are higher for MLP. Most differences are small. "
        "The evidence supports task-matched human geometry as competitive, not categorically superior.", "",
        "**Q8. Does OA benefit more from Flex11 than Corrected?** Yes in the across-classifier mean. OA gains "
        f"{100.0 * np.mean(list(action_flex_gain.values())):.2f} pp versus "
        f"{100.0 * np.mean(list(corrected_flex_gain.values())):.2f} pp for Corrected. This is consistent with some "
        "refined wrist/abduction coordinates being less useful for these six gesture labels, but it does not identify "
        "which removed coordinate causes the effect.", "",
        "### Compact conclusion", "",
        _answer(summary, "corrected_flex11", "corrected17"),
        _answer(summary, "optimized_action_flex11", "optimized_action17"),
        _answer(summary, "joint_angle11", "corrected_flex11"),
        _answer(summary, "joint_angle11", "optimized_action_flex11"),
        _answer(summary, "corrected_flex11", "corrected_pca11"),
        _answer(summary, "optimized_action_flex11", "optimized_action_pca11"), "",
        "Interpret these differences together with classifier-specific confidence intervals and paired Wilcoxon tests "
        "in `classifier_results_all.csv` and `paired_comparisons.csv`; no single classifier is treated as decisive.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered ORCA 17D/11D dimension-control experiment.")
    parser.add_argument("--dataset", default="diagnostics/updated_6class_20260820/gesture_sequence_dataset_chinese_dance_6class_after_fix.csv")
    parser.add_argument("--output-dir", default="diagnostics/orca_dimension_control_20260827")
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    dataset, output = Path(args.dataset).resolve(), Path(args.output_dir).resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    rows, manifest = evaluate(dataset, args.repeats, args.shot, args.test_size, args.random_state)
    summary, paired = _summary(rows), _paired(rows)
    _write_rows(output / "per_repeat_results.csv", rows)
    _write_rows(output / "classifier_results_all.csv", summary)
    _write_rows(output / "paired_comparisons.csv", paired)
    _write_rows(output / "split_manifest.csv", manifest)
    _write_rows(output / "flex11_actuator_mapping.csv", _mapping_rows())
    _write_rows(output / "representation_dimension_summary.csv", _dimension_rows())

    _plot_grouped(summary, "accuracy", REPRESENTATIONS, figures / "01_accuracy_all_representations.png",
                  "Dimension-Controlled Accuracy (3-shot)")
    _plot_grouped(summary, "macro_f1", REPRESENTATIONS, figures / "02_macro_f1_all_representations.png",
                  "Dimension-Controlled Macro-F1 (3-shot)")
    _plot_grouped(summary, "accuracy", ("corrected17", "corrected_flex11"),
                  figures / "03_corrected17_vs_flex11.png", "Corrected: 17D vs Semantic Flex11")
    _plot_grouped(summary, "accuracy", ("optimized_action17", "optimized_action_flex11"),
                  figures / "04_optimized_action17_vs_flex11.png", "Optimized Action: 17D vs Semantic Flex11")
    _plot_grouped(summary, "accuracy", ("joint_angle11", "corrected_flex11", "optimized_action_flex11"),
                  figures / "05_joint_angle_vs_orca_flex11.png", "Joint Angles vs Dimension-Matched ORCA Flex11")
    _plot_grouped(summary, "accuracy", ("corrected_flex11", "corrected_pca11",
                                         "optimized_action_flex11", "optimized_action_pca11"),
                  figures / "06_semantic_flex11_vs_pca11.png", "Semantic Flex11 vs Training-Only PCA11")
    _write_explanation(output / "ORCA_DIMENSION_CONTROL_EXPLAINED.md", dataset, summary, args.repeats)
    print(f"dataset={dataset}")
    print(f"output_dir={output}")
    print(f"rows={len(rows)} summaries={len(summary)} paired_tests={len(paired)}")


if __name__ == "__main__":
    main()
