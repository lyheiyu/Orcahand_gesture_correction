from __future__ import annotations

import argparse
import csv
import hashlib
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate_orca_dimension_control as dimension_control
import evaluate_sequence_encodings as sequence_encoding
import generate_shot_sweep_figures as shot_sweep
import train_svm


CANDIDATE_K = (5, 7, 9, 11, 13)
FROZEN_K = 7
FROZEN_INDICES = (3, 6, 9, 11, 12, 15, 16)
FROZEN_DEVELOPMENT_SHA256 = "b9d3769a00107f29941dc60b58baacec39e113cd30d1d1c1d775546f1c503b17"
FROZEN_FINAL_SHA256 = "dc995a69e178393bbf1cc056d084185d935b1fcee4248e0cea17cc314711a33f"
UTILITY_WEIGHTS = {
    "within_class": 0.25,
    "instability": 0.20,
    "redundancy": 0.15,
    "refinement_sensitivity": 0.15,
}
ACTUATOR_META = (
    ("wrist", "wrist", "wrist"),
    ("little abduction", "little", "abduction"),
    ("little MCP flexion", "little", "flexion"),
    ("little PIP flexion", "little", "flexion"),
    ("ring abduction", "ring", "abduction"),
    ("ring MCP flexion", "ring", "flexion"),
    ("ring PIP flexion", "ring", "flexion"),
    ("middle abduction", "middle", "abduction"),
    ("middle MCP flexion", "middle", "flexion"),
    ("middle PIP flexion", "middle", "flexion"),
    ("index abduction", "index", "abduction"),
    ("index MCP flexion", "index", "flexion"),
    ("index PIP flexion", "index", "flexion"),
    ("thumb opposition / CMC", "thumb", "opposition"),
    ("thumb abduction", "thumb", "abduction"),
    ("thumb MCP flexion", "thumb", "flexion"),
    ("thumb IP flexion", "thumb", "flexion"),
)
FINGER_GROUPS = ("thumb", "index", "middle", "ring", "little")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def freeze_outer_split(
    sequence_ids: list[str], labels: list[str], output: Path, test_size: float, seed: int
) -> tuple[list[int], list[int]]:
    development_path = output / "development_sequences.csv"
    final_path = output / "final_test_sequences.csv"
    id_to_index = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}
    if development_path.exists() or final_path.exists():
        if not (development_path.exists() and final_path.exists()):
            raise RuntimeError("Only one frozen manifest exists; refusing to recreate the split.")
        development = [id_to_index[row["sequence_id"]] for row in _read_manifest(development_path)]
        final = [id_to_index[row["sequence_id"]] for row in _read_manifest(final_path)]
    else:
        development, final = train_test_split(
            np.arange(len(sequence_ids)), test_size=test_size, random_state=seed, stratify=labels
        )
        development, final = sorted(map(int, development)), sorted(map(int, final))
        _write_rows(development_path, [
            {"sequence_id": sequence_ids[index], "label": labels[index], "outer_split_seed": seed}
            for index in development
        ])
        _write_rows(final_path, [
            {"sequence_id": sequence_ids[index], "label": labels[index], "outer_split_seed": seed}
            for index in final
        ])
    if set(development) & set(final) or set(development) | set(final) != set(range(len(sequence_ids))):
        raise RuntimeError("Frozen outer split is overlapping or incomplete.")
    return development, final


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(np.min(values)), float(np.max(values))
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _base_components(sequences: list[np.ndarray], labels: list[str]) -> dict[str, np.ndarray]:
    frames = np.concatenate(sequences, axis=0).astype(np.float64)
    mean = np.mean(frames, axis=0)
    scale = np.std(frames, axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = [(sequence.astype(np.float64) - mean) / scale for sequence in sequences]
    sequence_means = np.stack([np.mean(sequence, axis=0) for sequence in normalized])
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f_score, _ = f_classif(sequence_means, labels)
    f_score = np.nan_to_num(f_score, nan=0.0, posinf=np.finfo(np.float32).max)

    within = np.zeros(17, dtype=np.float64)
    for label in sorted(set(labels)):
        class_values = sequence_means[np.asarray(labels) == label]
        within += np.var(class_values, axis=0)
    within /= len(set(labels))

    velocity, acceleration = [], []
    for sequence in normalized:
        velocity.append(np.sqrt(np.mean(np.diff(sequence, axis=0) ** 2, axis=0)))
        if len(sequence) >= 3:
            second = sequence[2:] - 2.0 * sequence[1:-1] + sequence[:-2]
            acceleration.append(np.sqrt(np.mean(second ** 2, axis=0)))
    instability = np.mean(velocity, axis=0) + np.mean(acceleration, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.nan_to_num(np.corrcoef(sequence_means, rowvar=False), nan=0.0)
    absolute = np.abs(correlation)
    np.fill_diagonal(absolute, 0.0)
    redundancy = np.max(absolute, axis=1)
    return {
        "f_score_raw": f_score,
        "discriminative": _minmax(np.log1p(f_score)),
        "within_raw": within,
        "within": _minmax(within),
        "instability_raw": instability,
        "instability": _minmax(instability),
        "redundancy": redundancy,
        "correlation": correlation,
        "constant": np.std(sequence_means, axis=0) <= 1e-12,
    }


def actuator_scores(
    corrected: list[np.ndarray], optimized: list[np.ndarray], labels: list[str]
) -> tuple[list[dict[str, object]], np.ndarray]:
    corrected_parts = _base_components(corrected, labels)
    optimized_parts = _base_components(optimized, labels)
    sensitivity_raw = (
        np.maximum(0.0, corrected_parts["discriminative"] - optimized_parts["discriminative"])
        + np.maximum(0.0, optimized_parts["instability"] - corrected_parts["instability"])
    ) / 2.0
    sensitivity = _minmax(sensitivity_raw)
    utility = (
        optimized_parts["discriminative"]
        - UTILITY_WEIGHTS["within_class"] * optimized_parts["within"]
        - UTILITY_WEIGHTS["instability"] * optimized_parts["instability"]
        - UTILITY_WEIGHTS["redundancy"] * optimized_parts["redundancy"]
        - UTILITY_WEIGHTS["refinement_sensitivity"] * sensitivity
    )
    ranking = np.argsort(-utility)
    rank_by_index = {int(index): rank + 1 for rank, index in enumerate(ranking)}
    rows: list[dict[str, object]] = []
    for index, actuator in enumerate(dimension_control.ACTUATORS):
        role, finger, actuator_type = ACTUATOR_META[index]
        rows.append({
            "index": index,
            "actuator_name": actuator,
            "semantic_role": role,
            "finger_group": finger,
            "type": actuator_type,
            "included_in_flex11": "yes" if index in dimension_control.FLEX11_INDICES else "no",
            "constant_on_analysis_partition": "yes" if optimized_parts["constant"][index] else "no",
            "oa_discriminative_f_raw": float(optimized_parts["f_score_raw"][index]),
            "oa_discriminative_scaled": float(optimized_parts["discriminative"][index]),
            "oa_within_class_variability_raw": float(optimized_parts["within_raw"][index]),
            "oa_within_class_penalty_scaled": float(optimized_parts["within"][index]),
            "oa_instability_raw": float(optimized_parts["instability_raw"][index]),
            "oa_instability_penalty_scaled": float(optimized_parts["instability"][index]),
            "oa_redundancy_max_abs_correlation": float(optimized_parts["redundancy"][index]),
            "refinement_sensitivity_penalty": float(sensitivity[index]),
            "utility_score": float(utility[index]),
            "utility_rank": rank_by_index[index],
        })
    return rows, optimized_parts["correlation"]


def select_semantic_subset(score_rows: list[dict[str, object]], k: int) -> list[int]:
    if k < len(FINGER_GROUPS):
        raise ValueError("Semantic selection requires at least one coordinate per finger.")
    ordered = sorted(score_rows, key=lambda row: float(row["utility_score"]), reverse=True)
    selected: list[int] = []
    for finger in FINGER_GROUPS:
        best = next(row for row in ordered if row["finger_group"] == finger)
        selected.append(int(best["index"]))
    for row in ordered:
        index = int(row["index"])
        if index not in selected:
            selected.append(index)
        if len(selected) == k:
            break
    return sorted(selected)


def _encode_subset(sequences: list[np.ndarray], indices: list[int]) -> np.ndarray:
    return np.stack([
        sequence_encoding.encode_sequence(sequence[:, indices], "resample16")
        for sequence in sequences
    ])


def development_validation(
    sequence_ids: list[str], labels: list[str], sequences: dict[str, list[np.ndarray]],
    development_indices: list[int], repeats: int, shot: int, validation_size: float, seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    development_labels = np.asarray([labels[index] for index in development_indices])
    rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        repeat_seed = seed + repeat
        inner_train_local, validation_local = train_test_split(
            np.arange(len(development_indices)), test_size=validation_size,
            random_state=repeat_seed, stratify=development_labels,
        )
        inner_train = [development_indices[index] for index in inner_train_local]
        validation = [development_indices[index] for index in validation_local]
        train_labels_all = [labels[index] for index in inner_train]
        score_rows, _ = actuator_scores(
            [sequences["corrected17"][index] for index in inner_train],
            [sequences["optimized_action17"][index] for index in inner_train],
            train_labels_all,
        )
        for score_row in score_rows:
            ranking_rows.append({"repeat": repeat, **score_row})

        class_labels = sorted(set(train_labels_all))
        chosen_train: list[int] = []
        for class_offset, label in enumerate(class_labels):
            pool = np.asarray([index for index in inner_train if labels[index] == label], dtype=int)
            rng = np.random.RandomState(repeat_seed + 1009 * (class_offset + 1))
            chosen_train.extend(rng.permutation(pool)[:shot].tolist())
        chosen_train = sorted(chosen_train)
        y_train = [labels[index] for index in chosen_train]
        y_validation = [labels[index] for index in validation]
        for k in CANDIDATE_K:
            selected = select_semantic_subset(score_rows, k)
            x_train = _encode_subset(
                [sequences["optimized_action17"][index] for index in chosen_train], selected
            )
            x_validation = _encode_subset(
                [sequences["optimized_action17"][index] for index in validation], selected
            )
            for classifier in dimension_control.CLASSIFIERS:
                model = train_svm._build_model(shot_sweep._model_args(classifier), repeat_seed)
                model.fit(x_train, y_train)
                prediction = model.predict(x_validation)
                accuracy = float(accuracy_score(y_validation, prediction))
                macro_f1 = float(f1_score(y_validation, prediction, average="macro", zero_division=0))
                rows.append({
                    "repeat": repeat,
                    "seed": repeat_seed,
                    "k": k,
                    "selected_indices_inner_train": ";".join(map(str, selected)),
                    "classifier": classifier,
                    "num_train": len(chosen_train),
                    "num_validation": len(validation),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "kappa": float(cohen_kappa_score(y_validation, prediction)),
                    "combined_score": 0.3 * accuracy + 0.7 * macro_f1,
                })
    return rows, ranking_rows


def select_k(rows: list[dict[str, object]]) -> tuple[int, list[dict[str, object]]]:
    by_repeat: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        by_repeat[(int(row["repeat"]), int(row["k"]))].append(float(row["combined_score"]))
    repeat_scores = {
        key: float(np.mean(values)) for key, values in by_repeat.items()
    }
    summary: list[dict[str, object]] = []
    for k in CANDIDATE_K:
        values = np.asarray([value for (repeat, candidate), value in repeat_scores.items() if candidate == k])
        summary.append({
            "k": k,
            "encoded_features": 16 * k,
            "repeats": len(values),
            "combined_mean": float(np.mean(values)),
            "combined_std": float(np.std(values)),
            "combined_ci95": float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))),
            "combined_standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        })
    best = max(summary, key=lambda row: float(row["combined_mean"]))
    threshold = float(best["combined_mean"]) - float(best["combined_standard_error"])
    selected_k = min(int(row["k"]) for row in summary if float(row["combined_mean"]) >= threshold)
    for row in summary:
        row["best_k_by_mean"] = int(best["k"])
        row["one_se_threshold"] = threshold
        row["eligible_by_one_se_rule"] = "yes" if float(row["combined_mean"]) >= threshold else "no"
        row["selected_k"] = selected_k
    return selected_k, summary


def _inventory() -> list[dict[str, object]]:
    return [{
        "index": index,
        "actuator_name": actuator,
        "anatomical_semantic_role": ACTUATOR_META[index][0],
        "finger_group": ACTUATOR_META[index][1],
        "type": ACTUATOR_META[index][2],
        "included_in_current_flex11": "yes" if index in dimension_control.FLEX11_INDICES else "no",
    } for index, actuator in enumerate(dimension_control.ACTUATORS)]


def _redundancy_rows(matrix: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for row_index, actuator in enumerate(dimension_control.ACTUATORS):
        row: dict[str, object] = {"row_index": row_index, "row_actuator": actuator}
        row.update({f"corr_{column_index}": float(matrix[row_index, column_index]) for column_index in range(17)})
        rows.append(row)
    return rows


def _plot_selection(output: Path, scores: list[dict[str, object]], correlation: np.ndarray,
                    dimension_summary: list[dict[str, object]]) -> None:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    ordered = sorted(scores, key=lambda row: float(row["utility_score"]))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([str(row["semantic_role"]) for row in ordered],
            [float(row["utility_score"]) for row in ordered], color="#E66A21")
    ax.set_xlabel("Development-only utility score")
    ax.set_title("Optimized Action Actuator Utility")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "01_actuator_utility_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [meta[0] for meta in ACTUATOR_META]
    ax.set_xticks(range(17), labels, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(17), labels, fontsize=7)
    ax.set_title("Development Optimized-Action Correlation")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(figures / "02_actuator_redundancy_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    values = sorted(dimension_summary, key=lambda row: int(row["k"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar([int(row["k"]) for row in values], [float(row["combined_mean"]) for row in values],
                yerr=[float(row["combined_ci95"]) for row in values], marker="o", capsize=4,
                color="#4178A8")
    selected = next(int(row["selected_k"]) for row in values)
    ax.axvline(selected, color="#E66A21", linestyle="--", label=f"Selected K*={selected}")
    ax.set_xticks(CANDIDATE_K)
    ax.set_xlabel("Selected actuator dimensions K")
    ax.set_ylabel("Development combined score")
    ax.set_title("Nested Development Performance vs Compactness")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "03_development_performance_vs_k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


FINAL_REPRESENTATIONS = (
    "joint_angle11",
    "corrected17",
    "optimized_action17",
    "corrected_flex11",
    "optimized_action_flex11",
    "compact_corrected7",
    "compact_optimized_action7",
    "corrected_pca11",
    "optimized_action_pca11",
)
FINAL_DISPLAY = {
    "joint_angle11": "JointAngle-11",
    "corrected17": "Corrected-17",
    "optimized_action17": "OptimizedAction-17",
    "corrected_flex11": "Corrected-Flex11",
    "optimized_action_flex11": "OptimizedAction-Flex11",
    "compact_corrected7": "Compact Corrected-7",
    "compact_optimized_action7": "Compact OA-7",
    "corrected_pca11": "Corrected-PCA11",
    "optimized_action_pca11": "OptimizedAction-PCA11",
}
FINAL_DIMENSIONS = {
    "joint_angle11": 11, "corrected17": 17, "optimized_action17": 17,
    "corrected_flex11": 11, "optimized_action_flex11": 11,
    "compact_corrected7": 7, "compact_optimized_action7": 7,
    "corrected_pca11": 11, "optimized_action_pca11": 11,
}
FINAL_COLORS = {
    "joint_angle11": "#D99A20", "corrected17": "#448C57",
    "optimized_action17": "#E66A21", "corrected_flex11": "#74B985",
    "optimized_action_flex11": "#F29A55", "compact_corrected7": "#2D6A4F",
    "compact_optimized_action7": "#C84630", "corrected_pca11": "#4178A8",
    "optimized_action_pca11": "#8B65A8",
}
PRIMARY_FINAL_COMPARISONS = (
    ("compact_optimized_action7", "joint_angle11"),
    ("compact_optimized_action7", "optimized_action_flex11"),
    ("compact_optimized_action7", "optimized_action17"),
    ("compact_corrected7", "compact_optimized_action7"),
    ("compact_corrected7", "corrected_flex11"),
    ("compact_corrected7", "corrected17"),
)


def _verify_frozen_gate(output: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    development_path = output / "development_sequences.csv"
    final_path = output / "final_test_sequences.csv"
    spec_path = output / "FINAL_COMPACT_ORCA_SPEC.md"
    for path in (development_path, final_path, spec_path):
        if not path.exists():
            raise RuntimeError(f"Frozen prerequisite missing: {path}")
    if _file_digest(development_path) != FROZEN_DEVELOPMENT_SHA256:
        raise RuntimeError("Development manifest hash changed after selection; final test aborted.")
    if _file_digest(final_path) != FROZEN_FINAL_SHA256:
        raise RuntimeError("Final-test manifest hash changed after selection; final test aborted.")
    spec = spec_path.read_text(encoding="utf-8")
    if "Chosen K*: **7**" not in spec:
        raise RuntimeError("Frozen specification no longer states K*=7.")
    for index in FROZEN_INDICES:
        if f"| {index} |" not in spec:
            raise RuntimeError(f"Frozen actuator index {index} missing from specification.")
    final_outputs = (
        "final_test_results.csv", "final_test_per_repeat.csv",
        "final_test_paired_comparisons.csv", "final_test_paired_comparisons_holm.csv",
    )
    existing = [name for name in final_outputs if (output / name).exists()]
    if existing:
        raise RuntimeError(f"Final evaluation has already been written ({existing}); refusing a second run.")
    development_rows, final_rows = _read_manifest(development_path), _read_manifest(final_path)
    if len(development_rows) != 456 or len(final_rows) != 115:
        raise RuntimeError("Frozen split sizes no longer match 456/115.")
    return development_rows, final_rows


def _summarize_final(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["classifier"]), str(row["representation"]))].append(row)
    output: list[dict[str, object]] = []
    for (classifier, representation), values in sorted(groups.items()):
        result: dict[str, object] = {
            "classifier": classifier,
            "representation": representation,
            "display_name": FINAL_DISPLAY[representation],
            "frame_dimensions": FINAL_DIMENSIONS[representation],
            "encoded_features": 16 * FINAL_DIMENSIONS[representation],
            "repeats": len(values),
        }
        for metric in ("accuracy", "macro_f1", "kappa"):
            data = np.asarray([float(row[metric]) for row in values])
            result[f"{metric}_mean"] = float(np.mean(data))
            result[f"{metric}_std"] = float(np.std(data))
            result[f"{metric}_ci95"] = float(1.96 * np.std(data, ddof=1) / np.sqrt(len(data)))
        output.append(result)
    return output


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, original_index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[original_index]))
        running = max(running, value)
        adjusted[original_index] = running
    return adjusted.tolist()


def _paired_final(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from scipy.stats import wilcoxon

    lookup = {
        (int(row["repeat"]), str(row["classifier"]), str(row["representation"])): row
        for row in rows
    }
    repeats = sorted({int(row["repeat"]) for row in rows})
    raw: list[dict[str, object]] = []
    for classifier in dimension_control.CLASSIFIERS:
        for first, second in PRIMARY_FINAL_COMPARISONS:
            for metric in ("accuracy", "macro_f1", "kappa"):
                differences = np.asarray([
                    float(lookup[(repeat, classifier, first)][metric])
                    - float(lookup[(repeat, classifier, second)][metric])
                    for repeat in repeats
                ])
                p_value = 1.0 if not np.any(differences != 0) else float(wilcoxon(differences).pvalue)
                sample_std = float(np.std(differences, ddof=1))
                raw.append({
                    "classifier": classifier,
                    "metric": metric,
                    "first": first,
                    "second": second,
                    "comparison": f"{first}_minus_{second}",
                    "mean_difference": float(np.mean(differences)),
                    "std_difference": float(np.std(differences)),
                    "ci95_difference": float(1.96 * sample_std / np.sqrt(len(differences))),
                    "positive_repeats": int(np.sum(differences > 0)),
                    "equal_repeats": int(np.sum(differences == 0)),
                    "negative_repeats": int(np.sum(differences < 0)),
                    "wilcoxon_p_raw": p_value,
                    "cohen_dz": float(np.mean(differences) / sample_std) if sample_std > 1e-12 else 0.0,
                })
    corrected = [dict(row) for row in raw]
    for classifier in dimension_control.CLASSIFIERS:
        for metric in ("accuracy", "macro_f1", "kappa"):
            indices = [index for index, row in enumerate(corrected)
                       if row["classifier"] == classifier and row["metric"] == metric]
            adjusted = _holm_adjust([float(corrected[index]["wilcoxon_p_raw"]) for index in indices])
            for index, value in zip(indices, adjusted, strict=True):
                corrected[index]["wilcoxon_p_holm"] = value
                corrected[index]["holm_significant_0_05"] = "yes" if value < 0.05 else "no"
                corrected[index]["holm_family"] = f"{classifier}:{metric}:six_primary_comparisons"
    return raw, corrected


def evaluate_frozen_final(
    sequence_ids: list[str], labels: list[str], sequences: dict[str, list[np.ndarray]],
    development_rows: list[dict[str, str]], final_rows: list[dict[str, str]],
    repeats: int, shot: int, seed: int,
) -> list[dict[str, object]]:
    id_to_index = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}
    development = [id_to_index[row["sequence_id"]] for row in development_rows]
    final = [id_to_index[row["sequence_id"]] for row in final_rows]
    for row in development_rows + final_rows:
        if labels[id_to_index[row["sequence_id"]]] != row["label"]:
            raise RuntimeError(f"Label changed for frozen sequence {row['sequence_id']}")
    y_test = [labels[index] for index in final]
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        repeat_seed = seed + repeat
        chosen_train: list[int] = []
        for class_offset, label in enumerate(sorted(set(labels[index] for index in development))):
            pool = np.asarray([index for index in development if labels[index] == label], dtype=int)
            rng = np.random.RandomState(repeat_seed + 1009 * (class_offset + 1))
            chosen_train.extend(rng.permutation(pool)[:shot].tolist())
        chosen_train = sorted(chosen_train)
        y_train = [labels[index] for index in chosen_train]
        split_sequences: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        for representation in (
            "joint_angle11", "corrected17", "optimized_action17",
            "corrected_flex11", "optimized_action_flex11",
        ):
            split_sequences[representation] = (
                [sequences[representation][index] for index in chosen_train],
                [sequences[representation][index] for index in final],
            )
        split_sequences["compact_corrected7"] = (
            [sequences["corrected17"][index][:, FROZEN_INDICES] for index in chosen_train],
            [sequences["corrected17"][index][:, FROZEN_INDICES] for index in final],
        )
        split_sequences["compact_optimized_action7"] = (
            [sequences["optimized_action17"][index][:, FROZEN_INDICES] for index in chosen_train],
            [sequences["optimized_action17"][index][:, FROZEN_INDICES] for index in final],
        )
        for source, target in (("corrected17", "corrected_pca11"),
                               ("optimized_action17", "optimized_action_pca11")):
            split_sequences[target] = train_svm._project_sequences_with_pca(
                *split_sequences[source], n_components=11, seed=repeat_seed
            )
        for representation in FINAL_REPRESENTATIONS:
            train_values, test_values = split_sequences[representation]
            x_train = np.stack([sequence_encoding.encode_sequence(value, "resample16") for value in train_values])
            x_test = np.stack([sequence_encoding.encode_sequence(value, "resample16") for value in test_values])
            for classifier in dimension_control.CLASSIFIERS:
                model = train_svm._build_model(shot_sweep._model_args(classifier), repeat_seed)
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                rows.append({
                    "repeat": repeat, "seed": repeat_seed, "classifier": classifier,
                    "representation": representation,
                    "frame_dimensions": FINAL_DIMENSIONS[representation],
                    "encoded_features": int(x_train.shape[1]),
                    "num_train": len(chosen_train), "num_test": len(final),
                    "train_sequence_ids": ";".join(sequence_ids[index] for index in chosen_train),
                    "accuracy": float(accuracy_score(y_test, prediction)),
                    "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
                    "kappa": float(cohen_kappa_score(y_test, prediction)),
                })
    return rows


def _plot_final_grouped(summary: list[dict[str, object]], metric: str, selected: tuple[str, ...],
                        path: Path, title: str) -> None:
    lookup = {(str(row["classifier"]), str(row["representation"])): row for row in summary}
    x = np.arange(len(dimension_control.CLASSIFIERS), dtype=float)
    width = 0.82 / len(selected)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for offset, representation in enumerate(selected):
        values = [float(lookup[(classifier, representation)][f"{metric}_mean"])
                  for classifier in dimension_control.CLASSIFIERS]
        errors = [float(lookup[(classifier, representation)][f"{metric}_ci95"])
                  for classifier in dimension_control.CLASSIFIERS]
        positions = x - 0.41 + width / 2 + offset * width
        ax.bar(positions, values, width, yerr=errors, capsize=3,
               color=FINAL_COLORS[representation], label=FINAL_DISPLAY[representation])
    ax.set_xticks(x, [shot_sweep.DISPLAY_CLASSIFIERS[value] for value in dimension_control.CLASSIFIERS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dimension(summary: list[dict[str, object]], metric: str, path: Path) -> None:
    primary = FINAL_REPRESENTATIONS[:7]
    lookup = {(str(row["classifier"]), str(row["representation"])): row for row in summary}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, classifier in zip(axes.flat, dimension_control.CLASSIFIERS, strict=True):
        for representation in primary:
            row = lookup[(classifier, representation)]
            ax.errorbar(FINAL_DIMENSIONS[representation], float(row[f"{metric}_mean"]),
                        yerr=float(row[f"{metric}_ci95"]), marker="o", capsize=3,
                        color=FINAL_COLORS[representation])
            ax.annotate(FINAL_DISPLAY[representation],
                        (FINAL_DIMENSIONS[representation], float(row[f"{metric}_mean"])),
                        xytext=(3, 3), textcoords="offset points", fontsize=6)
        ax.set_title(shot_sweep.DISPLAY_CLASSIFIERS[classifier])
        ax.grid(alpha=0.2)
    fig.supxlabel("Frame-level dimensions")
    fig.supylabel(metric.replace("_", " ").title())
    fig.suptitle(f"Final-test {metric.replace('_', ' ').title()} vs Representation Dimension")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_differences(rows: list[dict[str, object]], path: Path) -> None:
    lookup = {
        (int(row["repeat"]), str(row["classifier"]), str(row["representation"])): row
        for row in rows
    }
    repeats = sorted({int(row["repeat"]) for row in rows})
    fig, ax = plt.subplots(figsize=(8.5, 5))
    data = []
    for classifier in dimension_control.CLASSIFIERS:
        values = [
            float(lookup[(repeat, classifier, "compact_optimized_action7")]["accuracy"])
            - float(lookup[(repeat, classifier, "joint_angle11")]["accuracy"])
            for repeat in repeats
        ]
        data.append(values)
    ax.boxplot(data, tick_labels=[shot_sweep.DISPLAY_CLASSIFIERS[value]
                                  for value in dimension_control.CLASSIFIERS], showmeans=True)
    for position, values in enumerate(data, start=1):
        jitter = np.linspace(-0.10, 0.10, len(values))
        ax.scatter(position + jitter, values, s=15, alpha=0.55, color="#C84630")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel("Compact OA-7 minus JointAngle-11 accuracy")
    ax.set_title("Paired Final-test Differences Across Repeats")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compactness_rows() -> list[dict[str, object]]:
    return [
        {"representation": "JointAngle-11", "frame_dimensions": 11, "resample_frames": 16,
         "encoded_features": 176, "reduction_vs_orca17": 1 - 11 / 17,
         "reduction_vs_jointangle11": 0.0},
        {"representation": "OptimizedAction-Flex11", "frame_dimensions": 11, "resample_frames": 16,
         "encoded_features": 176, "reduction_vs_orca17": 1 - 11 / 17,
         "reduction_vs_jointangle11": 0.0},
        {"representation": "Compact OptimizedAction-7", "frame_dimensions": 7, "resample_frames": 16,
         "encoded_features": 112, "reduction_vs_orca17": 1 - 7 / 17,
         "reduction_vs_jointangle11": 1 - 7 / 11},
        {"representation": "OptimizedAction-17", "frame_dimensions": 17, "resample_frames": 16,
         "encoded_features": 272, "reduction_vs_orca17": 0.0,
         "reduction_vs_jointangle11": 1 - 17 / 11},
    ]


def _safe_final_claim(summary: list[dict[str, object]], holm: list[dict[str, object]]) -> tuple[str, str]:
    lookup = {(str(row["classifier"]), str(row["representation"])): row for row in summary}
    oa_minus_joint = [row for row in holm if row["comparison"] ==
                      "compact_optimized_action7_minus_joint_angle11"]
    supported_superiority = True
    all_oa_higher = True
    for classifier in dimension_control.CLASSIFIERS:
        oa = lookup[(classifier, "compact_optimized_action7")]
        joint = lookup[(classifier, "joint_angle11")]
        all_oa_higher &= (float(oa["accuracy_mean"]) > float(joint["accuracy_mean"])
                          and float(oa["macro_f1_mean"]) > float(joint["macro_f1_mean"]))
        accuracy_test = next(row for row in oa_minus_joint
                             if row["classifier"] == classifier and row["metric"] == "accuracy")
        f1_test = next(row for row in oa_minus_joint
                       if row["classifier"] == classifier and row["metric"] == "macro_f1")
        supported_superiority &= (
            float(accuracy_test["mean_difference"]) > 0
            and float(accuracy_test["mean_difference"]) - float(accuracy_test["ci95_difference"]) > 0
            and accuracy_test["holm_significant_0_05"] == "yes"
            and float(f1_test["mean_difference"]) > 0
        )
    if supported_superiority and all_oa_higher:
        return "superiority", (
            "The frozen 7D compact refined ORCA representation outperformed JointAngle-11 under the "
            "evaluated few-shot protocol while using fewer frame-level dimensions."
        )
    # Comparable is used when no classifier has Holm-supported evidence that JointAngle is better.
    joint_supported = any(
        float(row["mean_difference"]) < 0
        and float(row["mean_difference"]) + float(row["ci95_difference"]) < 0
        and row["holm_significant_0_05"] == "yes"
        for row in oa_minus_joint if row["metric"] in {"accuracy", "macro_f1"}
    )
    if not joint_supported:
        return "comparability", (
            "The frozen 7D compact refined ORCA representation achieved comparable recognition performance "
            "to JointAngle-11 while using 36.4% fewer frame-level dimensions."
        )
    return "compactness_only", (
        "JointAngle-11 retained higher recognition performance, while the frozen Compact OA-7 representation "
        "provided a 36.4% reduction in frame-level dimensions under the evaluated protocol."
    )


def _write_final_document(output: Path, summary: list[dict[str, object]],
                          holm: list[dict[str, object]], interpretation: str, claim: str) -> None:
    lookup = {(str(row["classifier"]), str(row["representation"])): row for row in summary}
    selected_rows = [row for row in holm if row["comparison"] in {
        "compact_optimized_action7_minus_joint_angle11",
        "compact_optimized_action7_minus_optimized_action_flex11",
        "compact_optimized_action7_minus_optimized_action17",
        "compact_corrected7_minus_compact_optimized_action7",
    } and row["metric"] in {"accuracy", "macro_f1"}]
    lines = [
        "# Final Compact ORCA Test Explained", "",
        "## A. What was frozen before final testing?", "",
        "K*=7, actuator indices `[3, 6, 9, 11, 12, 15, 16]`, the 456/115 outer split, "
        "Resample-16, 3-shot sampling, 20 seeds, classifier settings, utility formula, and semantic selection "
        "rule were frozen. Manifest hashes were verified before execution.", "",
        "## B. Why could the final test not influence selection?", "",
        "Actuator utility and K were derived exclusively from nested development validation. The final program "
        "loaded the frozen subset directly and contains no ranking or K-selection call. It also refuses to run if "
        "final output files already exist.", "",
        "## C. Why was K=7 selected?", "",
        "K=7 had the highest four-classifier development combined score (0.8088); K=9 was nearly identical "
        "at 0.8085. The predefined smallest-within-one-standard-error rule therefore froze K=7.", "",
        "## D. Which actuators were selected?", "",
        "The seven coordinates are little PIP (3), ring PIP (6), middle PIP (9), index MCP/PIP (11/12), "
        "and thumb MCP/IP (15/16). They describe finger flexion and retain coverage of all five fingers.", "",
        "## E. How is 7D different from JointAngle-11?", "",
        "Compact OA-7 contains seven ORCA actuator coordinates after MuJoCo-constrained refinement. JointAngle-11 "
        "contains eleven angles computed directly from triples of MediaPipe landmarks. They differ in dimension, "
        "semantics, and whether temporal refinement is applied.", "",
        "## F. Why is the comparison fair?", "",
        "All representations use identical 3-shot training sequence IDs in every repeat, the same frozen 115-sequence "
        "test set, Resample-16, training-only scaling, and fixed classifier settings. No classifier is tuned for Compact ORCA.", "",
        "## G. Exact final-test protocol", "",
        "Each of 20 repeats samples three development sequences per class using seed `42 + repeat`. The frozen final "
        "test never changes. Models are trained separately for SVM, KNN, RandomForest, and MLP. Repeat, not frame, "
        "is the statistical unit.", "",
        "## H. Final results for all representations", "",
        "Values are mean Accuracy / Macro-F1 / Kappa.", "",
        "| Representation | SVM | KNN | RandomForest | MLP |", "|---|---|---|---|---|",
    ]
    for representation in FINAL_REPRESENTATIONS:
        cells = []
        for classifier in dimension_control.CLASSIFIERS:
            row = lookup[(classifier, representation)]
            cells.append(f"{float(row['accuracy_mean']):.4f} / {float(row['macro_f1_mean']):.4f} / "
                         f"{float(row['kappa_mean']):.4f}")
        lines.append(f"| {FINAL_DISPLAY[representation]} | " + " | ".join(cells) + " |")
    lines += ["", "## I-L. Primary paired comparisons", "",
              "Differences are first representation minus second representation.", "",
              "| Classifier | Metric | Comparison | Difference | 95% CI | Raw p | Holm p | dz |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in selected_rows:
        lines.append(f"| {row['classifier']} | {row['metric']} | {row['comparison']} | "
                     f"{float(row['mean_difference']):+.4f} | {float(row['ci95_difference']):.4f} | "
                     f"{float(row['wilcoxon_p_raw']):.4g} | {float(row['wilcoxon_p_holm']):.4g} | "
                     f"{float(row['cohen_dz']):+.3f} |")
    lines += [
        "", "## M. Statistical interpretation", "",
        "Wilcoxon tests are paired by repeat. Holm adjustment is applied within each classifier-and-metric family "
        "across the six predefined primary comparisons. Raw p-values are not treated as sufficient when Holm-adjusted "
        "p-values are non-significant.", "",
        "## N. Overall outcome", "", f"The predefined decision rule classifies this result as **{interpretation}**.", "",
        "## O. Safe paper claim", "", f"> {claim}", "",
        "## P. Claims that must not be made", "",
        "Do not claim universal superiority, optimality over all actuator subsets, generalization beyond this six-class "
        "dataset, or that removed actuators are biologically unimportant. Do not alter K or the seven coordinates after "
        "these results.", "",
        "## Q. Code locations", "",
        "- `run_compact_orca_selection.py::_verify_frozen_gate`: validates hashes/spec and prevents reruns.",
        "- `run_compact_orca_selection.py::evaluate_frozen_final`: applies the frozen subset and common repeats.",
        "- `evaluate_sequence_encodings.py::encode_sequence`: performs Resample-16.",
        "- `train_svm.py::_build_model`: fixed classifiers with training-only StandardScaler.",
        "- `run_compact_orca_selection.py::_paired_final`: repeat-level Wilcoxon and effect size.",
        "- `run_compact_orca_selection.py::_holm_adjust`: Holm correction within predefined families.", "",
        "Core frozen selection code:", "", "```python",
        "FROZEN_INDICES = (3, 6, 9, 11, 12, 15, 16)",
        "compact_sequence = actuator_sequence[:, FROZEN_INDICES]",
        "encoded = resample16(compact_sequence)",
        "```", "",
        "## R. Plain-language conclusion", "",
        claim,
    ]
    (output / "FINAL_COMPACT_ORCA_TEST_EXPLAINED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_final_stage(dataset: Path, output: Path, args: argparse.Namespace) -> None:
    development_rows, final_rows = _verify_frozen_gate(output)
    sequence_ids, labels, sequences = dimension_control._load_base(dataset)
    rows = evaluate_frozen_final(
        sequence_ids, labels, sequences, development_rows, final_rows,
        args.repeats, args.shot, args.inner_seed,
    )
    summary = _summarize_final(rows)
    paired, holm = _paired_final(rows)
    interpretation, claim = _safe_final_claim(summary, holm)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    primary = FINAL_REPRESENTATIONS[:7]
    _plot_final_grouped(summary, "accuracy", primary, figures / "04_final_accuracy_by_representation.png",
                        "Frozen Final-test Accuracy")
    _plot_final_grouped(summary, "macro_f1", primary, figures / "05_final_macro_f1_by_representation.png",
                        "Frozen Final-test Macro-F1")
    _plot_final_grouped(summary, "accuracy",
                        ("joint_angle11", "optimized_action_flex11", "compact_optimized_action7"),
                        figures / "06_jointangle_vs_oa_flex_vs_compact_oa.png",
                        "JointAngle-11 vs OA-Flex11 vs Compact OA-7")
    _plot_dimension(summary, "accuracy", figures / "07_accuracy_vs_frame_dimension.png")
    _plot_dimension(summary, "macro_f1", figures / "08_macro_f1_vs_frame_dimension.png")
    _plot_final_grouped(summary, "accuracy", ("compact_corrected7", "compact_optimized_action7"),
                        figures / "09_compact_corrected_vs_compact_oa.png",
                        "Compact Corrected-7 vs Compact OA-7")
    _plot_paired_differences(rows, figures / "10_compact_oa_vs_jointangle_paired_differences.png")
    _write_rows(output / "final_test_per_repeat.csv", rows)
    _write_rows(output / "final_test_results.csv", summary)
    _write_rows(output / "final_test_paired_comparisons.csv", paired)
    _write_rows(output / "final_test_paired_comparisons_holm.csv", holm)
    _write_rows(output / "compactness_summary.csv", _compactness_rows())
    _write_final_document(output, summary, holm, interpretation, claim)
    print(f"final_test_evaluated=yes")
    print(f"rows={len(rows)} summaries={len(summary)} paired_tests={len(paired)}")
    print(f"interpretation={interpretation}")
    print(f"safe_claim={claim}")


def _write_frozen_spec(output: Path, selected_k: int, selected_indices: list[int],
                       dimension_summary: list[dict[str, object]], scores: list[dict[str, object]],
                       development_count: int, final_count: int, args: argparse.Namespace) -> None:
    score_lookup = {int(row["index"]): row for row in scores}
    lines = [
        "# Final Compact ORCA Specification", "",
        f"Frozen at: `{datetime.now(timezone.utc).isoformat()}`", "",
        "This specification was produced before final-test evaluation. The final-test labels and performance "
        "were not used for actuator ranking, K selection, preprocessing, or classifier settings.", "",
        f"- Development sequences: {development_count}",
        f"- Frozen final-test sequences: {final_count}",
        f"- Outer split seed: {args.outer_seed}",
        f"- Development repeats: {args.repeats}",
        f"- Few-shot training: {args.shot} sequences per class", "- Encoding: Resample-16",
        "- Combined development score: 0.7 Macro-F1 + 0.3 Accuracy, averaged across four classifiers",
        "- K rule: smallest candidate within one standard error of the best development mean",
        f"- Chosen K*: **{selected_k}**",
        f"- Encoded features: **{16 * selected_k}**",
        f"- Reduction from ORCA-17: **{100.0 * (1.0 - selected_k / 17.0):.1f}%**",
        f"- Reduction from JointAngle-11: **{100.0 * (1.0 - selected_k / 11.0):.1f}%**", "",
        "## Frozen Actuators", "", "| Index | Actuator | Role | Utility |", "|---:|---|---|---:|",
    ]
    for index in selected_indices:
        row = score_lookup[index]
        lines.append(f"| {index} | `{row['actuator_name']}` | {row['semantic_role']} | "
                     f"{float(row['utility_score']):.4f} |")
    lines += ["", "## Candidate Development Results", "",
              "| K | Features | Combined mean | 95% CI | Eligible |", "|---:|---:|---:|---:|---|"]
    for row in sorted(dimension_summary, key=lambda value: int(value["k"])):
        lines.append(f"| {row['k']} | {row['encoded_features']} | {float(row['combined_mean']):.4f} | "
                     f"{float(row['combined_ci95']):.4f} | {row['eligible_by_one_se_rule']} |")
    lines += ["", "## Frozen Parameters", "",
              "Utility = discriminative - 0.25 within-class - 0.20 instability - 0.15 redundancy "
              "- 0.15 refinement-sensitivity. Components are scaled across actuators using development data only.",
              "Semantic constraint: at least one actuator from each of thumb, index, middle, ring, and little; "
              "remaining positions are filled by descending utility.", "",
              f"Development manifest SHA256: `{_file_digest(output / 'development_sequences.csv')}`",
              f"Final-test manifest SHA256: `{_file_digest(output / 'final_test_sequences.csv')}`", "",
              "The subset must not be changed after inspecting final-test performance."]
    (output / "FINAL_COMPACT_ORCA_SPEC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _limited_reason(row: dict[str, object]) -> str:
    if row.get("constant_on_analysis_partition") == "yes":
        return "constant on the development analysis partition"
    discriminative = float(row["oa_discriminative_scaled"])
    penalties = {
        "within-class variability": UTILITY_WEIGHTS["within_class"] * float(row["oa_within_class_penalty_scaled"]),
        "temporal instability": UTILITY_WEIGHTS["instability"] * float(row["oa_instability_penalty_scaled"]),
        "redundancy": UTILITY_WEIGHTS["redundancy"] * float(row["oa_redundancy_max_abs_correlation"]),
        "refinement sensitivity": UTILITY_WEIGHTS["refinement_sensitivity"] * float(row["refinement_sensitivity_penalty"]),
    }
    strongest = max(penalties, key=penalties.get)
    if discriminative < 0.25:
        return f"limited additional development discrimination; strongest penalty was {strongest}"
    return f"ranked below the frozen K* cutoff; strongest penalty was {strongest}"


def _write_teaching_document(
    output: Path, scores: list[dict[str, object]], selected_k: int, selected_indices: list[int],
    dimension_summary: list[dict[str, object]], development_count: int, final_count: int,
    args: argparse.Namespace,
) -> None:
    lookup = {int(row["index"]): row for row in scores}
    lines = [
        "# Compact ORCA Selection Explained", "",
        "## A. Why may 17D contain redundancy?", "",
        "The 17 ORCA coordinates include wrist, abduction, opposition, and flexion. Some may be correlated or "
        "nearly constant for the current six Chinese-dance gestures. Such coordinates can increase a 3-shot "
        "classifier's input size without adding proportional class information.", "",
        "## B. Why may JointAngle-11 benefit from lower dimension?", "",
        "Resample-16 converts JointAngle-11 into 176 inputs but ORCA-17 into 272. With only 18 training sequences "
        "per repeat, the smaller representation can be easier to estimate. The preceding dimension-control experiment "
        "also showed that semantic Flex11 was generally stronger than the corresponding ORCA-17 representation.", "",
        "## C. Why did Flex11 motivate this experiment?", "",
        "Flex11 removed wrist and abduction coordinates using a predefined semantic rule. Its improvement suggested "
        "that some ORCA coordinates contributed limited additional recognition information under this protocol. The "
        "current experiment asks whether development data support an even smaller, still interpretable subset.", "",
        "## D. Why test an even smaller subset?", "",
        "A compact representation may reduce model input, training variance, storage, and downstream computation. "
        "The scientific goal is not to force a higher score, but to determine the smallest development-supported "
        "actuator set before touching the final test.", "",
        "## E. Why development-only selection is necessary", "",
        "Actuator ranking, K selection, thresholds, and semantic rules are all model-design decisions. They must be "
        "made without final-test feedback so the final test remains an unbiased evaluation of the frozen design.", "",
        "## F. Why final-test selection would be invalid", "",
        "Trying several subsets on final-test labels and retaining the best would indirectly train on the test set. "
        "The reported score would then include selection luck and would overestimate generalization.", "",
        "## G. Exact frozen split", "",
        f"The outer stratified sequence split uses seed `{args.outer_seed}`: **{development_count} development** "
        f"sequences and **{final_count} frozen final-test** sequences. Exact IDs are in "
        "`development_sequences.csv` and `final_test_sequences.csv`. The manifests are reused if the script runs again.", "",
        "## H. Exact actuator scoring method", "",
        "For each actuator, sequence means provide an ANOVA F score for between-class discrimination. Class-conditional "
        "variance measures within-class variability. Z-scaled first and second temporal differences measure instability. "
        "Maximum absolute Pearson correlation measures redundancy. Refinement sensitivity penalizes loss of normalized "
        "discrimination or increased instability from Corrected to Optimized Action.", "",
        "All five components are computed using development data only and scaled across the 17 actuators. The frozen score is:", "",
        "```text",
        "utility = discriminative",
        "          - 0.25 * within_class",
        "          - 0.20 * instability",
        "          - 0.15 * redundancy",
        "          - 0.15 * refinement_sensitivity",
        "```", "",
        "To keep the result anatomically interpretable, each candidate contains at least one coordinate from each "
        "finger. Remaining slots follow descending development utility. No arbitrary subset search is performed.", "",
        "## I. Exact selected K*", "",
        f"The candidate path was `{', '.join(map(str, CANDIDATE_K))}`. The best development mean occurred at "
        f"**K={max(dimension_summary, key=lambda row: float(row['combined_mean']))['k']}**. The predefined one-standard-error "
        f"rule selected the smallest eligible candidate, **K*={selected_k}**. This produces **{selected_k * 16}** "
        "Resample-16 classifier inputs.", "",
        "| K | Encoded inputs | Combined mean | 95% CI | One-SE eligible |", "|---:|---:|---:|---:|---|",
    ]
    for row in sorted(dimension_summary, key=lambda value: int(value["k"])):
        lines.append(f"| {row['k']} | {row['encoded_features']} | {float(row['combined_mean']):.4f} | "
                     f"{float(row['combined_ci95']):.4f} | {row['eligible_by_one_se_rule']} |")
    lines += ["", "## J. Retained actuator names and indices", "",
              "| Index | Actuator | Meaning | Why retained |", "|---:|---|---|---|"]
    for index in selected_indices:
        row = lookup[index]
        lines.append(f"| {index} | `{row['actuator_name']}` | {row['semantic_role']} | development utility "
                     f"rank {row['utility_rank']}; contributes to {row['finger_group']} coverage |")
    lines += ["", "## K. Why each removed actuator was not retained", "",
              "These statements apply only to the current dataset and protocol; they do not imply biological uselessness.", "",
              "| Index | Actuator | Development interpretation |", "|---:|---|---|"]
    for index in range(17):
        if index not in selected_indices:
            row = lookup[index]
            lines.append(f"| {index} | `{row['actuator_name']}` | {_limited_reason(row)} |")
    lines += [
        "", "## L. Resample-16 dimensions", "",
        f"Each selected trajectory is linearly resampled to 16 ordered time positions. Compact OA-{selected_k} therefore "
        f"has `{selected_k} x 16 = {selected_k * 16}` inputs, compared with `11 x 16 = 176` for JointAngle and "
        "`17 x 16 = 272` for full ORCA. Temporal order is retained by flattening the 16 samples in order.", "",
        "## M. Classifier training", "",
        "Within each development repeat, ranking uses only the inner training partition. The classifier then receives "
        "three sequences per class, while validation sequences remain separate. SVM, KNN, RandomForest, and MLP use "
        "the fixed paper hyperparameters. The model pipeline fits StandardScaler on classifier training data only.", "",
        "## N. Statistics", "",
        "Development selection averages `0.7 * Macro-F1 + 0.3 * Accuracy` across all four classifiers for each repeat. "
        "The one-standard-error rule compares repeat-level combined scores and prefers the smallest K within one "
        "standard error of the best mean. Final paired tests will use identical repeat-level training selections and "
        "the same frozen final-test sequences; frames will not be treated as independent samples.", "",
        "## O. How to interpret possible final outcomes", "",
        f"- Compact OA-{selected_k} above JointAngle-11 with supported paired statistics: compact refined ORCA wins under this frozen protocol.",
        f"- Compact OA-{selected_k} similar to JointAngle-11: comparable recognition with {100.0 * (1-selected_k/11):.1f}% fewer frame dimensions.",
        "- JointAngle-11 higher: human-joint geometry remains better matched to this classification task.",
        "- Compact OA above OA-Flex11/OA-17: additional coordinates were redundant or noisy for this protocol.",
        "- Compact Corrected above Compact OA: refinement did not improve recognition for the selected coordinates.", "",
        "## P. Relevant code", "",
        "- `run_compact_orca_selection.py::freeze_outer_split`: creates or reloads the immutable outer manifests.",
        "- `run_compact_orca_selection.py::actuator_scores`: calculates the five development-only score components.",
        "- `run_compact_orca_selection.py::select_semantic_subset`: enforces five-finger coverage and utility ordering.",
        "- `run_compact_orca_selection.py::development_validation`: reranks inside every inner training partition.",
        "- `run_compact_orca_selection.py::select_k`: applies the four-classifier one-standard-error rule.",
        "- `evaluate_sequence_encodings.py::encode_sequence`: performs Resample-16.",
        "- `train_svm.py::_build_model`: training-only scaling and fixed classifiers.", "",
        "Core selection snippet:", "", "```python",
        "selected = best_actuator_per_finger(score_rows)",
        "selected += remaining_actuators_in_utility_order",
        "selected = selected[:k]",
        "```", "",
        "## Q. Plain-language conclusion", "",
        f"Development analysis selected a seven-actuator representation containing flexion coordinates from all five "
        "fingers. K=7 slightly exceeded K=9 in the predefined combined development score and reduced ORCA-17 by "
        f"{100.0 * (1-selected_k/17):.1f}%. **The final test has not been evaluated**, so no claim about superiority "
        "over JointAngle-11 or OA-Flex11 is made yet. The next permitted action is a single evaluation using the frozen specification.",
    ]
    (output / "COMPACT_ORCA_SELECTION_EXPLAINED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-only compact ORCA actuator selection.")
    parser.add_argument("--dataset", default="diagnostics/updated_6class_20260820/gesture_sequence_dataset_chinese_dance_6class_after_fix.csv")
    parser.add_argument("--output-dir", default="diagnostics/orca_compact_selection_20260827")
    parser.add_argument("--stage", choices=("select", "final"), default="select")
    parser.add_argument("--outer-seed", type=int, default=20260827)
    parser.add_argument("--inner-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    dataset, output = Path(args.dataset).resolve(), Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "final":
        run_final_stage(dataset, output, args)
        return
    if (output / "final_test_results.csv").exists():
        raise RuntimeError("Final-test results already exist; refusing to rerun selection.")
    sequence_ids, labels, sequences = dimension_control._load_base(dataset)
    development, final = freeze_outer_split(sequence_ids, labels, output, args.test_size, args.outer_seed)

    development_labels = [labels[index] for index in development]
    full_scores, correlation = actuator_scores(
        [sequences["corrected17"][index] for index in development],
        [sequences["optimized_action17"][index] for index in development],
        development_labels,
    )
    validation_rows, inner_rankings = development_validation(
        sequence_ids, labels, sequences, development, args.repeats, args.shot,
        args.validation_size, args.inner_seed,
    )
    selected_k, dimension_summary = select_k(validation_rows)
    selected_indices = select_semantic_subset(full_scores, selected_k)

    _write_rows(output / "orca_actuator_inventory.csv", _inventory())
    _write_rows(output / "actuator_development_scores.csv", full_scores)
    _write_rows(output / "actuator_redundancy_matrix.csv", _redundancy_rows(correlation))
    _write_rows(output / "compact_candidate_development_results.csv", validation_rows)
    _write_rows(output / "compact_dimension_selection.csv", dimension_summary)
    _write_rows(output / "inner_actuator_rankings.csv", inner_rankings)
    _plot_selection(output, full_scores, correlation, dimension_summary)
    _write_frozen_spec(output, selected_k, selected_indices, dimension_summary, full_scores,
                       len(development), len(final), args)
    _write_teaching_document(output, full_scores, selected_k, selected_indices, dimension_summary,
                             len(development), len(final), args)
    print(f"dataset={dataset}")
    print(f"development_sequences={len(development)} final_test_sequences={len(final)}")
    print(f"selected_k={selected_k}")
    print(f"selected_indices={','.join(map(str, selected_indices))}")
    print(f"frozen_spec={output / 'FINAL_COMPACT_ORCA_SPEC.md'}")
    print("final_test_evaluated=no")


if __name__ == "__main__":
    main()
