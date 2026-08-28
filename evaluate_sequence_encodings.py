from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_joint_angle_baseline as joint_eval
import generate_shot_sweep_figures as sweep
import train_svm as tsvm


ENCODINGS = ("global4", "global5", "pyramid", "resample16")
CLASSIFIERS = ("svm", "knn", "rf", "mlp")


def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    old_time = np.linspace(0.0, 1.0, len(sequence))
    new_time = np.linspace(0.0, 1.0, target_len)
    output = np.empty((target_len, sequence.shape[1]), dtype=np.float32)
    for feature_index in range(sequence.shape[1]):
        output[:, feature_index] = np.interp(
            new_time, old_time, sequence[:, feature_index]
        )
    return output


def encode_sequence(sequence: np.ndarray, encoding: str) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or len(sequence) == 0:
        raise ValueError(f"Expected a non-empty T x D sequence, got {sequence.shape}")

    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0)
    delta = sequence[-1] - sequence[0]
    if encoding == "global4":
        return np.concatenate((mean, std, np.max(sequence, axis=0), delta)).astype(np.float32)
    if encoding == "global5":
        return np.concatenate(
            (mean, std, np.min(sequence, axis=0), np.max(sequence, axis=0), delta)
        ).astype(np.float32)
    if encoding == "pyramid":
        descriptors: list[np.ndarray] = []
        for bins in (1, 2, 4):
            for segment in np.array_split(sequence, bins):
                descriptors.extend((np.mean(segment, axis=0), np.std(segment, axis=0)))
        descriptors.append(delta)
        return np.concatenate(descriptors).astype(np.float32)
    if encoding == "resample16":
        return resample_sequence(sequence, 16).reshape(-1).astype(np.float32)
    raise ValueError(f"Unsupported sequence encoding: {encoding}")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    dataset: Path,
    shot: int,
    repeats: int,
    test_size: float,
    random_state: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sequence_ids, labels, sequences, _ = joint_eval._load(dataset)
    splits, manifest = sweep._build_nested_splits(
        sequence_ids, labels, (shot,), repeats, test_size, random_state
    )
    rows: list[dict[str, object]] = []

    for repeat in range(repeats):
        seed = random_state + repeat
        train_indices = splits[(repeat, shot)]["train"]
        test_indices = splits[(repeat, shot)]["test"]
        y_train = [labels[index] for index in train_indices]
        y_test = [labels[index] for index in test_indices]

        representation_sequences: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        for feature_set in ("raw", "joint_angle", "corrected", "optimized_action", "optimized_full"):
            representation_sequences[feature_set] = (
                [sequences[feature_set][index] for index in train_indices],
                [sequences[feature_set][index] for index in test_indices],
            )

        raw_train, raw_test = representation_sequences["raw"]
        for dimensions in (11, 17):
            representation_sequences[f"raw_pca{dimensions}"] = tsvm._project_sequences_with_pca(
                raw_train, raw_test, dimensions, seed
            )

        for feature_set in joint_eval.FEATURE_SETS:
            train_sequences, test_sequences = representation_sequences[feature_set]
            for encoding in ENCODINGS:
                x_train = np.stack([encode_sequence(value, encoding) for value in train_sequences])
                x_test = np.stack([encode_sequence(value, encoding) for value in test_sequences])
                for classifier in CLASSIFIERS:
                    model = tsvm._build_model(sweep._model_args(classifier), seed)
                    model.fit(x_train, y_train)
                    prediction = model.predict(x_test)
                    rows.append(
                        {
                            "repeat": repeat,
                            "shot": shot,
                            "classifier": classifier,
                            "feature_set": feature_set,
                            "encoding": encoding,
                            "num_features": int(x_train.shape[1]),
                            "accuracy": float(accuracy_score(y_test, prediction)),
                            "macro_f1": float(
                                f1_score(y_test, prediction, average="macro", zero_division=0)
                            ),
                        }
                    )
    return rows, manifest


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["classifier"]), str(row["feature_set"]), str(row["encoding"]))].append(row)
    output: list[dict[str, object]] = []
    for (classifier, feature_set, encoding), values in sorted(groups.items()):
        accuracy = np.asarray([float(value["accuracy"]) for value in values])
        macro_f1 = np.asarray([float(value["macro_f1"]) for value in values])
        output.append(
            {
                "classifier": classifier,
                "feature_set": feature_set,
                "encoding": encoding,
                "num_features": values[0]["num_features"],
                "repeats": len(values),
                "accuracy_mean": float(np.mean(accuracy)),
                "accuracy_std": float(np.std(accuracy)),
                "macro_f1_mean": float(np.mean(macro_f1)),
                "macro_f1_std": float(np.std(macro_f1)),
            }
        )
    return output


def paired_statistics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    lookup = {
        (
            int(row["repeat"]),
            str(row["classifier"]),
            str(row["feature_set"]),
            str(row["encoding"]),
        ): row
        for row in rows
    }
    repeats = sorted({int(row["repeat"]) for row in rows})
    comparisons: list[tuple[str, str, str, str]] = []
    for classifier in CLASSIFIERS:
        for feature_set in joint_eval.FEATURE_SETS:
            comparisons.append((classifier, feature_set, "resample16", "global4"))
        for encoding in ENCODINGS:
            comparisons.extend(
                (
                    (classifier, encoding, "optimized_action", "joint_angle"),
                    (classifier, encoding, "optimized_action", "corrected"),
                )
            )

    output: list[dict[str, object]] = []
    for comparison in comparisons:
        classifier = comparison[0]
        if comparison[1] in joint_eval.FEATURE_SETS:
            feature_set, first, second = comparison[1], comparison[2], comparison[3]
            first_values = [
                float(lookup[(repeat, classifier, feature_set, first)]["accuracy"])
                for repeat in repeats
            ]
            second_values = [
                float(lookup[(repeat, classifier, feature_set, second)]["accuracy"])
                for repeat in repeats
            ]
            label = f"{first}_minus_{second}"
            scope = feature_set
        else:
            encoding, first, second = comparison[1], comparison[2], comparison[3]
            first_values = [
                float(lookup[(repeat, classifier, first, encoding)]["accuracy"])
                for repeat in repeats
            ]
            second_values = [
                float(lookup[(repeat, classifier, second, encoding)]["accuracy"])
                for repeat in repeats
            ]
            label = f"{first}_minus_{second}"
            scope = encoding
        differences = np.asarray(first_values) - np.asarray(second_values)
        p_value = float("nan")
        if wilcoxon is not None and np.any(differences != 0):
            p_value = float(wilcoxon(differences).pvalue)
        output.append(
            {
                "classifier": classifier,
                "scope": scope,
                "comparison": label,
                "mean_difference": float(np.mean(differences)),
                "ci95_difference": float(
                    1.96 * np.std(differences, ddof=1) / np.sqrt(len(differences))
                ),
                "wilcoxon_p": p_value,
            }
        )
    return output


def plot_summary(summary: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_encodings = {
        "global4": "Global statistics",
        "global5": "+ minimum",
        "pyramid": "Temporal pyramid",
        "resample16": "Resampled trajectory",
    }
    display_features = {
        "joint_angle": "JointAngle-11",
        "corrected": "Corrected-17",
        "optimized_action": "Optimized Action-17",
    }
    colors = {
        "joint_angle": "#ECA82C",
        "corrected": "#54A24B",
        "optimized_action": "#F58518",
    }
    lookup = {
        (str(row["classifier"]), str(row["feature_set"]), str(row["encoding"])): row
        for row in summary
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharey=True)
    x = np.arange(len(ENCODINGS))
    width = 0.24
    for ax, classifier in zip(axes.flat, CLASSIFIERS, strict=True):
        for offset_index, feature_set in enumerate(display_features):
            values = [
                float(lookup[(classifier, feature_set, encoding)]["accuracy_mean"])
                for encoding in ENCODINGS
            ]
            errors = [
                1.96
                * float(lookup[(classifier, feature_set, encoding)]["accuracy_std"])
                / np.sqrt(float(lookup[(classifier, feature_set, encoding)]["repeats"]))
                for encoding in ENCODINGS
            ]
            ax.bar(
                x + (offset_index - 1) * width,
                values,
                width,
                yerr=errors,
                capsize=3,
                color=colors[feature_set],
                label=display_features[feature_set],
            )
        ax.set_title(sweep.DISPLAY_CLASSIFIERS[classifier], fontweight="bold")
        ax.set_xticks(x, [display_encodings[value] for value in ENCODINGS], rotation=18, ha="right")
        ax.set_ylim(0.55, 0.90)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylabel("Accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Sequence Encoding Ablation (3-shot, 95% CI)", fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare order-free and order-aware sequence encodings on matched splits."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--existing-per-repeat",
        default="",
        help="Reuse an existing per-repeat CSV to regenerate summaries and figures without retraining.",
    )
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if args.existing_per_repeat:
        with Path(args.existing_per_repeat).resolve().open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        manifest: list[dict[str, object]] = []
    else:
        rows, manifest = evaluate(
            Path(args.dataset).resolve(),
            args.shot,
            args.repeats,
            args.test_size,
            args.random_state,
        )
    summary = summarize(rows)
    paired = paired_statistics(rows)
    _write_rows(output_dir / "sequence_encoding_per_repeat.csv", rows)
    _write_rows(output_dir / "sequence_encoding_summary.csv", summary)
    _write_rows(output_dir / "sequence_encoding_paired_stats.csv", paired)
    if manifest:
        _write_rows(output_dir / "sequence_encoding_split_manifest.csv", manifest)
    plot_summary(summary, output_dir / "sequence_encoding_ablation.png")
    print(f"output_dir={output_dir}")
    print(f"rows={len(rows)} shot={args.shot} repeats={args.repeats}")
    for row in summary:
        if row["feature_set"] in {"joint_angle", "corrected", "optimized_action"}:
            print(
                f"{row['classifier']:>4} {row['feature_set']:<16} {row['encoding']:<10} "
                f"D={row['num_features']:<4} accuracy={row['accuracy_mean']:.4f} "
                f"macro_f1={row['macro_f1_mean']:.4f}"
            )


if __name__ == "__main__":
    main()
