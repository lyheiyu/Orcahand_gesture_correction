from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_joint_angle_baseline as baseline
import generate_shot_sweep_figures as sweep
import train_svm as tsvm


CLASSIFIERS = ("svm", "knn", "mlp")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parameter_grid(classifier: str) -> dict[str, list[object]]:
    if classifier == "svm":
        return {
            "model__C": [0.1, 1.0, 5.0, 10.0, 50.0],
            "model__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        }
    if classifier == "knn":
        return {
            "model__n_neighbors": [1, 3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
        }
    if classifier == "mlp":
        return {
            "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "model__alpha": [1e-4, 1e-3, 1e-2],
        }
    raise ValueError(f"Unsupported classifier: {classifier}")


def _matrices(
    sequences: dict[str, list[np.ndarray]],
    descriptors: dict[str, np.ndarray],
    train_indices: list[int],
    test_indices: list[int],
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for feature_set in ("raw", "joint_angle", "corrected", "optimized_action", "optimized_full"):
        matrix = descriptors[feature_set]
        output[feature_set] = (matrix[train_indices], matrix[test_indices])

    raw_train = [sequences["raw"][index] for index in train_indices]
    raw_test = [sequences["raw"][index] for index in test_indices]
    for dimensions in (11, 17):
        projected_train, projected_test = tsvm._project_sequences_with_pca(
            raw_train, raw_test, dimensions, seed
        )
        output[f"raw_pca{dimensions}"] = (
            np.stack([tsvm._aggregate_sequence_array(value) for value in projected_train]),
            np.stack([tsvm._aggregate_sequence_array(value) for value in projected_test]),
        )
    return output


def evaluate(
    dataset: Path,
    shot: int,
    repeats: int,
    test_size: float,
    random_state: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sequence_ids, labels, sequences, descriptors = baseline._load(dataset)
    splits, manifest = sweep._build_nested_splits(
        sequence_ids, labels, (shot,), repeats, test_size, random_state
    )
    rows: list[dict[str, object]] = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for repeat in range(repeats):
        seed = random_state + repeat
        train_indices = splits[(repeat, shot)]["train"]
        test_indices = splits[(repeat, shot)]["test"]
        y_train = np.asarray([labels[index] for index in train_indices])
        y_test = np.asarray([labels[index] for index in test_indices])
        matrices = _matrices(sequences, descriptors, train_indices, test_indices, seed)
        inner_cv = StratifiedKFold(n_splits=shot, shuffle=True, random_state=seed)

        for feature_set in baseline.FEATURE_SETS:
            x_train, x_test = matrices[feature_set]
            for classifier in CLASSIFIERS:
                fixed_model = tsvm._build_model(sweep._model_args(classifier), seed)
                fixed_model.fit(x_train, y_train)
                fixed_prediction = fixed_model.predict(x_test)
                rows.append(
                    {
                        "repeat": repeat,
                        "shot": shot,
                        "classifier": classifier,
                        "feature_set": feature_set,
                        "training": "fixed",
                        "accuracy": float(accuracy_score(y_test, fixed_prediction)),
                        "macro_f1": float(f1_score(y_test, fixed_prediction, average="macro", zero_division=0)),
                        "selected_parameters": "",
                    }
                )

                search = GridSearchCV(
                    estimator=tsvm._build_model(sweep._model_args(classifier), seed),
                    param_grid=_parameter_grid(classifier),
                    scoring="f1_macro",
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True,
                    error_score="raise",
                )
                search.fit(x_train, y_train)
                tuned_prediction = search.predict(x_test)
                rows.append(
                    {
                        "repeat": repeat,
                        "shot": shot,
                        "classifier": classifier,
                        "feature_set": feature_set,
                        "training": "nested_tuned",
                        "accuracy": float(accuracy_score(y_test, tuned_prediction)),
                        "macro_f1": float(f1_score(y_test, tuned_prediction, average="macro", zero_division=0)),
                        "selected_parameters": json.dumps(search.best_params_, sort_keys=True),
                    }
                )
    return rows, manifest


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["classifier"]), str(row["feature_set"]), str(row["training"]))].append(row)

    output: list[dict[str, object]] = []
    for (classifier, feature_set, training), values in sorted(groups.items()):
        accuracies = np.asarray([float(value["accuracy"]) for value in values])
        macro_f1 = np.asarray([float(value["macro_f1"]) for value in values])
        parameters = Counter(
            str(value["selected_parameters"])
            for value in values
            if value["selected_parameters"]
        )
        output.append(
            {
                "classifier": classifier,
                "feature_set": feature_set,
                "training": training,
                "repeats": len(values),
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "macro_f1_mean": float(np.mean(macro_f1)),
                "macro_f1_std": float(np.std(macro_f1)),
                "most_common_parameters": parameters.most_common(1)[0][0] if parameters else "",
                "most_common_parameter_count": parameters.most_common(1)[0][1] if parameters else "",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose fixed versus train-only nested tuning for joint-angle and ORCA representations."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if args.shot < 2:
        raise SystemExit("Nested stratified tuning requires at least 2 shots per class.")

    output_dir = Path(args.output_dir).resolve()
    rows, manifest = evaluate(
        Path(args.dataset).resolve(),
        args.shot,
        args.repeats,
        args.test_size,
        args.random_state,
    )
    summary = summarize(rows)
    _write_rows(output_dir / "tuned_per_repeat.csv", rows)
    _write_rows(output_dir / "tuned_summary.csv", summary)
    _write_rows(output_dir / "tuned_split_manifest.csv", manifest)
    print(f"output_dir={output_dir}")
    print(f"rows={len(rows)} shot={args.shot} repeats={args.repeats}")
    for row in summary:
        if row["feature_set"] in {"joint_angle", "corrected", "optimized_action"}:
            print(
                f"{row['classifier']:>4} {row['feature_set']:<16} {row['training']:<12} "
                f"accuracy={row['accuracy_mean']:.4f} macro_f1={row['macro_f1_mean']:.4f}"
            )


if __name__ == "__main__":
    main()
