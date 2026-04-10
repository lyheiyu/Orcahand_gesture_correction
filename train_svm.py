import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _load_dataset(path: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        meta_fields = {"label", "sequence_id", "frame_id", "timestamp_sec"}
        feature_names = [name for name in fieldnames if name not in meta_fields]
        rows_meta: list[dict[str, str]] = []
        rows: list[list[float]] = []
        for row in reader:
            rows_meta.append(
                {
                    "label": row["label"],
                    "sequence_id": row.get("sequence_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "timestamp_sec": row.get("timestamp_sec", ""),
                }
            )
            rows.append([float(row[name]) for name in feature_names])
    return rows_meta, feature_names, np.asarray(rows, dtype=np.float32)


def _select_features(feature_names: list[str], features: np.ndarray, feature_set: str) -> tuple[list[str], np.ndarray]:
    if feature_set == "all":
        indices = list(range(len(feature_names)))
    elif feature_set == "optimized":
        indices = [i for i, name in enumerate(feature_names) if name.startswith("optimized_")]
    else:
        prefix = f"{feature_set}_"
        indices = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
    selected_names = [feature_names[i] for i in indices]
    selected_features = features[:, indices]
    return selected_names, selected_features


def _few_shot_subset(
    x_train: np.ndarray,
    y_train: list[str],
    shots_per_class: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, list[str]]:
    labels = np.asarray(y_train)
    keep_indices: list[int] = []
    for label in sorted(set(y_train)):
        class_indices = np.where(labels == label)[0]
        if len(class_indices) < shots_per_class:
            raise SystemExit(
                f"Class `{label}` only has {len(class_indices)} training samples, "
                f"cannot keep {shots_per_class} shots per class."
            )
        chosen = rng.choice(class_indices, size=shots_per_class, replace=False)
        keep_indices.extend(int(i) for i in chosen)

    keep_indices = sorted(keep_indices)
    return x_train[keep_indices], [y_train[i] for i in keep_indices]


def _sequence_aggregate(
    row_meta: list[dict[str, str]],
    features: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[tuple[int, float, np.ndarray]]] = {}
    sequence_labels: dict[str, str] = {}

    for meta, feature_row in zip(row_meta, features, strict=True):
        sequence_id = meta.get("sequence_id") or ""
        if not sequence_id:
            sequence_id = f"single_{len(grouped)}_{meta['label']}"
        frame_id = int(meta.get("frame_id") or 0)
        timestamp_sec = float(meta.get("timestamp_sec") or 0.0)
        grouped.setdefault(sequence_id, []).append((frame_id, timestamp_sec, feature_row))
        sequence_labels[sequence_id] = meta["label"]

    labels: list[str] = []
    aggregated_rows: list[np.ndarray] = []
    for sequence_id in sorted(grouped):
        entries = sorted(grouped[sequence_id], key=lambda item: (item[0], item[1]))
        sequence = np.stack([item[2] for item in entries], axis=0)
        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0)
        maxv = np.max(sequence, axis=0)
        start = sequence[0]
        end = sequence[-1]
        delta = end - start
        aggregated = np.concatenate([mean, std, maxv, delta], axis=0).astype(np.float32)
        labels.append(sequence_labels[sequence_id])
        aggregated_rows.append(aggregated)

    return labels, np.stack(aggregated_rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SVM baseline on gesture CSV features.")
    parser.add_argument("--dataset", default="gesture_dataset.csv", help="CSV created by collect_gesture_dataset.py")
    parser.add_argument(
        "--feature-set",
        choices=[
            "raw",
            "geom",
            "corrected",
            "optimized",
            "optimized_action",
            "optimized_sparse",
            "optimized_full",
            "optimized_loss",
            "all",
        ],
        default="corrected",
        help="Feature subset to use for training.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--c", type=float, default=5.0, help="SVM C parameter.")
    parser.add_argument("--gamma", default="scale", help="SVM gamma parameter.")
    parser.add_argument(
        "--shots-per-class",
        type=int,
        default=0,
        help="If > 0, keep only this many training samples per class after the train/test split.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat training with different random seeds and report mean/std accuracy.",
    )
    parser.add_argument(
        "--sequence-mode",
        action="store_true",
        help="Group rows by sequence_id and aggregate each full sequence into one sample.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    row_meta, feature_names, features = _load_dataset(dataset_path)
    labels = [meta["label"] for meta in row_meta]
    if len(set(labels)) < 2:
        raise SystemExit("Need at least two gesture classes in the dataset.")
    if args.shots_per_class < 0:
        raise SystemExit("--shots-per-class must be >= 0.")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1.")

    selected_names, selected_features = _select_features(feature_names, features, args.feature_set)
    if args.sequence_mode:
        labels, selected_features = _sequence_aggregate(row_meta, selected_features)
        selected_names = (
            [f"mean_{name}" for name in selected_names]
            + [f"std_{name}" for name in selected_names]
            + [f"max_{name}" for name in selected_names]
            + [f"delta_{name}" for name in selected_names]
        )
    accuracies: list[float] = []
    last_report = ""
    last_confusion = None
    last_num_train = 0
    last_num_test = 0

    for repeat_index in range(args.repeats):
        seed = args.random_state + repeat_index
        x_train, x_test, y_train, y_test = train_test_split(
            selected_features,
            labels,
            test_size=args.test_size,
            random_state=seed,
            stratify=labels,
        )

        if args.shots_per_class > 0:
            rng = np.random.RandomState(seed)
            x_train, y_train = _few_shot_subset(x_train, y_train, args.shots_per_class, rng)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=args.c, gamma=args.gamma)),
            ]
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracies.append(float(np.mean(np.asarray(y_pred) == np.asarray(y_test))))
        last_report = classification_report(y_test, y_pred, digits=4)
        last_confusion = confusion_matrix(y_test, y_pred)
        last_num_train = len(y_train)
        last_num_test = len(y_test)

    print(f"dataset={dataset_path}")
    print(f"feature_set={args.feature_set}")
    print(f"num_features={len(selected_names)}")
    print(f"num_train={last_num_train} num_test={last_num_test}")
    print(f"sequence_mode={'yes' if args.sequence_mode else 'no'}")
    if args.shots_per_class > 0:
        print(f"shots_per_class={args.shots_per_class}")
    print(f"repeats={args.repeats}")
    print(f"accuracy_mean={np.mean(accuracies):.4f}")
    print(f"accuracy_std={np.std(accuracies):.4f}")
    print()
    print(last_report)
    print("Confusion matrix:")
    print(last_confusion)


if __name__ == "__main__":
    main()
