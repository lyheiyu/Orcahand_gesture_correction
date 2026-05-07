import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
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
    if selected_features.shape[1] == 0:
        raise SystemExit(f"No columns found for feature set `{feature_set}`.")
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


def _few_shot_index_subset(
    train_indices: np.ndarray,
    labels: list[str],
    shots_per_class: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    train_labels = np.asarray([labels[index] for index in train_indices])
    keep_positions: list[int] = []
    for label in sorted(set(train_labels.tolist())):
        class_positions = np.where(train_labels == label)[0]
        if len(class_positions) < shots_per_class:
            raise SystemExit(
                f"Class `{label}` only has {len(class_positions)} training sequences, "
                f"cannot keep {shots_per_class} shots per class."
            )
        chosen = rng.choice(class_positions, size=shots_per_class, replace=False)
        keep_positions.extend(int(position) for position in chosen)
    keep_positions = sorted(keep_positions)
    return train_indices[keep_positions]


def _compute_scores(y_true: list[str], y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def _mean_std(score_rows: list[dict[str, float]], metric_name: str) -> tuple[float, float]:
    values = np.asarray([row[metric_name] for row in score_rows], dtype=np.float64)
    return float(np.mean(values)), float(np.std(values))


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=180)
    image = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    threshold = 0.5
    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            value = normalized[row_index, col_index]
            count = cm[row_index, col_index]
            color = "white" if value > threshold else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}\n({count})",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _append_results_csv(
    output_path: Path,
    row: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_keys = list(row.keys())
    existing_rows: list[dict[str, str]] = []
    fieldnames = row_keys.copy()

    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        fieldnames = existing_fieldnames.copy()
        for key in row_keys:
            if key not in fieldnames:
                fieldnames.append(key)

    normalized_existing_rows: list[dict[str, object]] = []
    for existing_row in existing_rows:
        normalized_existing_rows.append({key: existing_row.get(key, "") for key in fieldnames})

    normalized_new_row = {key: row.get(key, "") for key in fieldnames}

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_existing_rows)
        writer.writerow(normalized_new_row)


def _build_model(args: argparse.Namespace, seed: int) -> Pipeline:
    if args.classifier == "svm":
        estimator = SVC(kernel="rbf", C=args.c, gamma=args.gamma)
    elif args.classifier == "knn":
        estimator = KNeighborsClassifier(
            n_neighbors=args.knn_neighbors,
            weights=args.knn_weights,
        )
    elif args.classifier == "rf":
        estimator = RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
            random_state=seed,
        )
    elif args.classifier == "mlp":
        hidden_layer_sizes = tuple(int(part.strip()) for part in args.mlp_hidden_sizes.split(",") if part.strip())
        if not hidden_layer_sizes:
            raise SystemExit("--mlp-hidden-sizes must contain at least one integer.")
        estimator = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=args.mlp_alpha,
            learning_rate_init=args.mlp_learning_rate,
            max_iter=args.mlp_max_iter,
            random_state=seed,
        )
    else:
        raise SystemExit(f"Unsupported classifier `{args.classifier}`.")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


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


def _group_sequences(
    row_meta: list[dict[str, str]],
    features: np.ndarray,
) -> tuple[list[str], list[str], list[np.ndarray]]:
    grouped: dict[str, list[tuple[int, float, np.ndarray]]] = {}
    sequence_labels: dict[str, str] = {}

    for index, (meta, feature_row) in enumerate(zip(row_meta, features, strict=True)):
        sequence_id = meta.get("sequence_id") or f"single_{index}_{meta['label']}"
        frame_id = int(meta.get("frame_id") or 0)
        timestamp_sec = float(meta.get("timestamp_sec") or 0.0)
        grouped.setdefault(sequence_id, []).append((frame_id, timestamp_sec, feature_row))
        sequence_labels[sequence_id] = meta["label"]

    sequence_ids: list[str] = []
    labels: list[str] = []
    sequences: list[np.ndarray] = []
    for sequence_id in sorted(grouped):
        entries = sorted(grouped[sequence_id], key=lambda item: (item[0], item[1]))
        sequence = np.stack([item[2] for item in entries], axis=0).astype(np.float32)
        sequence_ids.append(sequence_id)
        labels.append(sequence_labels[sequence_id])
        sequences.append(sequence)
    return sequence_ids, labels, sequences


def _aggregate_sequence_array(sequence: np.ndarray) -> np.ndarray:
    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0)
    maxv = np.max(sequence, axis=0)
    start = sequence[0]
    end = sequence[-1]
    delta = end - start
    return np.concatenate([mean, std, maxv, delta], axis=0).astype(np.float32)


def _project_sequences_with_pca(
    train_sequences: list[np.ndarray],
    test_sequences: list[np.ndarray],
    n_components: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    train_frames = np.concatenate(train_sequences, axis=0)
    scaler = StandardScaler()
    scaler.fit(train_frames)
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(scaler.transform(train_frames))

    def transform(sequence: np.ndarray) -> np.ndarray:
        return pca.transform(scaler.transform(sequence)).astype(np.float32)

    return [transform(sequence) for sequence in train_sequences], [transform(sequence) for sequence in test_sequences]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classifier baseline on gesture CSV features.")
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
    parser.add_argument(
        "--classifier",
        choices=["svm", "knn", "rf", "mlp"],
        default="svm",
        help="Classifier to train.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help="If > 0, apply PCA to the selected frame-level features before classification. In sequence mode, PCA is fit on training frames only.",
    )
    parser.add_argument("--c", type=float, default=5.0, help="SVM C parameter.")
    parser.add_argument("--gamma", default="scale", help="SVM gamma parameter.")
    parser.add_argument("--knn-neighbors", type=int, default=3, help="KNN number of neighbors.")
    parser.add_argument(
        "--knn-weights",
        choices=["uniform", "distance"],
        default="distance",
        help="KNN weighting scheme.",
    )
    parser.add_argument("--rf-estimators", type=int, default=200, help="RandomForest number of trees.")
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=0,
        help="RandomForest max depth. Use 0 for no depth limit.",
    )
    parser.add_argument(
        "--mlp-hidden-sizes",
        default="128,64",
        help="Comma-separated hidden layer sizes for MLP, for example 128,64.",
    )
    parser.add_argument("--mlp-alpha", type=float, default=1e-4, help="MLP L2 regularization strength.")
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3, help="MLP initial learning rate.")
    parser.add_argument("--mlp-max-iter", type=int, default=1200, help="MLP max iterations.")
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
    parser.add_argument(
        "--plot-confusion",
        default="",
        help="Optional output path for a normalized confusion matrix PNG.",
    )
    parser.add_argument(
        "--confusion-title",
        default="",
        help="Optional title for the confusion matrix figure.",
    )
    parser.add_argument(
        "--results-csv",
        default="",
        help="Optional CSV path. Appends one summary row for this run.",
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
    if args.pca_components < 0:
        raise SystemExit("--pca-components must be >= 0.")

    selected_names, selected_features = _select_features(feature_names, features, args.feature_set)
    if args.pca_components > 0 and args.pca_components > selected_features.shape[1]:
        raise SystemExit(
            f"--pca-components={args.pca_components} exceeds selected feature dimension {selected_features.shape[1]}."
        )

    display_feature_set = args.feature_set
    if args.pca_components > 0:
        display_feature_set = f"{args.feature_set}_pca{args.pca_components}"

    sequence_ids: list[str] = []
    sequence_labels: list[str] = []
    sequences: list[np.ndarray] = []
    frame_level_labels = labels
    if args.sequence_mode:
        sequence_ids, sequence_labels, sequences = _group_sequences(row_meta, selected_features)
        labels = sequence_labels

    score_rows: list[dict[str, float]] = []
    last_report = ""
    last_confusion = None
    aggregate_confusion: np.ndarray | None = None
    aggregate_y_true: list[str] = []
    aggregate_y_pred: list[str] = []
    last_num_train = 0
    last_num_test = 0
    class_labels = sorted(set(labels))
    final_num_features = 0

    for repeat_index in range(args.repeats):
        seed = args.random_state + repeat_index
        if args.sequence_mode:
            all_indices = np.arange(len(sequences))
            train_indices, test_indices = train_test_split(
                all_indices,
                test_size=args.test_size,
                random_state=seed,
                stratify=sequence_labels,
            )
            if args.shots_per_class > 0:
                rng = np.random.RandomState(seed)
                train_indices = _few_shot_index_subset(train_indices, sequence_labels, args.shots_per_class, rng)

            train_sequences = [sequences[index] for index in train_indices]
            test_sequences = [sequences[index] for index in test_indices]
            y_train = [sequence_labels[index] for index in train_indices]
            y_test = [sequence_labels[index] for index in test_indices]

            if args.pca_components > 0:
                train_sequences, test_sequences = _project_sequences_with_pca(
                    train_sequences,
                    test_sequences,
                    args.pca_components,
                    seed,
                )

            x_train = np.stack([_aggregate_sequence_array(sequence) for sequence in train_sequences], axis=0)
            x_test = np.stack([_aggregate_sequence_array(sequence) for sequence in test_sequences], axis=0)
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                selected_features,
                frame_level_labels,
                test_size=args.test_size,
                random_state=seed,
                stratify=frame_level_labels,
            )

            if args.shots_per_class > 0:
                rng = np.random.RandomState(seed)
                x_train, y_train = _few_shot_subset(x_train, y_train, args.shots_per_class, rng)

            if args.pca_components > 0:
                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(x_train)
                x_test_scaled = scaler.transform(x_test)
                pca = PCA(n_components=args.pca_components, random_state=seed)
                x_train = pca.fit_transform(x_train_scaled).astype(np.float32)
                x_test = pca.transform(x_test_scaled).astype(np.float32)

        final_num_features = int(x_train.shape[1])
        model = _build_model(args, seed)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        score_rows.append(_compute_scores(y_test, y_pred))
        aggregate_y_true.extend(y_test)
        aggregate_y_pred.extend(str(label) for label in y_pred)
        last_report = classification_report(y_test, y_pred, labels=class_labels, digits=4, zero_division=0)
        last_confusion = confusion_matrix(y_test, y_pred, labels=class_labels)
        if aggregate_confusion is None:
            aggregate_confusion = last_confusion.copy()
        else:
            aggregate_confusion += last_confusion
        last_num_train = len(y_train)
        last_num_test = len(y_test)

    print(f"dataset={dataset_path}")
    print(f"classifier={args.classifier}")
    print(f"feature_set={display_feature_set}")
    print(f"num_features={final_num_features}")
    print(f"num_train={last_num_train} num_test={last_num_test}")
    print(f"sequence_mode={'yes' if args.sequence_mode else 'no'}")
    print(f"pca_components={args.pca_components}")
    if args.shots_per_class > 0:
        print(f"shots_per_class={args.shots_per_class}")
    print(f"repeats={args.repeats}")
    for metric_name in [
        "macro_recall",
        "macro_f1",
        "macro_precision",
        "accuracy",
        "cohen_kappa",
        "weighted_f1",
    ]:
        mean, std = _mean_std(score_rows, metric_name)
        print(f"{metric_name}_mean={mean:.4f}")
        print(f"{metric_name}_std={std:.4f}")
    print()
    print("Aggregate classification report across repeats:")
    print(classification_report(aggregate_y_true, aggregate_y_pred, labels=class_labels, digits=4, zero_division=0))
    print("Last-repeat classification report:")
    print(last_report)
    print("Last-repeat confusion matrix:")
    print(last_confusion)
    print("Aggregate confusion matrix across repeats:")
    print(aggregate_confusion)

    if args.plot_confusion:
        if aggregate_confusion is None:
            raise SystemExit("No confusion matrix was generated.")
        confusion_path = Path(args.plot_confusion).resolve()
        title = args.confusion_title or f"{display_feature_set} aggregate confusion matrix"
        _plot_confusion_matrix(aggregate_confusion, class_labels, title, confusion_path)
        print(f"confusion_plot={confusion_path}")

    if args.results_csv:
        csv_path = Path(args.results_csv).resolve()
        result_row: dict[str, object] = {
            "dataset": str(dataset_path),
            "classifier": args.classifier,
            "feature_set": display_feature_set,
            "feature_set_base": args.feature_set,
            "num_features": final_num_features,
            "num_train": last_num_train,
            "num_test": last_num_test,
            "sequence_mode": "yes" if args.sequence_mode else "no",
            "shots_per_class": args.shots_per_class,
            "repeats": args.repeats,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "pca_components": args.pca_components,
            "c": args.c,
            "gamma": args.gamma,
            "knn_neighbors": args.knn_neighbors,
            "knn_weights": args.knn_weights,
            "rf_estimators": args.rf_estimators,
            "rf_max_depth": args.rf_max_depth,
            "mlp_hidden_sizes": args.mlp_hidden_sizes,
            "mlp_alpha": args.mlp_alpha,
            "mlp_learning_rate": args.mlp_learning_rate,
            "mlp_max_iter": args.mlp_max_iter,
        }
        for metric_name in [
            "macro_recall",
            "macro_f1",
            "macro_precision",
            "accuracy",
            "cohen_kappa",
            "weighted_f1",
        ]:
            mean, std = _mean_std(score_rows, metric_name)
            result_row[f"{metric_name}_mean"] = f"{mean:.6f}"
            result_row[f"{metric_name}_std"] = f"{std:.6f}"
        _append_results_csv(csv_path, result_row)
        print(f"results_csv={csv_path}")


if __name__ == "__main__":
    main()
