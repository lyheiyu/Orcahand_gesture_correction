import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


CLASSIFIER_ORDER = ["svm", "knn", "rf", "mlp"]
STRUCTURED_FEATURES = ["raw", "corrected", "optimized_action", "optimized_full"]
SMOOTHING_FEATURES = ["raw", "moving_average_raw", "savgol_raw", "oneeuro_raw", "kalman_raw", "optimized_full"]
PCA_APPENDIX_FEATURES = ["raw", "raw_pca12", "raw_pca17", "corrected", "optimized_action", "optimized_full"]

SMOOTHING_CLASSIFICATION_ROWS = [
    {"method": "Raw", "accuracy_mean": 0.593750, "macro_f1_mean": 0.541780, "cohen_kappa_mean": 0.368823},
    {"method": "Moving Average", "accuracy_mean": 0.606200, "macro_f1_mean": 0.557700, "cohen_kappa_mean": 0.389300},
    {"method": "Savitzky-Golay", "accuracy_mean": 0.612500, "macro_f1_mean": 0.567100, "cohen_kappa_mean": 0.397000},
    {"method": "One-Euro", "accuracy_mean": 0.637500, "macro_f1_mean": 0.589700, "cohen_kappa_mean": 0.436500},
    {"method": "Kalman", "accuracy_mean": 0.625000, "macro_f1_mean": 0.581500, "cohen_kappa_mean": 0.419000},
    {"method": "Optimized Full", "accuracy_mean": 0.762500, "macro_f1_mean": 0.747976, "cohen_kappa_mean": 0.637073},
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _latest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("classifier", ""), row["feature_set"])
        latest[key] = row
    return list(latest.values())


def _metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def _plot_grouped_metrics(
    rows: list[dict[str, object]],
    labels: list[str],
    metrics: list[tuple[str, str]],
    output_path: Path,
    title: str,
    ylabel: str = "Mean score across repeats",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(labels))
    width = 0.22 if len(metrics) <= 3 else 0.15
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2.0) * width
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]

    fig, ax = plt.subplots(figsize=(11.2, 5.4), dpi=180)
    containers = []
    for index, (metric_key, metric_label) in enumerate(metrics):
        values = np.asarray([float(row[metric_key]) for row in rows], dtype=np.float64)
        bars = ax.bar(
            x + offsets[index],
            values,
            width,
            label=metric_label,
            color=colors[index % len(colors)],
        )
        containers.append(bars)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.legend(frameon=False, ncol=len(metrics), loc="upper center")
    ax.grid(axis="y", alpha=0.25)

    for bars in containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, rotation=90)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_classifier_accuracy(
    suite_rows: list[dict[str, str]],
    output_path: Path,
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_classifier: dict[str, dict[str, dict[str, str]]] = {}
    for row in _latest_rows(suite_rows):
        classifier = row.get("classifier", "")
        rows_by_classifier.setdefault(classifier, {})[row["feature_set"]] = row

    table_rows: list[dict[str, object]] = []
    for classifier in CLASSIFIER_ORDER:
        table_row: dict[str, object] = {"classifier": classifier.upper()}
        for feature_set in STRUCTURED_FEATURES:
            table_row[feature_set] = _metric(rows_by_classifier[classifier][feature_set], "accuracy_mean")
        table_rows.append(table_row)

    x = np.arange(len(CLASSIFIER_ORDER))
    width = 0.18
    offsets = (np.arange(len(STRUCTURED_FEATURES)) - (len(STRUCTURED_FEATURES) - 1) / 2.0) * width
    colors = {
        "raw": "#9D755D",
        "corrected": "#4C78A8",
        "optimized_action": "#54A24B",
        "optimized_full": "#F58518",
    }

    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=180)
    containers = []
    for feature_index, feature_set in enumerate(STRUCTURED_FEATURES):
        values = [float(row[feature_set]) for row in table_rows]
        bars = ax.bar(
            x + offsets[feature_index],
            values,
            width,
            label=feature_set,
            color=colors[feature_set],
        )
        containers.append(bars)

    ax.set_title("Classifier Comparison Across Structured Representations", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([row["classifier"] for row in table_rows])
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    for bars in containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, rotation=90)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return table_rows


def _plot_feature_family_across_classifiers(
    rows: list[dict[str, str]],
    feature_order: list[str],
    output_path: Path,
    title: str,
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_classifier: dict[str, dict[str, dict[str, str]]] = {}
    for row in _latest_rows(rows):
        classifier = row.get("classifier", "")
        rows_by_classifier.setdefault(classifier, {})[row["feature_set"]] = row

    present_features = []
    for feature in feature_order:
        if any(feature in rows_by_classifier.get(classifier, {}) for classifier in CLASSIFIER_ORDER):
            present_features.append(feature)

    table_rows: list[dict[str, object]] = []
    for classifier in CLASSIFIER_ORDER:
        if classifier not in rows_by_classifier:
            continue
        table_row: dict[str, object] = {"classifier": classifier.upper()}
        for feature_set in present_features:
            if feature_set in rows_by_classifier[classifier]:
                table_row[feature_set] = _metric(rows_by_classifier[classifier][feature_set], "accuracy_mean")
        table_rows.append(table_row)

    if not table_rows or not present_features:
        return []

    x = np.arange(len(table_rows))
    width = min(0.8 / max(len(present_features), 1), 0.18)
    offsets = (np.arange(len(present_features)) - (len(present_features) - 1) / 2.0) * width
    palette = ["#9D755D", "#4C78A8", "#E45756", "#72B7B2", "#B279A2", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(12.0, 5.6), dpi=180)
    containers = []
    for feature_index, feature_set in enumerate(present_features):
        values = [float(row.get(feature_set, np.nan)) for row in table_rows]
        bars = ax.bar(
            x + offsets[feature_index],
            values,
            width,
            label=feature_set,
            color=palette[feature_index % len(palette)],
        )
        containers.append(bars)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([row["classifier"] for row in table_rows])
    ax.legend(frameon=False, loc="upper left", ncol=min(3, len(present_features)))
    ax.grid(axis="y", alpha=0.25)

    for bars in containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, rotation=90)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return table_rows


def _build_rf_representation_rows(
    suite_rows: list[dict[str, str]],
    pca_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    suite_latest = {
        (row.get("classifier", ""), row["feature_set"]): row
        for row in _latest_rows(suite_rows)
    }
    pca_by_classifier = {
        (row["classifier"], row["feature_set"]): row
        for row in pca_rows
    }

    rf_rows = []
    ordered_keys = [
        ("raw", "Raw"),
        ("raw_pca12", "Best PCA (12D)"),
        ("raw_pca17", "PCA-17"),
        ("corrected", "Corrected"),
        ("optimized_action", "Optimized Action"),
        ("optimized_full", "Optimized Full"),
    ]
    for feature_key, display_name in ordered_keys:
        row = pca_by_classifier[("rf", feature_key)] if feature_key.startswith("raw_pca") else suite_latest[("rf", feature_key)]
        rf_rows.append(
            {
                "method": display_name,
                "accuracy_mean": _metric(row, "accuracy_mean"),
                "macro_f1_mean": _metric(row, "macro_f1_mean"),
                "cohen_kappa_mean": _metric(row, "cohen_kappa_mean"),
            }
        )
    return rf_rows


def _build_jitter_rows(
    jitter_rows: list[dict[str, str]],
    ordered_feature_sets: list[tuple[str, str]],
) -> list[dict[str, object]]:
    jitter_by_feature = {row["feature_set"]: row for row in jitter_rows}
    rows: list[dict[str, object]] = []
    for feature_set, label in ordered_feature_sets:
        if feature_set not in jitter_by_feature:
            continue
        row = jitter_by_feature[feature_set]
        rows.append(
            {
                "method": label,
                "velocity_mean": float(row["velocity_mean"]),
                "acceleration_mean": float(row["acceleration_mean"]),
            }
        )
    return rows


def _plot_cm_manifest(figures_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    search_targets = [
        ("best_cm_optimized_action.png", ["rf", "optimized_action"]),
        ("best_cm_best_pca.png", ["rf", "raw_pca12"]),
        ("best_cm_raw.png", ["rf", "raw"]),
    ]

    png_paths = list(figures_dir.rglob("*.png"))
    manifest_rows: list[dict[str, object]] = []

    for output_name, keywords in search_targets:
        matched = None
        for path in png_paths:
            name_lower = path.name.lower()
            if all(keyword in name_lower for keyword in keywords):
                matched = path
                break
        copied_to = ""
        if matched is not None:
            copied_path = output_dir / output_name
            shutil.copyfile(matched, copied_path)
            copied_to = str(copied_path)
        manifest_rows.append(
            {
                "target": output_name,
                "keywords": ",".join(keywords),
                "found_source": str(matched) if matched is not None else "",
                "copied_to": copied_to,
            }
        )
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready summary figures and tables.")
    parser.add_argument("--suite-csv", default="figures/classifier_suite_v2/classification_suite_v2.csv")
    parser.add_argument("--pca-csv", default="figures/pca_sweep_v2/pca_sweep_summary.csv")
    parser.add_argument("--smoothing-suite-csv", default="")
    parser.add_argument("--jitter-csv", default="figures/jitter_v2.csv")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--output-dir", default="figures/paper_summary")
    args = parser.parse_args()

    suite_path = Path(args.suite_csv).resolve()
    pca_path = Path(args.pca_csv).resolve()
    smoothing_suite_path = Path(args.smoothing_suite_csv).resolve() if args.smoothing_suite_csv else None
    jitter_path = Path(args.jitter_csv).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_rows = _read_csv(suite_path)
    pca_rows = _read_csv(pca_path)
    smoothing_suite_rows = _read_csv(smoothing_suite_path) if smoothing_suite_path and smoothing_suite_path.exists() else []
    jitter_rows = _read_csv(jitter_path) if jitter_path.exists() else []

    classifier_table_rows = _plot_classifier_accuracy(
        suite_rows,
        output_dir / "classifier_accuracy_structured.png",
    )
    _write_csv(
        output_dir / "classifier_accuracy_structured.csv",
        ["classifier"] + STRUCTURED_FEATURES,
        classifier_table_rows,
    )

    rf_representation_rows = _build_rf_representation_rows(suite_rows, pca_rows)
    _write_csv(
        output_dir / "representation_comparison.csv",
        ["method", "accuracy_mean", "macro_f1_mean", "cohen_kappa_mean"],
        rf_representation_rows,
    )
    _plot_grouped_metrics(
        rf_representation_rows,
        [str(row["method"]) for row in rf_representation_rows],
        [("accuracy_mean", "Accuracy"), ("macro_f1_mean", "Macro-F1"), ("cohen_kappa_mean", "Kappa")],
        output_dir / "representation_comparison.png",
        "RandomForest Comparison: Raw, PCA, and ORCA/MuJoCo Representations",
    )

    _write_csv(
        output_dir / "smoothing_comparison.csv",
        ["method", "accuracy_mean", "macro_f1_mean", "cohen_kappa_mean"],
        SMOOTHING_CLASSIFICATION_ROWS,
    )
    _plot_grouped_metrics(
        SMOOTHING_CLASSIFICATION_ROWS,
        [str(row["method"]) for row in SMOOTHING_CLASSIFICATION_ROWS],
        [("accuracy_mean", "Accuracy"), ("macro_f1_mean", "Macro-F1"), ("cohen_kappa_mean", "Kappa")],
        output_dir / "smoothing_comparison.png",
        "RandomForest Comparison: Landmark-space Smoothing vs Optimized Full",
    )

    pca_appendix_rows = _plot_feature_family_across_classifiers(
        pca_rows,
        PCA_APPENDIX_FEATURES,
        output_dir / "appendix_pca_across_classifiers.png",
        "Appendix: PCA and Structured Representations Across Classifiers",
    )
    if pca_appendix_rows:
        _write_csv(
            output_dir / "appendix_pca_across_classifiers.csv",
            ["classifier"] + [feature for feature in PCA_APPENDIX_FEATURES if any(feature in row for row in pca_appendix_rows)],
            pca_appendix_rows,
        )

    smoothing_appendix_rows = _plot_feature_family_across_classifiers(
        smoothing_suite_rows,
        SMOOTHING_FEATURES,
        output_dir / "appendix_smoothing_across_classifiers.png",
        "Appendix: Smoothing Baselines Across Classifiers",
    )
    if smoothing_appendix_rows:
        _write_csv(
            output_dir / "appendix_smoothing_across_classifiers.csv",
            ["classifier"] + [feature for feature in SMOOTHING_FEATURES if any(feature in row for row in smoothing_appendix_rows)],
            smoothing_appendix_rows,
        )

    if jitter_rows:
        actuator_rows = _build_jitter_rows(
            jitter_rows,
            [("corrected", "Corrected"), ("optimized_action", "Optimized Action")],
        )
        if actuator_rows:
            _write_csv(
                output_dir / "jitter_actuator_space.csv",
                ["method", "velocity_mean", "acceleration_mean"],
                actuator_rows,
            )
            _plot_grouped_metrics(
                actuator_rows,
                [str(row["method"]) for row in actuator_rows],
                [("velocity_mean", "Velocity"), ("acceleration_mean", "Acceleration")],
                output_dir / "jitter_actuator_space.png",
                "Actuator-space Temporal Stability",
                ylabel="Mean temporal difference",
            )

        landmark_rows = _build_jitter_rows(
            jitter_rows,
            [("raw", "Raw"), ("optimized_full", "Optimized Full")],
        )
        if landmark_rows:
            _write_csv(
                output_dir / "jitter_landmark_space.csv",
                ["method", "velocity_mean", "acceleration_mean"],
                landmark_rows,
            )
            _plot_grouped_metrics(
                landmark_rows,
                [str(row["method"]) for row in landmark_rows],
                [("velocity_mean", "Velocity"), ("acceleration_mean", "Acceleration")],
                output_dir / "jitter_landmark_space.png",
                "Landmark-space Temporal Stability",
                ylabel="Mean temporal difference",
            )

    cm_rows = _plot_cm_manifest(figures_dir, output_dir)
    _write_csv(
        output_dir / "best_confusion_matrix_manifest.csv",
        ["target", "keywords", "found_source", "copied_to"],
        cm_rows,
    )

    summary_rows = [
        {"artifact": "smoothing_comparison", "path": str(output_dir / "smoothing_comparison.png")},
        {"artifact": "representation_comparison", "path": str(output_dir / "representation_comparison.png")},
        {"artifact": "classifier_accuracy_structured", "path": str(output_dir / "classifier_accuracy_structured.png")},
        {"artifact": "appendix_pca_across_classifiers", "path": str(output_dir / "appendix_pca_across_classifiers.png")},
        {"artifact": "appendix_smoothing_across_classifiers", "path": str(output_dir / "appendix_smoothing_across_classifiers.png")},
        {"artifact": "jitter_actuator_space", "path": str(output_dir / "jitter_actuator_space.png")},
        {"artifact": "jitter_landmark_space", "path": str(output_dir / "jitter_landmark_space.png")},
        {"artifact": "best_confusion_matrix_manifest", "path": str(output_dir / "best_confusion_matrix_manifest.csv")},
    ]
    _write_csv(output_dir / "artifact_index.csv", ["artifact", "path"], summary_rows)

    print(f"paper_figures_dir={output_dir}")


if __name__ == "__main__":
    main()
