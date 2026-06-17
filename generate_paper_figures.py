import argparse
import csv
from pathlib import Path

import numpy as np


CLASSIFIER_ORDER = ["svm", "knn", "rf", "mlp"]
STRUCTURED_FEATURES = ["raw", "corrected", "optimized_action", "optimized_full"]
SMOOTHING_FEATURES = [
    "raw",
    "moving_average_raw",
    "savgol_raw",
    "oneeuro_raw",
    "kalman_raw",
    "optimized_full",
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
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(labels))
    width = 0.22 if len(metrics) <= 3 else 0.15
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2.0) * width
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]

    fig, ax = plt.subplots(figsize=(10.8, 5.2), dpi=180)
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
    ax.set_ylabel("Mean score across repeats")
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
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_classifier: dict[str, dict[str, dict[str, str]]] = {}
    for row in _latest_rows(suite_rows):
        classifier = row.get("classifier", "")
        rows_by_classifier.setdefault(classifier, {})[row["feature_set"]] = row

    x = np.arange(len(CLASSIFIER_ORDER))
    width = 0.18
    offsets = (np.arange(len(STRUCTURED_FEATURES)) - (len(STRUCTURED_FEATURES) - 1) / 2.0) * width
    colors = {
        "raw": "#9D755D",
        "corrected": "#4C78A8",
        "optimized_action": "#54A24B",
        "optimized_full": "#F58518",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
    containers = []
    for feature_index, feature_set in enumerate(STRUCTURED_FEATURES):
        values = []
        for classifier in CLASSIFIER_ORDER:
            row = rows_by_classifier[classifier][feature_set]
            values.append(_metric(row, "accuracy_mean"))
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
    ax.set_xticklabels([name.upper() for name in CLASSIFIER_ORDER])
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    for bars in containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, rotation=90)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_rf_method_rows(
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
        if feature_key.startswith("raw_pca"):
            row = pca_by_classifier[("rf", feature_key)]
        else:
            row = suite_latest[("rf", feature_key)]
        rf_rows.append(
            {
                "method": display_name,
                "accuracy_mean": _metric(row, "accuracy_mean"),
                "macro_f1_mean": _metric(row, "macro_f1_mean"),
                "cohen_kappa_mean": _metric(row, "cohen_kappa_mean"),
            }
        )
    return rf_rows


def _build_smoothing_rows() -> list[dict[str, object]]:
    return [
        {"method": "Raw", "accuracy_mean": 0.593750, "macro_f1_mean": 0.541780, "cohen_kappa_mean": 0.368823},
        {
            "method": "Moving Average",
            "accuracy_mean": 0.606200,
            "macro_f1_mean": 0.557700,
            "cohen_kappa_mean": 0.389300,
        },
        {"method": "Savitzky-Golay", "accuracy_mean": 0.612500, "macro_f1_mean": 0.567100, "cohen_kappa_mean": 0.397000},
        {"method": "One-Euro", "accuracy_mean": 0.637500, "macro_f1_mean": 0.589700, "cohen_kappa_mean": 0.436500},
        {"method": "Kalman", "accuracy_mean": 0.625000, "macro_f1_mean": 0.581500, "cohen_kappa_mean": 0.419000},
        {"method": "Optimized Full", "accuracy_mean": 0.762500, "macro_f1_mean": 0.747976, "cohen_kappa_mean": 0.637073},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready summary figures from existing experiment CSVs.")
    parser.add_argument("--suite-csv", default="figures/classifier_suite_v2/classification_suite_v2.csv")
    parser.add_argument("--pca-csv", default="figures/pca_sweep_v2/pca_sweep_summary.csv")
    parser.add_argument("--output-dir", default="figures/paper_summary")
    args = parser.parse_args()

    suite_path = Path(args.suite_csv).resolve()
    pca_path = Path(args.pca_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_rows = _read_csv(suite_path)
    pca_rows = _read_csv(pca_path)

    _plot_classifier_accuracy(
        suite_rows,
        output_dir / "classifier_accuracy_structured.png",
    )

    rf_method_rows = _build_rf_method_rows(suite_rows, pca_rows)
    _write_csv(
        output_dir / "rf_method_comparison.csv",
        ["method", "accuracy_mean", "macro_f1_mean", "cohen_kappa_mean"],
        rf_method_rows,
    )
    _plot_grouped_metrics(
        rf_method_rows,
        [str(row["method"]) for row in rf_method_rows],
        [("accuracy_mean", "Accuracy"), ("macro_f1_mean", "Macro-F1"), ("cohen_kappa_mean", "Kappa")],
        output_dir / "rf_method_comparison.png",
        "RandomForest Comparison: Raw, PCA, and ORCA/MuJoCo Methods",
    )

    smoothing_rows = _build_smoothing_rows()
    _write_csv(
        output_dir / "rf_smoothing_comparison.csv",
        ["method", "accuracy_mean", "macro_f1_mean", "cohen_kappa_mean"],
        smoothing_rows,
    )
    _plot_grouped_metrics(
        smoothing_rows,
        [str(row["method"]) for row in smoothing_rows],
        [("accuracy_mean", "Accuracy"), ("macro_f1_mean", "Macro-F1"), ("cohen_kappa_mean", "Kappa")],
        output_dir / "rf_smoothing_comparison.png",
        "RandomForest Comparison: Landmark-space Smoothing vs Optimized Full",
    )

    print(f"paper_figures_dir={output_dir}")


if __name__ == "__main__":
    main()
