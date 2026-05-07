import argparse
import csv
from pathlib import Path

import numpy as np


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _select_best_pca_row(
    rows: list[dict[str, str]],
    classifier: str,
    base_feature_set: str,
    metric: str,
) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if row.get("classifier", "") == classifier
        and row.get("feature_set_base", "") == base_feature_set
        and row.get("pca_components", "")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[metric]))


def _select_rows(
    rows: list[dict[str, str]],
    classifier: str,
    feature_sets: list[str],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for feature_set in feature_sets:
        matching = [
            row for row in rows if row.get("classifier", "") == classifier and row.get("feature_set", "") == feature_set
        ]
        if matching:
            selected.append(matching[-1])
    return selected


def _write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "classifier",
        "feature_set",
        "feature_set_base",
        "pca_components",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_precision_mean",
        "macro_precision_std",
        "macro_recall_mean",
        "macro_recall_std",
        "cohen_kappa_mean",
        "cohen_kappa_std",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot(rows: list[dict[str, str]], output_path: Path, classifier: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [row["feature_set"] for row in rows]
    metrics = [
        ("macro_recall", "Recall", "#4C78A8"),
        ("macro_f1", "F1", "#E45756"),
        ("macro_precision", "Precision", "#F58518"),
        ("accuracy", "Accuracy", "#54A24B"),
        ("cohen_kappa", "Kappa", "#B279A2"),
    ]

    x = np.arange(len(labels))
    width = 0.15
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(10.6, 5.2), dpi=180)

    for offset, (metric_key, metric_label, color) in zip(offsets, metrics, strict=True):
        mean = np.asarray([float(row[f"{metric_key}_mean"]) for row in rows], dtype=np.float64)
        std = np.asarray([float(row[f"{metric_key}_std"]) for row in rows], dtype=np.float64)
        bars = ax.bar(
            x + offset,
            mean,
            width,
            yerr=std,
            capsize=2,
            label=metric_label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, rotation=90)

    ax.set_title(f"{classifier.upper()} Comparison: Raw vs PCA vs Structured Optimization", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean score across repeats")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a focused comparison subset from result CSVs.")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--classifier", default="rf")
    parser.add_argument("--metric", default="accuracy_mean", help="Metric used to choose the best PCA row.")
    parser.add_argument("--base-feature-set", default="raw", help="Base feature set whose PCA sweep should be searched.")
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["raw", "corrected", "optimized_action", "optimized_full"],
        help="Fixed feature sets to keep alongside the best PCA row.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--plot", required=True)
    args = parser.parse_args()

    results_path = Path(args.results_csv).resolve()
    rows = _load_rows(results_path)

    selected_rows = _select_rows(rows, args.classifier, args.feature_sets)
    best_pca_row = _select_best_pca_row(rows, args.classifier, args.base_feature_set, args.metric)
    if best_pca_row is not None:
        insert_index = 1 if selected_rows else 0
        selected_rows = selected_rows[:insert_index] + [best_pca_row] + selected_rows[insert_index:]

    if not selected_rows:
        raise SystemExit("No rows selected for export.")

    output_csv = Path(args.output_csv).resolve()
    plot_path = Path(args.plot).resolve()
    _write_csv(selected_rows, output_csv)
    _plot(selected_rows, plot_path, args.classifier)
    print(f"comparison_csv={output_csv}")
    print(f"comparison_plot={plot_path}")
    if best_pca_row is not None:
        print(
            f"best_pca={best_pca_row['feature_set']} "
            f"accuracy={best_pca_row['accuracy_mean']} "
            f"macro_f1={best_pca_row['macro_f1_mean']}"
        )


if __name__ == "__main__":
    main()
