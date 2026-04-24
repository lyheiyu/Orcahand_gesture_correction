import argparse
import csv
from pathlib import Path

import numpy as np


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _latest_per_feature(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for row in rows:
        feature_set = row["feature_set"]
        if feature_set not in latest:
            order.append(feature_set)
        latest[feature_set] = row
    return [latest[feature_set] for feature_set in order]


def _plot(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [row["feature_set"] for row in rows]
    metrics = [
        ("macro_recall", "Recall"),
        ("macro_f1", "F1"),
        ("macro_precision", "Precision"),
        ("accuracy", "Accuracy"),
        ("cohen_kappa", "Kappa"),
    ]
    colors = ["#4C78A8", "#E45756", "#F58518", "#54A24B", "#B279A2"]

    x = np.arange(len(labels))
    width = 0.15
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(10.4, 5.2), dpi=180)

    bar_containers = []
    for offset, (metric_key, metric_label), color in zip(offsets, metrics, colors, strict=True):
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
        bar_containers.append(bars)

    ax.set_title("Few-Shot Sequence Classification: Five Core Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean score across repeats")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    for bars in bar_containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, rotation=90)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_five_metric_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_set",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "accuracy_mean",
        "accuracy_std",
        "kappa_mean",
        "kappa_std",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "feature_set": row["feature_set"],
                    "recall_mean": row["macro_recall_mean"],
                    "recall_std": row["macro_recall_std"],
                    "f1_mean": row["macro_f1_mean"],
                    "f1_std": row["macro_f1_std"],
                    "precision_mean": row["macro_precision_mean"],
                    "precision_std": row["macro_precision_std"],
                    "accuracy_mean": row["accuracy_mean"],
                    "accuracy_std": row["accuracy_std"],
                    "kappa_mean": row["cohen_kappa_mean"],
                    "kappa_std": row["cohen_kappa_std"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot classification summary from train_svm.py result CSV.")
    parser.add_argument("--results-csv", default="figures/classification_v2.csv")
    parser.add_argument("--plot", default="figures/classification_v2.png")
    parser.add_argument("--five-metric-csv", default="")
    parser.add_argument("--classifier", default="", help="Optional classifier filter, e.g. svm / knn / rf / mlp.")
    args = parser.parse_args()

    results_path = Path(args.results_csv).resolve()
    rows = _load_rows(results_path)
    if args.classifier:
        rows = [row for row in rows if row.get("classifier", "svm") == args.classifier]
    rows = _latest_per_feature(rows)
    if not rows:
        raise SystemExit(f"No rows found in {results_path}.")

    output_path = Path(args.plot).resolve()
    _plot(rows, output_path)
    print(f"classification_plot={output_path}")

    if args.five_metric_csv:
        five_metric_path = Path(args.five_metric_csv).resolve()
        _write_five_metric_csv(rows, five_metric_path)
        print(f"five_metric_csv={five_metric_path}")


if __name__ == "__main__":
    main()
