import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


DEFAULT_CLASSIFIERS = ["svm", "knn", "rf", "mlp"]
DEFAULT_PCA_COMPONENTS = [8, 12, 17, 24, 32, 48]
REFERENCE_FEATURES = ["raw", "corrected", "optimized_action", "optimized_full"]


def _run_command(command: list[str], workdir: Path) -> None:
    print("running:", " ".join(command))
    env = os.environ.copy()
    env.setdefault("LOKY_MAX_CPU_COUNT", "8")
    subprocess.run(command, cwd=workdir, check=True, env=env)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _latest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (row["classifier"], row["feature_set"])
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]


def _plot_classifier_sweep(rows: list[dict[str, str]], classifier: str, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classifier_rows = [row for row in rows if row["classifier"] == classifier]
    if not classifier_rows:
        raise SystemExit(f"No rows found for classifier `{classifier}`.")

    pca_rows = sorted(
        [row for row in classifier_rows if row.get("feature_set_base", "") == "raw" and row.get("pca_components", "")],
        key=lambda row: int(row["pca_components"]),
    )
    if not pca_rows:
        raise SystemExit(f"No PCA rows found for classifier `{classifier}`.")

    reference_rows = {row["feature_set"]: row for row in classifier_rows if row["feature_set"] in REFERENCE_FEATURES}

    x = np.asarray([int(row["pca_components"]) for row in pca_rows], dtype=np.int32)
    accuracy = np.asarray([float(row["accuracy_mean"]) for row in pca_rows], dtype=np.float64)
    accuracy_std = np.asarray([float(row["accuracy_std"]) for row in pca_rows], dtype=np.float64)
    macro_f1 = np.asarray([float(row["macro_f1_mean"]) for row in pca_rows], dtype=np.float64)
    macro_f1_std = np.asarray([float(row["macro_f1_std"]) for row in pca_rows], dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=180)

    ax.errorbar(x, accuracy, yerr=accuracy_std, marker="o", linewidth=2.0, capsize=3, label="PCA Accuracy", color="#4C78A8")
    ax.errorbar(x, macro_f1, yerr=macro_f1_std, marker="s", linewidth=2.0, capsize=3, label="PCA Macro-F1", color="#E45756")

    reference_styles = {
        "raw": ("Raw baseline", "#9D755D"),
        "corrected": ("Corrected", "#54A24B"),
        "optimized_action": ("Optimized action", "#F58518"),
        "optimized_full": ("Optimized full", "#B279A2"),
    }
    for feature_set, (label, color) in reference_styles.items():
        row = reference_rows.get(feature_set)
        if row is None:
            continue
        ax.axhline(float(row["accuracy_mean"]), linestyle="--", linewidth=1.5, color=color, alpha=0.9, label=f"{label} accuracy")

    ax.set_title(f"{classifier.upper()} PCA Dimension Sweep", fontsize=13, fontweight="bold")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Mean score across repeats")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_summary_csv(rows: list[dict[str, str]], output_path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA dimension sweep across multiple classifiers.")
    parser.add_argument("--dataset", default="gesture_sequence_dataset_optimized_v2.csv")
    parser.add_argument("--output-dir", default="figures/pca_sweep_v2")
    parser.add_argument("--results-csv", default="pca_sweep_results.csv")
    parser.add_argument("--summary-csv", default="pca_sweep_summary.csv")
    parser.add_argument("--shots-per-class", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sequence-mode", action="store_true")
    parser.add_argument("--classifiers", nargs="+", default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--pca-components", nargs="+", type=int, default=DEFAULT_PCA_COMPONENTS)
    args = parser.parse_args()

    workdir = Path.cwd()
    output_dir = (workdir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / args.results_csv
    summary_csv = output_dir / args.summary_csv

    if results_csv.exists():
        results_csv.unlink()

    base_command = [
        sys.executable,
        "train_svm.py",
        "--dataset",
        args.dataset,
        "--shots-per-class",
        str(args.shots_per_class),
        "--repeats",
        str(args.repeats),
        "--test-size",
        str(args.test_size),
        "--random-state",
        str(args.random_state),
        "--results-csv",
        str(results_csv),
    ]
    if args.sequence_mode:
        base_command.append("--sequence-mode")

    for classifier in args.classifiers:
        for feature_set in REFERENCE_FEATURES:
            command = base_command + [
                "--classifier",
                classifier,
                "--feature-set",
                feature_set,
            ]
            _run_command(command, workdir)

        for components in args.pca_components:
            command = base_command + [
                "--classifier",
                classifier,
                "--feature-set",
                "raw",
                "--pca-components",
                str(components),
            ]
            _run_command(command, workdir)

    rows = _latest_rows(_load_rows(results_csv))
    _write_summary_csv(rows, summary_csv)

    for classifier in args.classifiers:
        plot_path = output_dir / f"pca_sweep_{classifier}.png"
        _plot_classifier_sweep(rows, classifier, plot_path)
        print(f"plot={plot_path}")

    print(f"results_csv={results_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
