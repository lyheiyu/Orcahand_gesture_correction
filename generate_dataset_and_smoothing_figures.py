from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")


DISPLAY_NAMES = {
    "raw": "Raw",
    "moving_average_raw": "Moving Average",
    "savgol_raw": "Savitzky-Golay",
    "oneeuro_raw": "One-Euro",
    "kalman_raw": "Kalman",
    "corrected": "Corrected",
    "optimized_action": "Optimized Action",
    "optimized_full": "Optimized Full",
}


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_distribution(dataset: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    frame_counts: Counter[str] = Counter()
    sequence_ids: dict[str, set[str]] = defaultdict(set)
    with dataset.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = row["label"]
            frame_counts[label] += 1
            sequence_ids[label].add(row["sequence_id"])

    labels = sorted(frame_counts)
    rows = [
        {
            "label": label,
            "num_sequences": len(sequence_ids[label]),
            "num_frames": frame_counts[label],
        }
        for label in labels
    ]
    _write_rows(output_dir / "dataset_summary_6class.csv", rows)

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(9.0, 4.8))
    ax1.bar(x, [len(sequence_ids[label]) for label in labels], color="#4C78A8", label="Sequences")
    ax1.set_ylabel("Number of sequences")
    ax1.set_xticks(x, labels, rotation=25, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, [frame_counts[label] for label in labels], color="#F58518", marker="o", linewidth=2.2, label="Frames")
    ax2.set_ylabel("Number of frames")
    handles1, names1 = ax1.get_legend_handles_labels()
    handles2, names2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, names1 + names2, loc="upper right", frameon=False)
    ax1.grid(axis="y", alpha=0.25)
    ax2.grid(False)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_distribution_6class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def smoothing_comparison(results_csv: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    with results_csv.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"No rows found in {results_csv}")

    feature_sets = [row["feature_set"] for row in rows]
    means = [float(row["accuracy_mean"]) for row in rows]
    stds = [float(row["accuracy_std"]) for row in rows]
    colors = ["#4C78A8"] * len(rows)
    for index, feature_set in enumerate(feature_sets):
        if feature_set == "kalman_raw":
            colors[index] = "#B279A2"
        elif feature_set == "corrected":
            colors[index] = "#54A24B"
        elif feature_set == "optimized_action":
            colors[index] = "#F58518"

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors)
    ax.set_ylabel("RandomForest accuracy (mean +/- SD)")
    ax.set_xticks(x, [DISPLAY_NAMES.get(value, value) for value in feature_sets], rotation=25, ha="right")
    ax.set_ylim(max(0.0, min(means) - 0.12), min(1.0, max(means) + 0.12))
    ax.set_title("Coordinate Smoothing and Structured Representations", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "smoothing_baselines_6class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def jitter_comparison(jitter_csv: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    with jitter_csv.open("r", newline="", encoding="utf-8") as fh:
        rows = {row["feature_set"]: row for row in csv.DictReader(fh)}

    groups = {
        "actuator": ["corrected", "optimized_action"],
        "landmark": [
            "raw",
            "moving_average_raw",
            "savgol_raw",
            "oneeuro_raw",
            "kalman_raw",
            "optimized_full",
        ],
    }
    colors = {
        "raw": "#4C78A8",
        "moving_average_raw": "#72B7B2",
        "savgol_raw": "#ECA82C",
        "oneeuro_raw": "#B279A2",
        "kalman_raw": "#9D755D",
        "corrected": "#54A24B",
        "optimized_action": "#F58518",
        "optimized_full": "#E45756",
    }

    for space, feature_sets in groups.items():
        missing = [name for name in feature_sets if name not in rows]
        if missing:
            raise SystemExit(f"Missing jitter rows for {space} space: {missing}")

        x = np.arange(len(feature_sets))
        width = 0.36
        velocity = [float(rows[name]["velocity_mean"]) for name in feature_sets]
        acceleration = [float(rows[name]["acceleration_mean"]) for name in feature_sets]
        fig, ax = plt.subplots(figsize=(9.2 if space == "landmark" else 6.4, 4.8))
        ax.bar(x - width / 2, velocity, width, label="Velocity", color="#4C78A8")
        ax.bar(x + width / 2, acceleration, width, label="Acceleration", color="#F58518")
        ax.set_ylabel("Normalized temporal variation (lower is smoother)")
        ax.set_xticks(x, [DISPLAY_NAMES.get(name, name) for name in feature_sets], rotation=25, ha="right")
        ax.set_title(f"Temporal Stability in {space.title()} Space", fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"jitter_{space}_6class.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset and smoothing figures for an updated paper dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--smoothing-results", required=True)
    parser.add_argument("--jitter-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    smoothing_results = Path(args.smoothing_results).resolve()
    jitter_results = Path(args.jitter_results).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_distribution(dataset, output_dir)
    smoothing_comparison(smoothing_results, output_dir)
    jitter_comparison(jitter_results, output_dir)
    print(f"dataset_and_smoothing_figures={output_dir}")


if __name__ == "__main__":
    main()
