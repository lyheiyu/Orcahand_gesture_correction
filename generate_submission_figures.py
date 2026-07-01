from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures" / "paper_rewrite_main"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def dataset_distribution() -> None:
    df = pd.read_csv(OUT_DIR / "dataset_summary_6class.csv")
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.bar(x, df["num_sequences"], color="#4C78A8", label="Sequences")
    ax1.set_ylabel("Number of sequences")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], rotation=25, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, df["num_frames"], color="#F58518", marker="o", linewidth=2.2, label="Frames")
    ax2.set_ylabel("Number of frames")
    ax2.grid(False)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_distribution_6class.png")
    plt.close(fig)


def main_results_overview() -> None:
    df = pd.read_csv(OUT_DIR / "main_results_6class.csv")
    classifiers = ["SVM", "KNN", "RandomForest", "MLP"]
    reps = ["Raw", "Corrected", "Optimized Action", "Optimized Full"]
    colors = {
        "Raw": "#4C78A8",
        "Corrected": "#72B7B2",
        "Optimized Action": "#F58518",
        "Optimized Full": "#54A24B",
    }

    x = np.arange(len(classifiers))
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    for idx, rep in enumerate(reps):
        values = [
            float(df[(df["classifier"] == clf) & (df["representation"] == rep)]["accuracy_mean"].iloc[0])
            for clf in classifiers
        ]
        errors = [
            float(df[(df["classifier"] == clf) & (df["representation"] == rep)]["accuracy_std"].iloc[0])
            for clf in classifiers
        ]
        ax.bar(x + (idx - 1.5) * width, values, width, yerr=errors, capsize=3, color=colors[rep], label=rep)

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.6, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.legend(ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "main_results_overview_6class.png")
    plt.close(fig)


def smoothing_baselines() -> None:
    df = pd.read_csv(OUT_DIR / "smoothing_baseline_results_6class.csv")
    x = np.arange(len(df))
    colors = ["#4C78A8"] * len(df)
    for i, name in enumerate(df["representation"]):
        if name == "Kalman":
            colors[i] = "#B279A2"
        if name == "Optimized Action":
            colors[i] = "#F58518"

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x, df["accuracy_mean"], yerr=df["accuracy_std"], capsize=3, color=colors)
    ax.set_ylabel("RandomForest accuracy")
    ax.set_ylim(0.78, 0.98)
    ax.set_xticks(x)
    ax.set_xticklabels(df["representation"], rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smoothing_baselines_6class.png")
    plt.close(fig)


def pca_comparison() -> None:
    df = pd.read_csv(OUT_DIR / "pca_baseline_results_6class.csv")
    labels = df["classifier"] + " / " + df["representation"]
    x = np.arange(len(df))
    colors = ["#4C78A8", "#B279A2", "#72B7B2", "#F58518", "#F58518"]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(x, df["accuracy_mean"], color=colors)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.82, 0.98)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_comparison_6class.png")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    dataset_distribution()
    main_results_overview()
    smoothing_baselines()
    pca_comparison()
    print(f"submission_figures_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
