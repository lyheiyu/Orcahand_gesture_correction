import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


META_FIELDS = {"label", "sequence_id", "frame_id", "timestamp_sec"}


def _load_dataset(path: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        feature_names = [name for name in fieldnames if name not in META_FIELDS]
        rows_meta: list[dict[str, str]] = []
        rows: list[list[float]] = []
        for row in reader:
            rows_meta.append(
                {
                    "label": row.get("label", ""),
                    "sequence_id": row.get("sequence_id", ""),
                    "frame_id": row.get("frame_id", "0"),
                    "timestamp_sec": row.get("timestamp_sec", "0"),
                }
            )
            rows.append([float(row[name]) for name in feature_names])
    return rows_meta, feature_names, np.asarray(rows, dtype=np.float64)


def _select_features(
    feature_names: list[str],
    features: np.ndarray,
    feature_set: str,
) -> tuple[list[str], np.ndarray]:
    prefix = f"{feature_set}_"
    indices = [index for index, name in enumerate(feature_names) if name.startswith(prefix)]
    if not indices:
        raise SystemExit(f"No columns found for feature set `{feature_set}`.")
    return [feature_names[index] for index in indices], features[:, indices]


def _group_sequences(
    row_meta: list[dict[str, str]],
    features: np.ndarray,
) -> dict[str, list[tuple[int, float, str, np.ndarray]]]:
    grouped: dict[str, list[tuple[int, float, str, np.ndarray]]] = defaultdict(list)
    for index, (meta, feature_row) in enumerate(zip(row_meta, features, strict=True)):
        sequence_id = meta.get("sequence_id") or f"single_{index}"
        frame_id = int(float(meta.get("frame_id") or 0))
        timestamp_sec = float(meta.get("timestamp_sec") or 0.0)
        label = meta.get("label", "")
        grouped[sequence_id].append((frame_id, timestamp_sec, label, feature_row))
    return grouped


def _sequence_metrics(sequence: np.ndarray) -> dict[str, float]:
    if sequence.shape[0] < 2:
        return {
            "velocity_mean": 0.0,
            "velocity_rms": 0.0,
            "acceleration_mean": 0.0,
            "acceleration_rms": 0.0,
        }

    velocity = np.diff(sequence, axis=0)
    velocity_norm = np.linalg.norm(velocity, axis=1)
    metrics = {
        "velocity_mean": float(np.mean(velocity_norm)),
        "velocity_rms": float(np.sqrt(np.mean(velocity_norm**2))),
        "acceleration_mean": 0.0,
        "acceleration_rms": 0.0,
    }

    if sequence.shape[0] >= 3:
        acceleration = sequence[2:] - (2.0 * sequence[1:-1]) + sequence[:-2]
        acceleration_norm = np.linalg.norm(acceleration, axis=1)
        metrics["acceleration_mean"] = float(np.mean(acceleration_norm))
        metrics["acceleration_rms"] = float(np.sqrt(np.mean(acceleration_norm**2)))

    return metrics


def _summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def evaluate_feature_set(
    row_meta: list[dict[str, str]],
    feature_names: list[str],
    features: np.ndarray,
    feature_set: str,
) -> dict[str, float]:
    _, selected = _select_features(feature_names, features, feature_set)
    grouped = _group_sequences(row_meta, selected)

    per_sequence: dict[str, list[float]] = defaultdict(list)
    num_sequences = 0
    num_frames = 0
    for entries in grouped.values():
        sorted_entries = sorted(entries, key=lambda item: (item[0], item[1]))
        if len(sorted_entries) < 2:
            continue
        sequence = np.stack([entry[3] for entry in sorted_entries], axis=0)
        metrics = _sequence_metrics(sequence)
        for name, value in metrics.items():
            per_sequence[name].append(value)
        num_sequences += 1
        num_frames += len(sorted_entries)

    summary: dict[str, float] = {
        "num_sequences": float(num_sequences),
        "num_frames": float(num_frames),
    }
    for metric_name, values in per_sequence.items():
        mean, std = _summarize(values)
        summary[f"{metric_name}_mean"] = mean
        summary[f"{metric_name}_std"] = std
    return summary


def _write_summary_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "feature_set",
        "num_sequences",
        "num_frames",
        "velocity_mean",
        "velocity_std",
        "velocity_rms_mean",
        "velocity_rms_std",
        "acceleration_mean",
        "acceleration_std",
        "acceleration_rms_mean",
        "acceleration_rms_std",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_jitter_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["feature_set"]) for row in rows]
    velocity = np.asarray([float(row["velocity_mean"]) for row in rows], dtype=np.float64)
    acceleration = np.asarray([float(row["acceleration_mean"]) for row in rows], dtype=np.float64)

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    ax.bar(x - width / 2, velocity, width, label="Velocity", color="#4C78A8")
    ax.bar(x + width / 2, acceleration, width, label="Acceleration", color="#F58518")

    ax.set_title("Temporal Jitter Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean temporal difference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate temporal jitter of gesture feature groups.")
    parser.add_argument("--dataset", default="gesture_sequence_dataset_optimized.csv")
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["raw", "corrected", "optimized_action", "optimized_full"],
        help="Feature groups to evaluate by column prefix.",
    )
    parser.add_argument(
        "--results-csv",
        default="",
        help="Optional CSV path for jitter summary rows.",
    )
    parser.add_argument(
        "--plot",
        default="",
        help="Optional output path for a jitter comparison PNG.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    row_meta, feature_names, features = _load_dataset(dataset_path)
    summary_rows: list[dict[str, object]] = []

    print(f"dataset={dataset_path}")
    print("lower velocity/acceleration means smoother temporal behavior")
    print()

    for feature_set in args.feature_sets:
        summary = evaluate_feature_set(row_meta, feature_names, features, feature_set)
        summary_rows.append(
            {
                "dataset": str(dataset_path),
                "feature_set": feature_set,
                "num_sequences": int(summary["num_sequences"]),
                "num_frames": int(summary["num_frames"]),
                "velocity_mean": f"{summary.get('velocity_mean_mean', 0.0):.6f}",
                "velocity_std": f"{summary.get('velocity_mean_std', 0.0):.6f}",
                "velocity_rms_mean": f"{summary.get('velocity_rms_mean', 0.0):.6f}",
                "velocity_rms_std": f"{summary.get('velocity_rms_std', 0.0):.6f}",
                "acceleration_mean": f"{summary.get('acceleration_mean_mean', 0.0):.6f}",
                "acceleration_std": f"{summary.get('acceleration_mean_std', 0.0):.6f}",
                "acceleration_rms_mean": f"{summary.get('acceleration_rms_mean', 0.0):.6f}",
                "acceleration_rms_std": f"{summary.get('acceleration_rms_std', 0.0):.6f}",
            }
        )
        print(f"feature_set={feature_set}")
        print(f"  num_sequences={int(summary['num_sequences'])}")
        print(f"  num_frames={int(summary['num_frames'])}")
        print(f"  velocity_mean={summary.get('velocity_mean_mean', 0.0):.6f} +/- {summary.get('velocity_mean_std', 0.0):.6f}")
        print(f"  velocity_rms={summary.get('velocity_rms_mean', 0.0):.6f} +/- {summary.get('velocity_rms_std', 0.0):.6f}")
        print(
            f"  acceleration_mean={summary.get('acceleration_mean_mean', 0.0):.6f} +/- "
            f"{summary.get('acceleration_mean_std', 0.0):.6f}"
        )
        print(
            f"  acceleration_rms={summary.get('acceleration_rms_mean', 0.0):.6f} +/- "
            f"{summary.get('acceleration_rms_std', 0.0):.6f}"
        )
        print()

    if args.results_csv:
        csv_path = Path(args.results_csv).resolve()
        _write_summary_csv(csv_path, summary_rows)
        print(f"results_csv={csv_path}")

    if args.plot:
        plot_path = Path(args.plot).resolve()
        _plot_jitter_summary(summary_rows, plot_path)
        print(f"jitter_plot={plot_path}")


if __name__ == "__main__":
    main()
