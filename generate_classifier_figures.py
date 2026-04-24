import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CLASSIFIERS = ["svm", "knn", "rf", "mlp"]
DEFAULT_FEATURE_SETS = ["raw", "corrected", "optimized_action", "optimized_full"]


def _run_command(command: list[str], workdir: Path) -> None:
    print("running:", " ".join(command))
    subprocess.run(command, cwd=workdir, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate classifier comparison figures.")
    parser.add_argument("--dataset", default="gesture_sequence_dataset_optimized_v2.csv")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--results-csv", default="classification_multi.csv")
    parser.add_argument("--shots-per-class", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sequence-mode", action="store_true")
    parser.add_argument("--classifiers", nargs="+", default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS)
    args = parser.parse_args()

    workdir = Path.cwd()
    output_dir = (workdir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / args.results_csv

    if results_csv.exists():
        results_csv.unlink()

    base_train_command = [
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
        base_train_command.append("--sequence-mode")

    for classifier in args.classifiers:
        for feature_set in args.feature_sets:
            confusion_path = output_dir / f"cm_{classifier}_{feature_set}.png"
            command = base_train_command + [
                "--classifier",
                classifier,
                "--feature-set",
                feature_set,
                "--plot-confusion",
                str(confusion_path),
                "--confusion-title",
                f"{classifier.upper()} - {feature_set}",
            ]
            _run_command(command, workdir)

        plot_path = output_dir / f"classification_{classifier}.png"
        five_metric_csv = output_dir / f"classification_{classifier}.csv"
        plot_command = [
            sys.executable,
            "plot_classification_summary.py",
            "--results-csv",
            str(results_csv),
            "--classifier",
            classifier,
            "--plot",
            str(plot_path),
            "--five-metric-csv",
            str(five_metric_csv),
        ]
        _run_command(plot_command, workdir)

    print(f"results_csv={results_csv}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
