import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


META_FIELDS = {"label", "sequence_id", "frame_id", "timestamp_sec"}
DEFAULT_CLASSIFIERS = ["svm", "knn", "rf", "mlp"]
DEFAULT_STRUCTURED = ["raw", "corrected", "optimized_action", "optimized_full"]
DEFAULT_SMOOTHING = ["raw", "moving_average_raw", "savgol_raw", "oneeuro_raw", "kalman_raw", "optimized_full"]


def _run(command: list[str], workdir: Path) -> None:
    print("running:", " ".join(command))
    subprocess.run(command, cwd=workdir, check=True)


def _feature_sets_from_csv(dataset_path: Path) -> list[str]:
    with dataset_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

    detected: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"^(?P<prefix>.+)_\d+$")
    for field in fieldnames:
        if field in META_FIELDS:
            continue
        match = pattern.match(field)
        if not match:
            continue
        prefix = match.group("prefix")
        if prefix not in seen:
            seen.add(prefix)
            detected.append(prefix)
    return detected


def _ordered_present(candidates: list[str], available: list[str]) -> list[str]:
    available_set = set(available)
    return [name for name in candidates if name in available_set]


def _best_pca_dims_from_results(pca_summary_path: Path, classifiers: list[str]) -> dict[str, str]:
    if not pca_summary_path.exists():
        return {}
    with pca_summary_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    best_by_classifier: dict[str, tuple[str, float]] = {}
    for row in rows:
        classifier = row.get("classifier", "")
        feature_set = row.get("feature_set", "")
        if classifier not in classifiers or not feature_set.startswith("raw_pca"):
            continue
        accuracy = float(row["accuracy_mean"])
        current = best_by_classifier.get(classifier)
        if current is None or accuracy > current[1]:
            best_by_classifier[classifier] = (feature_set, accuracy)
    return {classifier: feature_set for classifier, (feature_set, _) in best_by_classifier.items()}


def _build_feature_plan(
    available: list[str],
    include_structured: bool,
    include_smoothing: bool,
    include_all_detected: bool,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def push_many(values: list[str]) -> None:
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)

    if include_structured:
        push_many(_ordered_present(DEFAULT_STRUCTURED, available))
    if include_smoothing:
        push_many(_ordered_present(DEFAULT_SMOOTHING, available))
    if include_all_detected:
        push_many(available)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically run classifier experiments and generate figures/CMs.")
    parser.add_argument("--dataset", required=True, help="Input gesture CSV dataset.")
    parser.add_argument("--output-dir", default="figures/auto_suite", help="Directory for generated outputs.")
    parser.add_argument("--classifiers", nargs="+", default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--shots-per-class", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sequence-mode", action="store_true")
    parser.add_argument("--include-structured", action="store_true")
    parser.add_argument("--include-smoothing", action="store_true")
    parser.add_argument("--include-all-detected", action="store_true")
    parser.add_argument("--include-pca17", action="store_true")
    parser.add_argument("--include-best-pca", action="store_true")
    parser.add_argument("--pca-summary-csv", default="figures/pca_sweep_v2/pca_sweep_summary.csv")
    args = parser.parse_args()

    workdir = Path.cwd()
    dataset_path = (workdir / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    output_dir = (workdir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    available_feature_sets = _feature_sets_from_csv(dataset_path)
    feature_plan = _build_feature_plan(
        available_feature_sets,
        include_structured=args.include_structured or not (args.include_structured or args.include_smoothing or args.include_all_detected),
        include_smoothing=args.include_smoothing,
        include_all_detected=args.include_all_detected,
    )

    results_csv = output_dir / "experiment_results.csv"
    if results_csv.exists():
        results_csv.unlink()

    base_command = [
        sys.executable,
        "train_svm.py",
        "--dataset",
        str(dataset_path),
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
        for feature_set in feature_plan:
            cm_path = output_dir / "cms" / f"cm_{classifier}_{feature_set}.png"
            command = base_command + [
                "--classifier",
                classifier,
                "--feature-set",
                feature_set,
                "--plot-confusion",
                str(cm_path),
                "--confusion-title",
                f"{classifier.upper()} - {feature_set}",
            ]
            _run(command, workdir)

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
        _run(plot_command, workdir)

    if args.include_pca17 or args.include_best_pca:
        pca_summary_path = (workdir / args.pca_summary_csv).resolve() if not Path(args.pca_summary_csv).is_absolute() else Path(args.pca_summary_csv)
        best_pca_by_classifier = _best_pca_dims_from_results(pca_summary_path, args.classifiers)
        for classifier in args.classifiers:
            pca_targets: list[tuple[str, int]] = []
            if args.include_pca17:
                pca_targets.append(("raw_pca17", 17))
            if args.include_best_pca and classifier in best_pca_by_classifier:
                feature_name = best_pca_by_classifier[classifier]
                match = re.search(r"raw_pca(\d+)$", feature_name)
                if match:
                    pca_targets.append((feature_name, int(match.group(1))))
            seen_names: set[str] = set()
            for feature_name, components in pca_targets:
                if feature_name in seen_names:
                    continue
                seen_names.add(feature_name)
                cm_path = output_dir / "cms" / f"cm_{classifier}_{feature_name}.png"
                command = base_command + [
                    "--classifier",
                    classifier,
                    "--feature-set",
                    "raw",
                    "--pca-components",
                    str(components),
                    "--plot-confusion",
                    str(cm_path),
                    "--confusion-title",
                    f"{classifier.upper()} - {feature_name}",
                ]
                _run(command, workdir)

    manifest_path = output_dir / "run_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "output_dir",
                "classifiers",
                "feature_sets",
                "sequence_mode",
                "results_csv",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": str(dataset_path),
                "output_dir": str(output_dir),
                "classifiers": ",".join(args.classifiers),
                "feature_sets": ",".join(feature_plan),
                "sequence_mode": "yes" if args.sequence_mode else "no",
                "results_csv": str(results_csv),
            }
        )

    print(f"available_feature_sets={','.join(available_feature_sets)}")
    print(f"selected_feature_sets={','.join(feature_plan)}")
    print(f"results_csv={results_csv}")
    print(f"output_dir={output_dir}")
    print(f"manifest_csv={manifest_path}")


if __name__ == "__main__":
    main()
