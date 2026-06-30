import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


CHINESE_DANCE_6_LABELS = [
    "orchid_palm",
    "orchid_finger",
    "flower_pinch",
    "prayer_beads",
    "three_finger_bent",
    "deer_horn",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a gesture CSV by selected labels and export a subset dataset."
    )
    parser.add_argument(
        "--input",
        default="gesture_sequence_dataset_optimized_v2.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="gesture_sequence_dataset_chinese_dance_6class.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--preset",
        choices=["chinese_dance_6"],
        default="chinese_dance_6",
        help="Named label preset to export.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional explicit labels. If provided, overrides --preset.",
    )
    return parser.parse_args()


def _selected_labels(args: argparse.Namespace) -> list[str]:
    if args.labels:
        return [label.strip() for label in args.labels if label.strip()]
    if args.preset == "chinese_dance_6":
        return CHINESE_DANCE_6_LABELS.copy()
    raise SystemExit(f"Unsupported preset: {args.preset}")


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input CSV does not exist: {input_path}")

    keep_labels = _selected_labels(args)
    keep_set = set(keep_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    row_counts = Counter()
    sequence_sets: dict[str, set[str]] = defaultdict(set)
    seen_labels = set()

    with input_path.open("r", newline="", encoding="utf-8") as fh_in:
        reader = csv.DictReader(fh_in)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit(f"CSV has no header: {input_path}")

        with output_path.open("w", newline="", encoding="utf-8") as fh_out:
            writer = csv.DictWriter(fh_out, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                label = str(row.get("label", "")).strip()
                seen_labels.add(label)
                if label not in keep_set:
                    continue

                writer.writerow({name: row.get(name, "") for name in fieldnames})
                rows_written += 1
                row_counts[label] += 1
                sequence_id = str(row.get("sequence_id", "")).strip()
                if sequence_id:
                    sequence_sets[label].add(sequence_id)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"rows_written={rows_written}")
    print("selected_labels=" + ",".join(keep_labels))

    missing_labels = [label for label in keep_labels if label not in seen_labels]
    if missing_labels:
        print("missing_labels=" + ",".join(missing_labels))

    print("\nPer-label summary:")
    for label in keep_labels:
        print(
            f"  {label}: rows={row_counts[label]} "
            f"sequences={len(sequence_sets[label])}"
        )


if __name__ == "__main__":
    main()
