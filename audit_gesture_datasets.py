import argparse
import csv
from collections import defaultdict
from pathlib import Path


def inspect(path: Path) -> dict[str, object]:
    labels: dict[str, set[str]] = defaultdict(set)
    frame_keys: set[tuple[str, str, str]] = set()
    rows = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        for row in reader:
            label = row.get("label", "")
            sequence_id = row.get("sequence_id", "")
            labels[label].add(sequence_id)
            frame_keys.add((label, sequence_id, row.get("frame_id", "")))
            rows += 1
    return {
        "path": path,
        "header": header,
        "rows": rows,
        "labels": labels,
        "sequence_keys": {
            (label, sequence_id)
            for label, sequence_ids in labels.items()
            for sequence_id in sequence_ids
        },
        "frame_keys": frame_keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit gesture CSV membership before merging.")
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()

    reports = [inspect(path.resolve()) for path in args.datasets]
    for report in reports:
        print(f"dataset={report['path']}")
        duplicate_frames = int(report["rows"]) - len(report["frame_keys"])
        print(
            f"rows={report['rows']} sequences={len(report['sequence_keys'])} "
            f"columns={len(report['header'])} duplicate_frame_keys={duplicate_frames}"
        )
        print(
            "labels="
            + ", ".join(
                f"{label}:{len(sequence_ids)}"
                for label, sequence_ids in sorted(report["labels"].items())
            )
        )
        print()

    print("Pairwise overlap (sequence keys use label + sequence_id):")
    for left_index, left in enumerate(reports):
        for right in reports[left_index + 1 :]:
            overlap = left["sequence_keys"] & right["sequence_keys"]
            left_only = left["sequence_keys"] - right["sequence_keys"]
            right_only = right["sequence_keys"] - left["sequence_keys"]
            print(
                f"{left['path'].name} <-> {right['path'].name}: "
                f"overlap={len(overlap)} left_only={len(left_only)} right_only={len(right_only)} "
                f"same_header={'yes' if left['header'] == right['header'] else 'no'}"
            )


if __name__ == "__main__":
    main()
