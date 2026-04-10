import argparse
import csv
import shutil
from pathlib import Path


META_KEY_FIELDS = ("label", "sequence_id", "frame_id")


def _read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration as exc:
            raise SystemExit(f"CSV is empty: {path}") from exc


def _row_key(row: dict[str, str], key_fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in key_fields)


def _load_existing_keys(path: Path, key_fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    if not path.exists():
        return keys
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            keys.add(_row_key(row, key_fields))
    return keys


def _append_rows(
    master_path: Path,
    source_paths: list[Path],
    *,
    dedupe: bool,
    key_fields: tuple[str, ...],
    make_backup: bool,
) -> tuple[int, int]:
    if not source_paths:
        raise SystemExit("No source CSVs were provided.")

    source_headers = [_read_header(path) for path in source_paths]
    master_exists = master_path.exists()
    master_header = _read_header(master_path) if master_exists else source_headers[0]

    for path, header in zip(source_paths, source_headers, strict=True):
        if header != master_header:
            raise SystemExit(
                f"Header mismatch between master/source CSVs.\n"
                f"master={master_path}\nsource={path}"
            )

    if make_backup and master_exists:
        backup_path = master_path.with_suffix(master_path.suffix + ".bak")
        shutil.copy2(master_path, backup_path)

    existing_keys = _load_existing_keys(master_path, key_fields) if dedupe else set()
    rows_added = 0
    rows_skipped = 0

    write_header = not master_exists
    with master_path.open("a", newline="", encoding="utf-8") as fh_out:
        writer = csv.DictWriter(fh_out, fieldnames=master_header)
        if write_header:
            writer.writeheader()

        for source_path in source_paths:
            with source_path.open("r", newline="", encoding="utf-8") as fh_in:
                reader = csv.DictReader(fh_in)
                for row in reader:
                    if dedupe:
                        key = _row_key(row, key_fields)
                        if key in existing_keys:
                            rows_skipped += 1
                            continue
                        existing_keys.add(key)
                    writer.writerow({name: row.get(name, "") for name in master_header})
                    rows_added += 1

    return rows_added, rows_skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge newly collected gesture CSV files into a master dataset."
    )
    parser.add_argument(
        "--master",
        required=True,
        help="Master CSV to append into. It will be created if it does not exist.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="One or more source CSV files to merge into the master dataset.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication by label + sequence_id + frame_id.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak copy of the master CSV before merging.",
    )
    parser.add_argument(
        "--delete-sources",
        action="store_true",
        help="Delete source CSV files after a successful merge.",
    )
    args = parser.parse_args()

    master_path = Path(args.master).resolve()
    source_paths = [Path(source).resolve() for source in args.sources]

    rows_added, rows_skipped = _append_rows(
        master_path,
        source_paths,
        dedupe=not args.no_dedupe,
        key_fields=META_KEY_FIELDS,
        make_backup=not args.no_backup,
    )

    if args.delete_sources:
        for source_path in source_paths:
            source_path.unlink(missing_ok=False)

    print(f"master={master_path}")
    print(f"sources={len(source_paths)}")
    print(f"rows_added={rows_added}")
    print(f"rows_skipped={rows_skipped}")
    print(f"dedupe={'no' if args.no_dedupe else 'yes'}")
    if not args.no_backup and master_path.exists():
        print(f"backup={master_path.with_suffix(master_path.suffix + '.bak')}")


if __name__ == "__main__":
    main()
