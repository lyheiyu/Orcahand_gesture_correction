import argparse
import csv
from pathlib import Path

import numpy as np


META_FIELDS = ("label", "sequence_id", "frame_id", "timestamp_sec")


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def _raw_feature_names(fieldnames: list[str]) -> list[str]:
    names = [name for name in fieldnames if name.startswith("raw_")]
    if not names:
        raise SystemExit("No raw_* columns found in the input CSV.")
    return names


def _group_row_indices(rows: list[dict[str, str]]) -> list[list[int]]:
    grouped: dict[str, list[tuple[int, float, int]]] = {}
    for index, row in enumerate(rows):
        sequence_id = row.get("sequence_id") or f"single_{index}_{row.get('label', 'unknown')}"
        frame_id = int(row.get("frame_id") or 0)
        timestamp_sec = float(row.get("timestamp_sec") or 0.0)
        grouped.setdefault(sequence_id, []).append((frame_id, timestamp_sec, index))

    ordered_groups: list[list[int]] = []
    for sequence_id in sorted(grouped):
        entries = sorted(grouped[sequence_id], key=lambda item: (item[0], item[1], item[2]))
        ordered_groups.append([entry[2] for entry in entries])
    return ordered_groups


def _moving_average(sequence: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(sequence) <= 1:
        return sequence.copy()
    radius = window // 2
    padded = np.pad(sequence, ((radius, radius), (0, 0)), mode="edge")
    output = np.zeros_like(sequence)
    for row_index in range(sequence.shape[0]):
        output[row_index] = np.mean(padded[row_index : row_index + window], axis=0)
    return output


def _fit_window_poly(window_values: np.ndarray, x_coords: np.ndarray, center_x: float, order: int) -> float:
    local_order = min(order, len(window_values) - 1)
    if local_order <= 0:
        return float(np.mean(window_values))
    coefficients = np.polyfit(x_coords, window_values, deg=local_order)
    return float(np.polyval(coefficients, center_x))


def _savitzky_golay(sequence: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if window <= 1 or len(sequence) <= 1:
        return sequence.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    output = np.zeros_like(sequence)
    x_coords = np.arange(-radius, radius + 1, dtype=np.float64)
    for dim_index in range(sequence.shape[1]):
        values = sequence[:, dim_index]
        padded = np.pad(values, (radius, radius), mode="edge")
        for row_index in range(len(values)):
            window_values = padded[row_index : row_index + window]
            output[row_index, dim_index] = _fit_window_poly(window_values, x_coords, 0.0, polyorder)
    return output


def _smoothing_factor(dt: float, cutoff: float) -> float:
    if cutoff <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


def _one_euro(sequence: np.ndarray, timestamps: np.ndarray, min_cutoff: float, beta: float, d_cutoff: float) -> np.ndarray:
    if len(sequence) <= 1:
        return sequence.copy()

    output = np.zeros_like(sequence)
    output[0] = sequence[0]
    prev_derivative = np.zeros(sequence.shape[1], dtype=np.float32)

    for row_index in range(1, len(sequence)):
        dt = float(timestamps[row_index] - timestamps[row_index - 1])
        if dt <= 0.0:
            dt = 1.0
        derivative = (sequence[row_index] - output[row_index - 1]) / dt
        alpha_d = _smoothing_factor(dt, d_cutoff)
        derivative_hat = alpha_d * derivative + (1.0 - alpha_d) * prev_derivative
        cutoff = min_cutoff + beta * np.abs(derivative_hat)
        alpha = np.asarray([_smoothing_factor(dt, float(value)) for value in cutoff], dtype=np.float32)
        output[row_index] = alpha * sequence[row_index] + (1.0 - alpha) * output[row_index - 1]
        prev_derivative = derivative_hat
    return output


def _kalman_1d(values: np.ndarray, process_var: float, measurement_var: float) -> np.ndarray:
    estimates = np.zeros_like(values)
    x_hat = float(values[0])
    p = 1.0
    estimates[0] = x_hat
    for index in range(1, len(values)):
        x_hat_minus = x_hat
        p_minus = p + process_var
        k = p_minus / (p_minus + measurement_var)
        x_hat = x_hat_minus + k * (float(values[index]) - x_hat_minus)
        p = (1.0 - k) * p_minus
        estimates[index] = x_hat
    return estimates


def _kalman(sequence: np.ndarray, process_var: float, measurement_var: float) -> np.ndarray:
    if len(sequence) <= 1:
        return sequence.copy()
    output = np.zeros_like(sequence)
    for dim_index in range(sequence.shape[1]):
        output[:, dim_index] = _kalman_1d(sequence[:, dim_index], process_var, measurement_var)
    return output


def _timestamps_for_group(rows: list[dict[str, str]], group_indices: list[int]) -> np.ndarray:
    timestamps = np.array([float(rows[index].get("timestamp_sec") or 0.0) for index in group_indices], dtype=np.float32)
    if np.allclose(timestamps, timestamps[0]):
        timestamps = np.arange(len(group_indices), dtype=np.float32)
    return timestamps


def _sequence_matrix(rows: list[dict[str, str]], group_indices: list[int], raw_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(rows[index][name]) for name in raw_names] for index in group_indices],
        dtype=np.float32,
    )


def _write_rows(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate smoothing/filtering baselines from raw landmark sequences.")
    parser.add_argument("--input", default="gesture_sequence_dataset_optimized_v2.csv")
    parser.add_argument("--output", default="gesture_sequence_dataset_with_smoothing.csv")
    parser.add_argument("--ma-window", type=int, default=5)
    parser.add_argument("--sg-window", type=int, default=7)
    parser.add_argument("--sg-polyorder", type=int, default=2)
    parser.add_argument("--oneeuro-min-cutoff", type=float, default=1.0)
    parser.add_argument("--oneeuro-beta", type=float, default=0.02)
    parser.add_argument("--oneeuro-dcutoff", type=float, default=1.0)
    parser.add_argument("--kalman-process-var", type=float, default=1e-4)
    parser.add_argument("--kalman-measurement-var", type=float, default=1e-2)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    fieldnames, rows = _load_csv(input_path)
    raw_names = _raw_feature_names(fieldnames)
    groups = _group_row_indices(rows)

    ma_names = [f"moving_average_raw_{index}" for index in range(len(raw_names))]
    sg_names = [f"savgol_raw_{index}" for index in range(len(raw_names))]
    oneeuro_names = [f"oneeuro_raw_{index}" for index in range(len(raw_names))]
    kalman_names = [f"kalman_raw_{index}" for index in range(len(raw_names))]

    extended_fieldnames = fieldnames.copy()
    for names in (ma_names, sg_names, oneeuro_names, kalman_names):
        for name in names:
            if name not in extended_fieldnames:
                extended_fieldnames.append(name)

    for group_indices in groups:
        sequence = _sequence_matrix(rows, group_indices, raw_names)
        timestamps = _timestamps_for_group(rows, group_indices)

        ma_sequence = _moving_average(sequence, args.ma_window)
        sg_sequence = _savitzky_golay(sequence, args.sg_window, args.sg_polyorder)
        oneeuro_sequence = _one_euro(
            sequence,
            timestamps,
            args.oneeuro_min_cutoff,
            args.oneeuro_beta,
            args.oneeuro_dcutoff,
        )
        kalman_sequence = _kalman(sequence, args.kalman_process_var, args.kalman_measurement_var)

        for local_index, row_index in enumerate(group_indices):
            row = rows[row_index]
            for feature_index, name in enumerate(ma_names):
                row[name] = f"{float(ma_sequence[local_index, feature_index]):.8f}"
            for feature_index, name in enumerate(sg_names):
                row[name] = f"{float(sg_sequence[local_index, feature_index]):.8f}"
            for feature_index, name in enumerate(oneeuro_names):
                row[name] = f"{float(oneeuro_sequence[local_index, feature_index]):.8f}"
            for feature_index, name in enumerate(kalman_names):
                row[name] = f"{float(kalman_sequence[local_index, feature_index]):.8f}"

    _write_rows(output_path, extended_fieldnames, rows)
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"num_rows={len(rows)}")
    print(f"num_sequences={len(groups)}")
    print("generated_feature_sets=moving_average_raw,savgol_raw,oneeuro_raw,kalman_raw")


if __name__ == "__main__":
    main()
