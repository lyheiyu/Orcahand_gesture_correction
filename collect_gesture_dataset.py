import argparse
import csv
import sys
import time
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit("Missing `opencv-python`. Install `python -m pip install -e \".[teleop]\"`.") from exc

try:
    import mediapipe as mp
except ModuleNotFoundError as exc:
    raise SystemExit("Missing `mediapipe`. Install `python -m pip install -e \".[teleop]\"`.") from exc

from orca_sim.gesture_features import OrcaFeatureProjector
from orca_sim.mujoco_optimizer import MujocoHandPoseOptimizer

try:
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    MP_BACKEND = "solutions"
except AttributeError:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        HandLandmarksConnections,
        RunningMode,
        drawing_utils as mp_drawing,
    )

    mp_hands = None
    MP_BACKEND = "tasks"


def _landmarks_to_array(hand_landmarks) -> np.ndarray:
    if hasattr(hand_landmarks, "landmark"):
        landmarks = hand_landmarks.landmark
    else:
        landmarks = hand_landmarks
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)


def _effective_handedness(label: str | None, mirror: bool) -> str | None:
    if label is None:
        return None
    lowered = label.lower()
    if not mirror:
        return lowered
    if lowered == "left":
        return "right"
    if lowered == "right":
        return "left"
    return lowered


def _select_target_hand_solutions(results, target_hand: str, mirror: bool):
    landmarks_list = getattr(results, "multi_hand_landmarks", None) or []
    handedness_list = getattr(results, "multi_handedness", None) or []
    for index, hand_landmarks in enumerate(landmarks_list):
        label = None
        if index < len(handedness_list) and handedness_list[index].classification:
            label = handedness_list[index].classification[0].label.lower()
        if target_hand == "either" or _effective_handedness(label, mirror) == target_hand:
            return hand_landmarks
    return None


def _select_target_hand_tasks(results, target_hand: str, mirror: bool):
    landmarks_list = getattr(results, "hand_landmarks", None) or []
    handedness_list = getattr(results, "handedness", None) or []
    for index, hand_landmarks in enumerate(landmarks_list):
        label = None
        if index < len(handedness_list) and handedness_list[index]:
            label = handedness_list[index][0].category_name.lower()
        if target_hand == "either" or _effective_handedness(label, mirror) == target_hand:
            return hand_landmarks
    return None


def _resolve_default_task_model_path() -> Path | None:
    candidates = [
        REPO_ROOT / "hand_landmarker.task",
        REPO_ROOT / "models" / "hand_landmarker.task",
        REPO_ROOT / "assets" / "hand_landmarker.task",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _draw_landmarks(frame, hand_landmarks) -> None:
    try:
        if MP_BACKEND == "solutions":
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            mp_drawing.draw_landmarks(frame, hand_landmarks, HandLandmarksConnections.HAND_CONNECTIONS)
    except AttributeError:
        pass


def _build_headers(projector: OrcaFeatureProjector) -> list[str]:
    dummy = np.zeros((21, 3), dtype=np.float64)
    groups = projector.all_feature_groups(dummy)
    headers = ["label", "sequence_id", "frame_id", "timestamp_sec"]
    for prefix, values in groups.items():
        headers.extend(f"{prefix}_{i}" for i in range(len(values)))
    return headers


def _append_optimizer_headers(headers: list[str]) -> list[str]:
    extended = headers.copy()
    extended.extend(f"optimized_action_{i}" for i in range(17))
    extended.extend(f"optimized_sparse_{i}" for i in range(8 * 3))
    extended.extend(f"optimized_full_{i}" for i in range(21 * 3))
    extended.extend(
        [
            "optimized_loss_total",
            "optimized_loss_landmark",
            "optimized_loss_palm",
            "optimized_loss_prior",
            "optimized_loss_temporal",
            "optimized_loss_acceleration",
            "optimized_loss_default_pose",
            "optimized_loss_boundary",
        ]
    )
    return extended


def _row_from_points(
    projector: OrcaFeatureProjector,
    label: str,
    sequence_id: str,
    frame_id: int,
    timestamp_sec: float,
    points: np.ndarray,
) -> list[float | str]:
    groups = projector.all_feature_groups(points)
    row: list[float | str] = [label, sequence_id, frame_id, timestamp_sec]
    for prefix in ("raw", "geom", "corrected"):
        row.extend(float(v) for v in groups[prefix])
    return row


def _append_optimizer_row(
    row: list[float | str],
    optimizer: MujocoHandPoseOptimizer,
    points: np.ndarray,
    prev_action: np.ndarray | None,
    prev_prev_action: np.ndarray | None,
) -> tuple[list[float | str], np.ndarray]:
    result = optimizer.optimize(
        points,
        prev_action=prev_action,
        prev_prev_action=prev_prev_action,
    )
    row.extend(float(v) for v in result.action)
    row.extend(float(v) for v in result.optimized_sparse_points.reshape(-1))
    row.extend(float(v) for v in result.optimized_full_points.reshape(-1))
    row.extend(
        [
            float(result.loss_terms["total"]),
            float(result.loss_terms["landmark"]),
            float(result.loss_terms["palm"]),
            float(result.loss_terms["prior"]),
            float(result.loss_terms["temporal"]),
            float(result.loss_terms["acceleration"]),
            float(result.loss_terms["default_pose"]),
            float(result.loss_terms["boundary"]),
        ]
    )
    return row, result.action.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled gesture samples to CSV.")
    parser.add_argument("--label", required=True, help="Gesture label to assign to captured samples.")
    parser.add_argument("--output", default="gesture_dataset.csv", help="CSV file to append samples to.")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index.")
    parser.add_argument("--hand-landmarker-model", default=None, help="Path to `hand_landmarker.task` if using MediaPipe Tasks.")
    parser.add_argument(
        "--target-hand",
        choices=["right", "left", "either"],
        default="right",
        help="Which detected hand to save.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable selfie-style mirroring.",
    )
    parser.add_argument(
        "--sequence-mode",
        action="store_true",
        help="Record continuous labeled sequences instead of single frames.",
    )
    parser.add_argument(
        "--sequence-id",
        default=None,
        help="Optional fixed sequence id. By default a new id is generated when recording starts.",
    )
    parser.add_argument(
        "--save-every-n-frames",
        type=int,
        default=1,
        help="When recording a sequence, save one sample every N tracked frames.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="ORCA hand version for projection and optimization, for example v2.",
    )
    parser.add_argument(
        "--export-optimized",
        action="store_true",
        help="Also export MuJoCo-optimized actions and refined landmarks.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    mirror = not args.no_mirror
    save_every_n_frames = max(1, args.save_every_n_frames)
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera_id}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with OrcaFeatureProjector(version=args.version) as projector:
        headers = _build_headers(projector)
        if args.export_optimized:
            headers = _append_optimizer_headers(headers)
        need_header = not output_path.exists()
        optimizer = MujocoHandPoseOptimizer(version=args.version) if args.export_optimized else None
        prev_optimized_action: np.ndarray | None = None
        prev_prev_optimized_action: np.ndarray | None = None

        if MP_BACKEND == "solutions":
            tracker = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
                model_complexity=1,
            )
        else:
            model_path = Path(args.hand_landmarker_model).resolve() if args.hand_landmarker_model else _resolve_default_task_model_path()
            if model_path is None or not model_path.exists():
                raise SystemExit("Need `hand_landmarker.task` for the MediaPipe tasks backend.")
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            tracker = HandLandmarker.create_from_options(options)

        try:
            with tracker, output_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if need_header:
                    writer.writerow(headers)

                saved = 0
                recording = False
                sequence_id = args.sequence_id or ""
                sequence_frame_id = 0
                tracked_frame_counter = 0
                sequence_start_time = 0.0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if mirror:
                        frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    hand_landmarks = None
                    if MP_BACKEND == "solutions":
                        results = tracker.process(rgb)
                        hand_landmarks = _select_target_hand_solutions(results, args.target_hand, mirror)
                    else:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        results = tracker.detect(mp_image)
                        hand_landmarks = _select_target_hand_tasks(results, args.target_hand, mirror)

                    points = None
                    if hand_landmarks is not None:
                        points = _landmarks_to_array(hand_landmarks)
                        if mirror:
                            points[:, 0] = 1.0 - points[:, 0]
                        _draw_landmarks(frame, hand_landmarks)
                        tracked_frame_counter += 1
                    else:
                        tracked_frame_counter = 0

                    cv2.putText(
                        frame,
                        f"label={args.label}",
                        (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (80, 255, 120),
                        2,
                    )
                    cv2.putText(frame, f"saved={saved}", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 120), 2)
                    cv2.putText(
                        frame,
                        f"detected={'yes' if points is not None else 'no'} hand={args.target_hand}",
                        (16, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (80, 255, 120) if points is not None else (70, 120, 255),
                        2,
                    )
                    if args.sequence_mode:
                        status_color = (80, 255, 120) if recording else (255, 210, 80)
                        cv2.putText(
                            frame,
                            f"recording={'yes' if recording else 'no'} seq={sequence_id or '-'} frame_id={sequence_frame_id}",
                            (16, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            status_color,
                            2,
                        )
                        help_text = "space start/stop seq | q quit"
                    else:
                        help_text = "space save | q quit"
                    cv2.putText(
                        frame,
                        help_text,
                        (16, frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (235, 235, 235),
                        1,
                    )
                    cv2.imshow("Gesture Dataset Collector", frame)

                    if args.sequence_mode and recording and points is not None and tracked_frame_counter % save_every_n_frames == 0:
                        timestamp_sec = time.perf_counter() - sequence_start_time
                        row = _row_from_points(
                            projector,
                            args.label,
                            sequence_id,
                            sequence_frame_id,
                            timestamp_sec,
                            points,
                        )
                        if optimizer is not None:
                            old_prev_action = prev_optimized_action
                            row, prev_optimized_action = _append_optimizer_row(
                                row,
                                optimizer,
                                points,
                                prev_optimized_action,
                                prev_prev_optimized_action,
                            )
                            prev_prev_optimized_action = old_prev_action
                        writer.writerow(row)
                        fh.flush()
                        saved += 1
                        sequence_frame_id += 1

                    key = cv2.waitKey(1) & 0xFF
                    if key in {27, ord("q")}:
                        break
                    if key == ord(" "):
                        if args.sequence_mode:
                            if recording:
                                recording = False
                                prev_optimized_action = None
                                prev_prev_optimized_action = None
                            else:
                                sequence_id = args.sequence_id or uuid.uuid4().hex[:12]
                                sequence_frame_id = 0
                                tracked_frame_counter = 0
                                sequence_start_time = time.perf_counter()
                                prev_optimized_action = None
                                prev_prev_optimized_action = None
                                recording = True
                        elif points is not None:
                            row = _row_from_points(
                                projector,
                                args.label,
                                "single_frame",
                                saved,
                                0.0,
                                points,
                            )
                            if optimizer is not None:
                                old_prev_action = prev_optimized_action
                                row, prev_optimized_action = _append_optimizer_row(
                                    row,
                                    optimizer,
                                    points,
                                    prev_optimized_action,
                                    prev_prev_optimized_action,
                                )
                                prev_prev_optimized_action = old_prev_action
                            writer.writerow(row)
                            fh.flush()
                            saved += 1
        finally:
            if optimizer is not None:
                optimizer.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
