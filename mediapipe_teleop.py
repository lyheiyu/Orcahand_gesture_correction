import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing runtime dependency 'opencv-python'. Install it with "
        "`python -m pip install -e \".[teleop]\"`."
    ) from exc

try:
    import mediapipe as mp
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing runtime dependency 'mediapipe'. Install it with "
        "`python -m pip install -e \".[teleop]\"`."
    ) from exc

from orca_sim import OrcaHandRight
from orca_sim.envs import BaseOrcaHandEnv


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

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


@dataclass
class HandFeatures:
    palm_state: str
    palm_normal_x: float
    palm_normal_y: float
    palm_normal_z: float
    base_yaw: float
    base_pitch: float
    base_roll: float
    wrist: float
    pinky_abd: float
    pinky_mcp: float
    pinky_pip: float
    ring_abd: float
    ring_mcp: float
    ring_pip: float
    middle_abd: float
    middle_mcp: float
    middle_pip: float
    index_abd: float
    index_mcp: float
    index_pip: float
    thumb_cmc: float
    thumb_abd: float
    thumb_mcp: float
    thumb_pip: float


@dataclass
class SelectedHand:
    landmarks: object
    label: str


@dataclass
class BaseCalibration:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    calibrated: bool = False


class OrcaHandRightTeleop(BaseOrcaHandEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
    ) -> None:
        super().__init__(
            "scene_right_teleop.xml",
            version=version,
            frame_skip=5,
            render_mode=render_mode,
        )
        self._teleop_base_qpos_indices = {
            "yaw": int(self.model.jnt_qposadr[self.model.joint("teleop_base_yaw").id]),
            "pitch": int(self.model.jnt_qposadr[self.model.joint("teleop_base_pitch").id]),
            "roll": int(self.model.jnt_qposadr[self.model.joint("teleop_base_roll").id]),
        }


def _vec(points: np.ndarray, start: int, end: int) -> np.ndarray:
    return points[end] - points[start]


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = _norm(vec)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return vec / norm


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = _norm(ba) * _norm(bc)
    if denom < 1e-8:
        return 180.0
    cosine = _clip(float(np.dot(ba, bc) / denom), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _signed_angle_degrees(v1: np.ndarray, v2: np.ndarray, normal: np.ndarray) -> float:
    u1 = _unit(v1)
    u2 = _unit(v2)
    plane_normal = _unit(normal)
    if _norm(u1) < 1e-8 or _norm(u2) < 1e-8 or _norm(plane_normal) < 1e-8:
        return 0.0
    cross = np.cross(u1, u2)
    sine = float(np.dot(cross, plane_normal))
    cosine = _clip(float(np.dot(u1, u2)), -1.0, 1.0)
    return math.degrees(math.atan2(sine, cosine))


def _project_to_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    plane_normal = _unit(normal)
    return vec - float(np.dot(vec, plane_normal)) * plane_normal


def _normalize_flex(angle_deg: float, full_extension_deg: float = 175.0) -> float:
    flex_deg = max(0.0, full_extension_deg - angle_deg)
    return _clip(flex_deg / 95.0, 0.0, 1.0)


def _normalize_spread(angle_deg: float, max_spread_deg: float = 25.0) -> float:
    return _clip(angle_deg / max_spread_deg, -1.0, 1.0)


def _map_unit_to_range(unit_value: float, low: float, high: float) -> float:
    return float(low + unit_value * (high - low))


def _map_signed_to_range(signed_value: float, low: float, high: float) -> float:
    midpoint = 0.5 * (low + high)
    half_range = 0.5 * (high - low)
    return float(midpoint + signed_value * half_range)


def _apply_deadzone(value: float, threshold: float = 0.08) -> float:
    if abs(value) <= threshold:
        return 0.0
    return float(value)


def _default_action(env: OrcaHandRight) -> np.ndarray:
    action = np.zeros(env.model.nu, dtype=np.float32)
    for actuator_id in range(env.model.nu):
        joint_id = int(env.model.actuator_trnid[actuator_id, 0])
        qpos_idx = int(env.model.jnt_qposadr[joint_id])
        default_qpos = float(env.model.qpos0[qpos_idx])
        action[actuator_id] = float(
            np.clip(default_qpos, env.action_low[actuator_id], env.action_high[actuator_id])
        )
    return action


def _landmarks_to_array(hand_landmarks) -> np.ndarray:
    if hasattr(hand_landmarks, "landmark"):
        landmarks = hand_landmarks.landmark
    else:
        landmarks = hand_landmarks
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float64,
    )


def _normalize_handedness_label(label: str | None) -> str:
    if not label:
        return "unknown"
    normalized = label.strip().lower()
    if normalized in {"left", "right"}:
        return normalized
    return "unknown"


def _effective_handedness(label: str | None, mirror: bool) -> str:
    handedness = _normalize_handedness_label(label)
    if not mirror:
        return handedness
    if handedness == "left":
        return "right"
    if handedness == "right":
        return "left"
    return handedness


def _unmirror_points(points: np.ndarray, mirror: bool) -> np.ndarray:
    if not mirror:
        return points
    corrected = points.copy()
    corrected[:, 0] = 1.0 - corrected[:, 0]
    return corrected


def _select_hand_solutions(results, target_hand: str, mirror: bool) -> SelectedHand | None:
    landmarks_list = getattr(results, "multi_hand_landmarks", None) or []
    handedness_list = getattr(results, "multi_handedness", None) or []
    for index, hand_landmarks in enumerate(landmarks_list):
        raw_label = None
        if index < len(handedness_list) and handedness_list[index].classification:
            raw_label = handedness_list[index].classification[0].label
        effective_label = _effective_handedness(raw_label, mirror)
        if target_hand == "either" or effective_label == target_hand:
            return SelectedHand(landmarks=hand_landmarks, label=effective_label)
    return None


def _select_hand_tasks(results, target_hand: str, mirror: bool) -> SelectedHand | None:
    landmarks_list = getattr(results, "hand_landmarks", None) or []
    handedness_list = getattr(results, "handedness", None) or []
    for index, hand_landmarks in enumerate(landmarks_list):
        raw_label = None
        if index < len(handedness_list) and handedness_list[index]:
            raw_label = handedness_list[index][0].category_name
        effective_label = _effective_handedness(raw_label, mirror)
        if target_hand == "either" or effective_label == target_hand:
            return SelectedHand(landmarks=hand_landmarks, label=effective_label)
    return None


def _draw_hand_landmarks_tasks(frame: np.ndarray, hand_landmarks) -> None:
    try:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
        )
    except AttributeError:
        pass


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


def _wrist_control_from_points(points: np.ndarray) -> float:
    wrist = points[WRIST]
    index_mcp = points[INDEX_MCP]
    pinky_mcp = points[PINKY_MCP]
    middle_mcp = points[MIDDLE_MCP]

    palm_across = _unit(index_mcp - pinky_mcp)
    palm_forward = _unit(middle_mcp - wrist)
    palm_normal = _unit(np.cross(palm_across, palm_forward))

    if _norm(palm_forward) < 1e-8 or _norm(palm_normal) < 1e-8:
        return 0.0

    # Approximate a single wrist DOF by blending palm yaw with palm roll.
    yaw_deg = math.degrees(math.atan2(palm_forward[0], max(1e-6, palm_forward[1])))
    roll_deg = math.degrees(math.atan2(palm_normal[0], max(1e-6, abs(palm_normal[2]))))
    wrist_deg = (0.65 * yaw_deg) + (0.35 * roll_deg)
    return _clip(wrist_deg / 65.0, -1.0, 1.0)


def _classify_palm_state(points: np.ndarray) -> str:
    wrist = points[WRIST]
    index_mcp = points[INDEX_MCP]
    pinky_mcp = points[PINKY_MCP]
    middle_mcp = points[MIDDLE_MCP]

    palm_across = _unit(index_mcp - pinky_mcp)
    palm_forward = _unit(middle_mcp - wrist)
    palm_normal = _unit(np.cross(palm_across, palm_forward))

    if _norm(palm_normal) < 1e-8:
        return "unknown"

    x_mag = abs(float(palm_normal[0]))
    z_mag = abs(float(palm_normal[2]))

    if z_mag >= x_mag:
        return "front_palm" if palm_normal[2] > 0.0 else "back_palm"
    return "right_side" if palm_normal[0] > 0.0 else "left_side"


def _palm_normal(points: np.ndarray) -> np.ndarray:
    wrist = points[WRIST]
    index_mcp = points[INDEX_MCP]
    pinky_mcp = points[PINKY_MCP]
    middle_mcp = points[MIDDLE_MCP]

    palm_across = _unit(index_mcp - pinky_mcp)
    palm_forward = _unit(middle_mcp - wrist)
    return _unit(np.cross(palm_across, palm_forward))


def _base_rotation_from_points(points: np.ndarray) -> tuple[float, float, float]:
    wrist = points[WRIST]
    index_mcp = points[INDEX_MCP]
    pinky_mcp = points[PINKY_MCP]
    middle_mcp = points[MIDDLE_MCP]

    palm_across = _unit(index_mcp - pinky_mcp)
    palm_forward = _unit(middle_mcp - wrist)
    palm_normal = _unit(np.cross(palm_across, palm_forward))

    if _norm(palm_across) < 1e-8 or _norm(palm_forward) < 1e-8 or _norm(palm_normal) < 1e-8:
        return 0.0, 0.0, 0.0

    yaw_deg = math.degrees(math.atan2(palm_forward[0], max(1e-6, palm_forward[1])))
    pitch_deg = math.degrees(math.atan2(-palm_forward[2], max(1e-6, abs(palm_forward[1]))))
    # Use the palm span and palm normal together so "flip" motion maps to roll.
    roll_deg = math.degrees(math.atan2(palm_across[2], max(1e-6, abs(palm_across[0]))))
    palm_twist_deg = math.degrees(math.atan2(palm_normal[0], max(1e-6, abs(palm_normal[2]))))

    base_yaw = _clip(yaw_deg / 55.0, -1.0, 1.0)
    base_pitch = _clip(-pitch_deg / 60.0, -1.0, 1.0)
    base_roll = _clip(((0.75 * roll_deg) + (0.25 * palm_twist_deg)) / 55.0, -1.0, 1.0)
    return base_yaw, base_pitch, base_roll


def extract_hand_features(points: np.ndarray) -> HandFeatures:
    palm_x = _vec(points, PINKY_MCP, INDEX_MCP)
    palm_y = _vec(points, WRIST, MIDDLE_MCP)
    palm_normal = np.cross(palm_x, palm_y)
    palm_normal_unit = _palm_normal(points)

    index_base = _project_to_plane(_vec(points, INDEX_MCP, INDEX_PIP), palm_normal)
    middle_base = _project_to_plane(_vec(points, MIDDLE_MCP, MIDDLE_PIP), palm_normal)
    ring_base = _project_to_plane(_vec(points, RING_MCP, RING_PIP), palm_normal)
    pinky_base = _project_to_plane(_vec(points, PINKY_MCP, PINKY_PIP), palm_normal)
    thumb_base = _project_to_plane(_vec(points, THUMB_CMC, THUMB_MCP), palm_normal)

    base_yaw, base_pitch, base_roll = _base_rotation_from_points(points)
    thumb_index_distance = _norm(points[THUMB_TIP] - points[INDEX_MCP])
    palm_width = max(_norm(points[INDEX_MCP] - points[PINKY_MCP]), 1e-6)
    thumb_open = _clip((thumb_index_distance / palm_width - 0.35) / 0.9, 0.0, 1.0)

    return HandFeatures(
        palm_state=_classify_palm_state(points),
        palm_normal_x=float(palm_normal_unit[0]),
        palm_normal_y=float(palm_normal_unit[1]),
        palm_normal_z=float(palm_normal_unit[2]),
        base_yaw=base_yaw,
        base_pitch=base_pitch,
        base_roll=base_roll,
        wrist=_wrist_control_from_points(points),
        pinky_abd=_normalize_spread(_signed_angle_degrees(middle_base, pinky_base, palm_normal)),
        pinky_mcp=_normalize_flex(_angle_degrees(points[WRIST], points[PINKY_MCP], points[PINKY_PIP])),
        pinky_pip=_normalize_flex(_angle_degrees(points[PINKY_MCP], points[PINKY_PIP], points[PINKY_DIP])),
        ring_abd=_normalize_spread(_signed_angle_degrees(middle_base, ring_base, palm_normal)),
        ring_mcp=_normalize_flex(_angle_degrees(points[WRIST], points[RING_MCP], points[RING_PIP])),
        ring_pip=_normalize_flex(_angle_degrees(points[RING_MCP], points[RING_PIP], points[RING_DIP])),
        middle_abd=0.0,
        middle_mcp=_normalize_flex(_angle_degrees(points[WRIST], points[MIDDLE_MCP], points[MIDDLE_PIP])),
        middle_pip=_normalize_flex(_angle_degrees(points[MIDDLE_MCP], points[MIDDLE_PIP], points[MIDDLE_DIP])),
        index_abd=_normalize_spread(_signed_angle_degrees(middle_base, index_base, palm_normal)),
        index_mcp=_normalize_flex(_angle_degrees(points[WRIST], points[INDEX_MCP], points[INDEX_PIP])),
        index_pip=_normalize_flex(_angle_degrees(points[INDEX_MCP], points[INDEX_PIP], points[INDEX_DIP])),
        thumb_cmc=(2.0 * thumb_open) - 1.0,
        thumb_abd=thumb_open,
        thumb_mcp=_normalize_flex(_angle_degrees(points[THUMB_CMC], points[THUMB_MCP], points[THUMB_IP])),
        thumb_pip=_normalize_flex(_angle_degrees(points[THUMB_MCP], points[THUMB_IP], points[THUMB_TIP])),
    )


def _build_action_name_to_index(env: BaseOrcaHandEnv) -> dict[str, int]:
    return {env.model.actuator(i).name: i for i in range(env.model.nu)}


def _set_named_signed(
    action: np.ndarray,
    env: BaseOrcaHandEnv,
    actuator_index: dict[str, int],
    actuator_name: str,
    signed_value: float,
) -> None:
    index = actuator_index.get(actuator_name)
    if index is None:
        return
    action[index] = _map_signed_to_range(signed_value, env.action_low[index], env.action_high[index])


def _set_named_unit(
    action: np.ndarray,
    env: BaseOrcaHandEnv,
    actuator_index: dict[str, int],
    actuator_name: str,
    unit_value: float,
) -> None:
    index = actuator_index.get(actuator_name)
    if index is None:
        return
    action[index] = _map_unit_to_range(unit_value, env.action_low[index], env.action_high[index])


def _apply_teleop_base_pose(action: np.ndarray, env: BaseOrcaHandEnv) -> None:
    if not isinstance(env, OrcaHandRightTeleop):
        return

    actuator_index = _build_action_name_to_index(env)
    yaw_actuator = actuator_index.get("teleop_base_yaw_actuator")
    pitch_actuator = actuator_index.get("teleop_base_pitch_actuator")
    roll_actuator = actuator_index.get("teleop_base_roll_actuator")
    if yaw_actuator is None or pitch_actuator is None or roll_actuator is None:
        return

    yaw_value = float(np.clip(action[yaw_actuator], env.action_low[yaw_actuator], env.action_high[yaw_actuator]))
    pitch_value = float(
        np.clip(action[pitch_actuator], env.action_low[pitch_actuator], env.action_high[pitch_actuator])
    )
    roll_value = float(
        np.clip(action[roll_actuator], env.action_low[roll_actuator], env.action_high[roll_actuator])
    )
    env.data.qpos[env._teleop_base_qpos_indices["yaw"]] = yaw_value
    env.data.qpos[env._teleop_base_qpos_indices["pitch"]] = pitch_value
    env.data.qpos[env._teleop_base_qpos_indices["roll"]] = roll_value
    env.data.qvel[env._teleop_base_qpos_indices["yaw"]] = 0.0
    env.data.qvel[env._teleop_base_qpos_indices["pitch"]] = 0.0
    env.data.qvel[env._teleop_base_qpos_indices["roll"]] = 0.0
    import mujoco

    mujoco.mj_forward(env.model, env.data)


def features_to_action(features: HandFeatures, env: BaseOrcaHandEnv) -> np.ndarray:
    action = _default_action(env)
    actuator_index = _build_action_name_to_index(env)

    _set_named_signed(action, env, actuator_index, "teleop_base_yaw_actuator", features.base_yaw)
    _set_named_signed(action, env, actuator_index, "teleop_base_pitch_actuator", features.base_pitch)
    _set_named_signed(action, env, actuator_index, "teleop_base_roll_actuator", features.base_roll)
    _set_named_signed(action, env, actuator_index, "right_wrist_actuator", features.wrist)
    _set_named_signed(action, env, actuator_index, "right_p-abd_actuator", features.pinky_abd)
    _set_named_unit(action, env, actuator_index, "right_p-mcp_actuator", features.pinky_mcp)
    _set_named_unit(action, env, actuator_index, "right_p-pip_actuator", features.pinky_pip)
    _set_named_signed(action, env, actuator_index, "right_r-abd_actuator", features.ring_abd)
    _set_named_unit(action, env, actuator_index, "right_r-mcp_actuator", features.ring_mcp)
    _set_named_unit(action, env, actuator_index, "right_r-pip_actuator", features.ring_pip)
    _set_named_signed(action, env, actuator_index, "right_m-abd_actuator", features.middle_abd)
    _set_named_unit(action, env, actuator_index, "right_m-mcp_actuator", features.middle_mcp)
    _set_named_unit(action, env, actuator_index, "right_m-pip_actuator", features.middle_pip)
    _set_named_signed(action, env, actuator_index, "right_i-abd_actuator", features.index_abd)
    _set_named_unit(action, env, actuator_index, "right_i-mcp_actuator", features.index_mcp)
    _set_named_unit(action, env, actuator_index, "right_i-pip_actuator", features.index_pip)
    _set_named_signed(action, env, actuator_index, "right_t-cmc_actuator", features.thumb_cmc)
    _set_named_unit(action, env, actuator_index, "right_t-abd_actuator", features.thumb_abd)
    _set_named_unit(action, env, actuator_index, "right_t-mcp_actuator", features.thumb_mcp)
    _set_named_unit(action, env, actuator_index, "right_t-pip_actuator", features.thumb_pip)
    return np.clip(action, env.action_low, env.action_high)


def _draw_status(
    frame: np.ndarray,
    fps: float,
    tracked: bool,
    handedness: str,
    calibrated: bool,
    palm_state: str,
    palm_normal: tuple[float, float, float],
    base_yaw: float,
    base_pitch: float,
    base_roll: float,
    help_text: str,
) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 255, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Tracked: {'yes' if tracked else 'no'}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 255, 120) if tracked else (70, 120, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Hand: {handedness}",
        (16, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 255, 120) if tracked else (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Calibrated: {'yes' if calibrated else 'no'}",
        (16, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 255, 120) if calibrated else (255, 210, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Palm: {palm_state}",
        (16, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            "Palm normal xyz: "
            f"{palm_normal[0]:+0.2f} {palm_normal[1]:+0.2f} {palm_normal[2]:+0.2f}"
        ),
        (16, 178),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Base y/p/r: {base_yaw:+0.2f} {base_pitch:+0.2f} {base_roll:+0.2f}",
        (16, 208),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        help_text,
        (16, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive the ORCA right hand with MediaPipe hand tracking."
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index.")
    parser.add_argument(
        "--version",
        default=None,
        help="Embodiment version to load, for example 'v1' or 'v2'.",
    )
    parser.add_argument(
        "--sim-render-mode",
        choices=["human", "rgb_array"],
        default="rgb_array",
        help="Use MuJoCo viewer or OpenCV simulator preview.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.18,
        help="Exponential smoothing factor in [0, 1]. Higher values follow faster.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="Maximum number of hands MediaPipe should track.",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.6,
        help="Minimum MediaPipe detection confidence.",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe tracking confidence.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1],
        default=1,
        help="MediaPipe Hands model complexity.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable selfie-style mirroring on the camera feed.",
    )
    parser.add_argument(
        "--hand-landmarker-model",
        default=None,
        help="Path to a MediaPipe `hand_landmarker.task` model file when using the tasks backend.",
    )
    parser.add_argument(
        "--target-hand",
        choices=["right", "left", "either"],
        default="right",
        help="Only drive the robot from the selected detected hand.",
    )
    parser.add_argument(
        "--fixed-base",
        action="store_true",
        help="Use the original fixed-base scene instead of the teleop scene with base yaw/pitch joints.",
    )
    parser.add_argument(
        "--base-gain",
        type=float,
        default=0.6,
        help="Gain applied to MediaPipe-derived base yaw/pitch before clipping.",
    )
    parser.add_argument(
        "--base-roll-gain",
        type=float,
        default=0.3,
        help="Gain applied specifically to MediaPipe-derived base roll.",
    )
    parser.add_argument(
        "--disable-auto-base",
        action="store_true",
        help="Keep the base controlled only by manual keys instead of MediaPipe-derived pose.",
    )
    parser.add_argument(
        "--disable-auto-calibration",
        action="store_true",
        help="Do not automatically treat the first stable tracked hand pose as the neutral base pose.",
    )
    parser.add_argument(
        "--sim-window-width",
        type=int,
        default=960,
        help="OpenCV simulator preview width when using `rgb_array`.",
    )
    parser.add_argument(
        "--sim-window-height",
        type=int,
        default=720,
        help="OpenCV simulator preview height when using `rgb_array`.",
    )
    parser.add_argument(
        "--camera-window-width",
        type=int,
        default=960,
        help="OpenCV camera preview width.",
    )
    parser.add_argument(
        "--camera-window-height",
        type=int,
        default=720,
        help="OpenCV camera preview height.",
    )
    args = parser.parse_args()

    smoothing = _clip(args.smoothing, 0.0, 1.0)
    mirror = not args.no_mirror
    base_gain = max(0.0, args.base_gain)
    base_roll_gain = max(0.0, args.base_roll_gain)

    env_cls = OrcaHandRight if args.fixed_base else OrcaHandRightTeleop
    env = env_cls(render_mode=args.sim_render_mode, version=args.version)
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        env.close()
        raise SystemExit(
            f"Could not open camera {args.camera_id}. Try a different `--camera-id`."
        )

    action = _default_action(env)
    env.reset()
    base_manual_yaw = 0.0
    base_manual_pitch = 0.0
    base_manual_roll = 0.0
    base_calibration = BaseCalibration()

    prev_time = time.perf_counter()

    cv2.namedWindow("MediaPipe Teleop", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        "MediaPipe Teleop",
        max(320, args.camera_window_width),
        max(240, args.camera_window_height),
    )
    if args.sim_render_mode == "rgb_array":
        cv2.namedWindow("ORCA Simulator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "ORCA Simulator",
            max(320, args.sim_window_width),
            max(240, args.sim_window_height),
        )

    try:
        if MP_BACKEND == "solutions":
            hand_tracker = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=args.max_hands,
                min_detection_confidence=args.detection_confidence,
                min_tracking_confidence=args.tracking_confidence,
                model_complexity=args.model_complexity,
            )
        else:
            model_path = (
                Path(args.hand_landmarker_model).expanduser().resolve()
                if args.hand_landmarker_model
                else _resolve_default_task_model_path()
            )
            if model_path is None or not model_path.exists():
                env.close()
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit(
                    "This MediaPipe install uses the newer tasks backend and needs a local "
                    "`hand_landmarker.task` model file. Download that model, place it next to "
                    "`mediapipe_teleop.py`, or pass `--hand-landmarker-model <path>`."
                )

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.IMAGE,
                num_hands=args.max_hands,
                min_hand_detection_confidence=args.detection_confidence,
                min_hand_presence_confidence=args.detection_confidence,
                min_tracking_confidence=args.tracking_confidence,
            )
            hand_tracker = HandLandmarker.create_from_options(options)

        with hand_tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                tracked = False
                tracked_label = "none"
                displayed_palm_state = "unknown"
                displayed_palm_normal = (0.0, 0.0, 0.0)
                displayed_base_yaw = 0.0
                displayed_base_pitch = 0.0
                displayed_base_roll = 0.0
                current_raw_base = None
                if MP_BACKEND == "solutions":
                    results = hand_tracker.process(rgb_frame)
                    selected = _select_hand_solutions(results, args.target_hand, mirror)
                    if selected is not None:
                        tracked = True
                        tracked_label = selected.label
                        hand_landmarks = selected.landmarks
                        points = _unmirror_points(_landmarks_to_array(hand_landmarks), mirror)
                        features = extract_hand_features(points)
                        displayed_palm_state = features.palm_state
                        displayed_palm_normal = (
                            features.palm_normal_x,
                            features.palm_normal_y,
                            features.palm_normal_z,
                        )
                        current_raw_base = (
                            features.base_yaw,
                            features.base_pitch,
                            features.base_roll,
                        )
                        if args.disable_auto_base:
                            features.base_yaw = 0.0
                            features.base_pitch = 0.0
                            features.base_roll = 0.0
                        else:
                            if not base_calibration.calibrated and not args.disable_auto_calibration:
                                base_calibration = BaseCalibration(
                                    yaw=features.base_yaw,
                                    pitch=features.base_pitch,
                                    roll=features.base_roll,
                                    calibrated=True,
                                )
                            features.base_yaw = _apply_deadzone(features.base_yaw - base_calibration.yaw)
                            features.base_pitch = _apply_deadzone(features.base_pitch - base_calibration.pitch)
                            features.base_roll = _apply_deadzone(features.base_roll - base_calibration.roll)
                            features.base_yaw = _clip(features.base_yaw * base_gain, -1.0, 1.0)
                            features.base_pitch = _clip(features.base_pitch * base_gain, -1.0, 1.0)
                            features.base_roll = _clip(features.base_roll * base_roll_gain, -1.0, 1.0)
                        displayed_base_yaw = features.base_yaw
                        displayed_base_pitch = features.base_pitch
                        displayed_base_roll = features.base_roll
                        target_action = features_to_action(features, env)
                        action = ((1.0 - smoothing) * action) + (smoothing * target_action)
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )
                else:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    results = hand_tracker.detect(mp_image)
                    selected = _select_hand_tasks(results, args.target_hand, mirror)
                    if selected is not None:
                        tracked = True
                        tracked_label = selected.label
                        hand_landmarks = selected.landmarks
                        points = _unmirror_points(_landmarks_to_array(hand_landmarks), mirror)
                        features = extract_hand_features(points)
                        displayed_palm_state = features.palm_state
                        displayed_palm_normal = (
                            features.palm_normal_x,
                            features.palm_normal_y,
                            features.palm_normal_z,
                        )
                        current_raw_base = (
                            features.base_yaw,
                            features.base_pitch,
                            features.base_roll,
                        )
                        if args.disable_auto_base:
                            features.base_yaw = 0.0
                            features.base_pitch = 0.0
                            features.base_roll = 0.0
                        else:
                            if not base_calibration.calibrated and not args.disable_auto_calibration:
                                base_calibration = BaseCalibration(
                                    yaw=features.base_yaw,
                                    pitch=features.base_pitch,
                                    roll=features.base_roll,
                                    calibrated=True,
                                )
                            features.base_yaw = _apply_deadzone(features.base_yaw - base_calibration.yaw)
                            features.base_pitch = _apply_deadzone(features.base_pitch - base_calibration.pitch)
                            features.base_roll = _apply_deadzone(features.base_roll - base_calibration.roll)
                            features.base_yaw = _clip(features.base_yaw * base_gain, -1.0, 1.0)
                            features.base_pitch = _clip(features.base_pitch * base_gain, -1.0, 1.0)
                            features.base_roll = _clip(features.base_roll * base_roll_gain, -1.0, 1.0)
                        displayed_base_yaw = features.base_yaw
                        displayed_base_pitch = features.base_pitch
                        displayed_base_roll = features.base_roll
                        target_action = features_to_action(features, env)
                        action = ((1.0 - smoothing) * action) + (smoothing * target_action)
                        _draw_hand_landmarks_tasks(frame, hand_landmarks)

                if not tracked:
                    neutral_action = _default_action(env)
                    actuator_index = _build_action_name_to_index(env)
                    if "teleop_base_yaw_actuator" in actuator_index:
                        displayed_base_yaw = base_manual_yaw
                        displayed_base_pitch = base_manual_pitch
                        displayed_base_roll = base_manual_roll
                        yaw_index = actuator_index["teleop_base_yaw_actuator"]
                        pitch_index = actuator_index["teleop_base_pitch_actuator"]
                        roll_index = actuator_index["teleop_base_roll_actuator"]
                        neutral_action[yaw_index] = _map_signed_to_range(
                            base_manual_yaw,
                            env.action_low[yaw_index],
                            env.action_high[yaw_index],
                        )
                        neutral_action[pitch_index] = _map_signed_to_range(
                            base_manual_pitch,
                            env.action_low[pitch_index],
                            env.action_high[pitch_index],
                        )
                        neutral_action[roll_index] = _map_signed_to_range(
                            base_manual_roll,
                            env.action_low[roll_index],
                            env.action_high[roll_index],
                        )
                    action = ((1.0 - smoothing) * action) + (smoothing * neutral_action)

                _apply_teleop_base_pose(action, env)
                env.step(action)

                if args.sim_render_mode == "rgb_array":
                    sim_frame = env.render()
                    if sim_frame is not None:
                        cv2.imshow("ORCA Simulator", cv2.cvtColor(sim_frame, cv2.COLOR_RGB2BGR))

                now = time.perf_counter()
                dt = max(now - prev_time, 1e-6)
                prev_time = now
                _draw_status(
                    frame,
                    fps=1.0 / dt,
                    tracked=tracked,
                    handedness=tracked_label,
                    calibrated=base_calibration.calibrated,
                    palm_state=displayed_palm_state,
                    palm_normal=displayed_palm_normal,
                    base_yaw=displayed_base_yaw,
                    base_pitch=displayed_base_pitch,
                    base_roll=displayed_base_roll,
                    help_text="q quit | r reset | c calibrate | m mirror | j/l yaw | i/k pitch | u/o roll",
                )
                cv2.imshow("MediaPipe Teleop", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break
                if key == ord("r"):
                    env.reset()
                    action = _default_action(env)
                    base_manual_yaw = 0.0
                    base_manual_pitch = 0.0
                    base_manual_roll = 0.0
                    base_calibration = BaseCalibration()
                if key == ord("c") and tracked and current_raw_base is not None:
                    base_calibration = BaseCalibration(
                        yaw=current_raw_base[0],
                        pitch=current_raw_base[1],
                        roll=current_raw_base[2],
                        calibrated=True,
                    )
                if key == ord("m"):
                    mirror = not mirror
                if key == ord("j"):
                    base_manual_yaw = _clip(base_manual_yaw - 0.15, -1.0, 1.0)
                if key == ord("l"):
                    base_manual_yaw = _clip(base_manual_yaw + 0.15, -1.0, 1.0)
                if key == ord("i"):
                    base_manual_pitch = _clip(base_manual_pitch + 0.15, -1.0, 1.0)
                if key == ord("k"):
                    base_manual_pitch = _clip(base_manual_pitch - 0.15, -1.0, 1.0)
                if key == ord("u"):
                    base_manual_roll = _clip(base_manual_roll - 0.15, -1.0, 1.0)
                if key == ord("o"):
                    base_manual_roll = _clip(base_manual_roll + 0.15, -1.0, 1.0)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
