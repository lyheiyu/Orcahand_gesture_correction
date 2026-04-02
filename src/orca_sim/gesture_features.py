from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orca_sim import OrcaHandRight

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

PALM_STATES = ("front_palm", "back_palm", "left_side", "right_side", "unknown")


@dataclass
class HandFeatures:
    palm_state: str
    palm_normal_x: float
    palm_normal_y: float
    palm_normal_z: float
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


def _classify_palm_state(points: np.ndarray) -> str:
    palm_normal = palm_normal_vector(points)
    if _norm(palm_normal) < 1e-8:
        return "unknown"

    x_mag = abs(float(palm_normal[0]))
    z_mag = abs(float(palm_normal[2]))
    if z_mag >= x_mag:
        return "front_palm" if palm_normal[2] > 0.0 else "back_palm"
    return "right_side" if palm_normal[0] > 0.0 else "left_side"


def palm_normal_vector(points: np.ndarray) -> np.ndarray:
    palm_across = _unit(_vec(points, PINKY_MCP, INDEX_MCP))
    palm_forward = _unit(_vec(points, WRIST, MIDDLE_MCP))
    return _unit(np.cross(palm_across, palm_forward))


def _wrist_control_from_points(points: np.ndarray) -> float:
    palm_across = _unit(_vec(points, PINKY_MCP, INDEX_MCP))
    palm_forward = _unit(_vec(points, WRIST, MIDDLE_MCP))
    palm_normal = _unit(np.cross(palm_across, palm_forward))

    if _norm(palm_forward) < 1e-8 or _norm(palm_normal) < 1e-8:
        return 0.0

    yaw_deg = math.degrees(math.atan2(palm_forward[0], max(1e-6, palm_forward[1])))
    roll_deg = math.degrees(math.atan2(palm_normal[0], max(1e-6, abs(palm_normal[2]))))
    wrist_deg = (0.65 * yaw_deg) + (0.35 * roll_deg)
    return _clip(wrist_deg / 65.0, -1.0, 1.0)


def normalize_landmarks(points: np.ndarray) -> np.ndarray:
    normalized = np.asarray(points, dtype=np.float64).copy()
    if normalized.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {normalized.shape}")

    origin = normalized[WRIST].copy()
    normalized -= origin

    palm_width = _norm(normalized[INDEX_MCP] - normalized[PINKY_MCP])
    palm_length = _norm(normalized[MIDDLE_MCP] - normalized[WRIST])
    scale = max(palm_width, palm_length, 1e-6)
    normalized /= scale
    return normalized


def extract_hand_features(points: np.ndarray) -> HandFeatures:
    points = np.asarray(points, dtype=np.float64)
    palm_normal = np.cross(_vec(points, PINKY_MCP, INDEX_MCP), _vec(points, WRIST, MIDDLE_MCP))

    index_base = _project_to_plane(_vec(points, INDEX_MCP, INDEX_PIP), palm_normal)
    middle_base = _project_to_plane(_vec(points, MIDDLE_MCP, MIDDLE_PIP), palm_normal)
    ring_base = _project_to_plane(_vec(points, RING_MCP, RING_PIP), palm_normal)
    pinky_base = _project_to_plane(_vec(points, PINKY_MCP, PINKY_PIP), palm_normal)

    thumb_index_distance = _norm(points[THUMB_TIP] - points[INDEX_MCP])
    palm_width = max(_norm(points[INDEX_MCP] - points[PINKY_MCP]), 1e-6)
    thumb_open = _clip((thumb_index_distance / palm_width - 0.35) / 0.9, 0.0, 1.0)
    palm_normal_unit = palm_normal_vector(points)

    return HandFeatures(
        palm_state=_classify_palm_state(points),
        palm_normal_x=float(palm_normal_unit[0]),
        palm_normal_y=float(palm_normal_unit[1]),
        palm_normal_z=float(palm_normal_unit[2]),
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


class OrcaFeatureProjector:
    def __init__(self, version: str | None = None) -> None:
        self.env = OrcaHandRight(render_mode=None, version=version)
        self._actuator_index = {
            self.env.model.actuator(i).name: i for i in range(self.env.model.nu)
        }

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "OrcaFeatureProjector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def raw_vector(self, points: np.ndarray) -> np.ndarray:
        normalized = normalize_landmarks(points)
        return normalized.astype(np.float32).reshape(-1)

    def geometry_vector(self, points: np.ndarray) -> np.ndarray:
        features = extract_hand_features(points)
        palm_one_hot = np.zeros(len(PALM_STATES), dtype=np.float32)
        palm_one_hot[PALM_STATES.index(features.palm_state)] = 1.0
        values = np.array(
            [
                features.palm_normal_x,
                features.palm_normal_y,
                features.palm_normal_z,
                features.wrist,
                features.pinky_abd,
                features.pinky_mcp,
                features.pinky_pip,
                features.ring_abd,
                features.ring_mcp,
                features.ring_pip,
                features.middle_abd,
                features.middle_mcp,
                features.middle_pip,
                features.index_abd,
                features.index_mcp,
                features.index_pip,
                features.thumb_cmc,
                features.thumb_abd,
                features.thumb_mcp,
                features.thumb_pip,
            ],
            dtype=np.float32,
        )
        return np.concatenate([values, palm_one_hot], dtype=np.float32)

    def corrected_vector(self, points: np.ndarray) -> np.ndarray:
        features = extract_hand_features(points)
        action = np.zeros(self.env.model.nu, dtype=np.float32)

        self._set_signed(action, "right_wrist_actuator", features.wrist)
        self._set_signed(action, "right_p-abd_actuator", features.pinky_abd)
        self._set_unit(action, "right_p-mcp_actuator", features.pinky_mcp)
        self._set_unit(action, "right_p-pip_actuator", features.pinky_pip)
        self._set_signed(action, "right_r-abd_actuator", features.ring_abd)
        self._set_unit(action, "right_r-mcp_actuator", features.ring_mcp)
        self._set_unit(action, "right_r-pip_actuator", features.ring_pip)
        self._set_signed(action, "right_m-abd_actuator", features.middle_abd)
        self._set_unit(action, "right_m-mcp_actuator", features.middle_mcp)
        self._set_unit(action, "right_m-pip_actuator", features.middle_pip)
        self._set_signed(action, "right_i-abd_actuator", features.index_abd)
        self._set_unit(action, "right_i-mcp_actuator", features.index_mcp)
        self._set_unit(action, "right_i-pip_actuator", features.index_pip)
        self._set_signed(action, "right_t-cmc_actuator", features.thumb_cmc)
        self._set_unit(action, "right_t-abd_actuator", features.thumb_abd)
        self._set_unit(action, "right_t-mcp_actuator", features.thumb_mcp)
        self._set_unit(action, "right_t-pip_actuator", features.thumb_pip)
        return np.clip(action, self.env.action_low, self.env.action_high)

    def all_feature_groups(self, points: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "raw": self.raw_vector(points),
            "geom": self.geometry_vector(points),
            "corrected": self.corrected_vector(points),
        }

    def _set_signed(self, action: np.ndarray, actuator_name: str, value: float) -> None:
        index = self._actuator_index[actuator_name]
        low = self.env.action_low[index]
        high = self.env.action_high[index]
        midpoint = 0.5 * (low + high)
        half_range = 0.5 * (high - low)
        action[index] = midpoint + (value * half_range)

    def _set_unit(self, action: np.ndarray, actuator_name: str, value: float) -> None:
        index = self._actuator_index[actuator_name]
        low = self.env.action_low[index]
        high = self.env.action_high[index]
        action[index] = low + value * (high - low)
