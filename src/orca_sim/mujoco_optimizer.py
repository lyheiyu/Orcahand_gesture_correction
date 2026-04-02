from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from orca_sim import OrcaHandRight
from orca_sim.gesture_features import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    THUMB_TIP,
    WRIST,
    OrcaFeatureProjector,
    normalize_landmarks,
    palm_normal_vector,
)

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ModuleNotFoundError:
    minimize = None
    SCIPY_AVAILABLE = False


@dataclass
class OptimizationWeights:
    landmark: float = 1.0
    palm: float = 0.2
    prior: float = 0.3
    temporal: float = 0.1
    default_pose: float = 0.15
    boundary: float = 0.05


@dataclass
class OptimizationResult:
    action: np.ndarray
    loss: float
    success: bool
    iterations: int
    method: str
    loss_terms: dict[str, float]


class MujocoHandPoseOptimizer:
    """Fit ORCA joint controls to MediaPipe landmarks using MuJoCo forward kinematics.

    This is an optimization-based upgrade over the heuristic projection in
    `gesture_features.py`. It does not perform full differentiable physics,
    but it does optimize directly in the ORCA hand state space while enforcing
    MuJoCo joint/control bounds and evaluating candidate poses through
    `mujoco.mj_forward`.
    """

    _POINT_LABELS = (
        "wrist",
        "thumb_tip",
        "index_mcp",
        "index_tip",
        "middle_mcp",
        "middle_tip",
        "pinky_mcp",
        "pinky_tip",
    )

    def __init__(self, version: str | None = None) -> None:
        self.env = OrcaHandRight(render_mode=None, version=version)
        self.projector = OrcaFeatureProjector(version=version)
        self._joint_qpos_indices = self._build_joint_qpos_indices()
        self._body_ids = self._build_body_ids()
        self._default_action = self._compute_default_action()

    def close(self) -> None:
        self.projector.close()
        self.env.close()

    def __enter__(self) -> "MujocoHandPoseOptimizer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def optimize(
        self,
        points: np.ndarray,
        *,
        initial_action: np.ndarray | None = None,
        prev_action: np.ndarray | None = None,
        weights: OptimizationWeights | None = None,
        max_iterations: int = 120,
    ) -> OptimizationResult:
        weights = weights or OptimizationWeights()
        target_points = normalize_landmarks(points)
        target_sparse = self._target_sparse_points(target_points)
        target_palm = palm_normal_vector(target_points)

        if initial_action is None:
            initial_action = self.projector.corrected_vector(points).astype(np.float64)
        else:
            initial_action = np.asarray(initial_action, dtype=np.float64).copy()

        if prev_action is not None:
            prev_action = np.asarray(prev_action, dtype=np.float64).copy()

        initial_action = np.clip(initial_action, self.env.action_low, self.env.action_high)

        def loss_terms(action: np.ndarray) -> dict[str, float]:
            current_sparse, current_palm = self._forward_sparse_points(action)
            landmark_loss = self._landmark_loss(current_sparse, target_sparse)
            palm_loss = float(np.sum((current_palm - target_palm) ** 2))
            prior_loss = float(np.sum((action - initial_action) ** 2))
            default_pose_loss = float(np.sum((action - self._default_action) ** 2))
            temporal_loss = 0.0
            if prev_action is not None:
                temporal_loss = float(np.sum((action - prev_action) ** 2))
            normalized = (action - self.env.action_low) / np.maximum(
                self.env.action_high - self.env.action_low, 1e-6
            )
            boundary_loss = float(
                np.sum((np.clip(0.1 - normalized, 0.0, None) ** 2) + (np.clip(normalized - 0.9, 0.0, None) ** 2))
            )
            total = (
                weights.landmark * landmark_loss
                + weights.palm * palm_loss
                + weights.prior * prior_loss
                + weights.temporal * temporal_loss
                + weights.default_pose * default_pose_loss
                + weights.boundary * boundary_loss
            )
            return {
                "landmark": landmark_loss,
                "palm": palm_loss,
                "prior": prior_loss,
                "temporal": temporal_loss,
                "default_pose": default_pose_loss,
                "boundary": boundary_loss,
                "total": float(total),
            }

        def objective(action: np.ndarray) -> float:
            return loss_terms(action)["total"]

        if SCIPY_AVAILABLE:
            bounds = list(zip(self.env.action_low.astype(float), self.env.action_high.astype(float)))
            result = minimize(
                objective,
                x0=initial_action,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max_iterations},
            )
            action = np.clip(np.asarray(result.x, dtype=np.float64), self.env.action_low, self.env.action_high)
            return OptimizationResult(
                action=action.astype(np.float32),
                loss=float(result.fun),
                success=bool(result.success),
                iterations=int(getattr(result, "nit", 0)),
                method="scipy_lbfgsb",
                loss_terms=loss_terms(action),
            )

        action, loss, iterations = self._coordinate_descent(objective, initial_action, max_iterations)
        return OptimizationResult(
            action=action.astype(np.float32),
            loss=loss,
            success=True,
            iterations=iterations,
            method="coordinate_descent",
            loss_terms=loss_terms(action),
        )

    def _coordinate_descent(
        self,
        objective,
        initial_action: np.ndarray,
        max_iterations: int,
    ) -> tuple[np.ndarray, float, int]:
        action = initial_action.copy()
        best_loss = float(objective(action))
        step_sizes = np.array([0.20, 0.10, 0.05, 0.025], dtype=np.float64)
        iterations = 0

        for step_scale in step_sizes:
            improved = True
            while improved and iterations < max_iterations:
                improved = False
                for dim in range(action.shape[0]):
                    iterations += 1
                    base = action[dim]
                    dim_range = float(self.env.action_high[dim] - self.env.action_low[dim])
                    step = step_scale * dim_range
                    for direction in (-1.0, 1.0):
                        candidate = action.copy()
                        candidate[dim] = np.clip(base + direction * step, self.env.action_low[dim], self.env.action_high[dim])
                        candidate_loss = float(objective(candidate))
                        if candidate_loss + 1e-12 < best_loss:
                            action = candidate
                            best_loss = candidate_loss
                            improved = True
        return action, best_loss, iterations

    def _build_joint_qpos_indices(self) -> dict[int, int]:
        indices: dict[int, int] = {}
        for actuator_id in range(self.env.model.nu):
            joint_id = int(self.env.model.actuator_trnid[actuator_id, 0])
            indices[actuator_id] = int(self.env.model.jnt_qposadr[joint_id])
        return indices

    def _compute_default_action(self) -> np.ndarray:
        action = np.zeros(self.env.model.nu, dtype=np.float64)
        for actuator_id, qpos_index in self._joint_qpos_indices.items():
            action[actuator_id] = float(
                np.clip(
                    self.env.model.qpos0[qpos_index],
                    self.env.action_low[actuator_id],
                    self.env.action_high[actuator_id],
                )
            )
        return action

    def _build_body_ids(self) -> dict[str, int]:
        names = {
            "wrist": "right_R-Carpals_8d1f1041",
            "thumb_tip": "right_T-DP_b7429e50",
            "index_mcp": "right_I-AP-R_d95d02d1",
            "index_tip": "right_I-FingerTipAssembly_ec49c16c",
            "middle_mcp": "right_M-AP_e04a96f2",
            "middle_tip": "right_M-FingerTipAssembly_34afb748",
            "pinky_mcp": "right_P-AP_f5e42b61",
            "pinky_tip": "right_P-FingerTipAssembly_cd219176",
        }
        return {label: self.env.model.body(name).id for label, name in names.items()}

    def _target_sparse_points(self, normalized_points: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "wrist": normalized_points[WRIST],
            "thumb_tip": normalized_points[THUMB_TIP],
            "index_mcp": normalized_points[INDEX_MCP],
            "index_tip": normalized_points[INDEX_TIP],
            "middle_mcp": normalized_points[MIDDLE_MCP],
            "middle_tip": normalized_points[MIDDLE_TIP],
            "pinky_mcp": normalized_points[PINKY_MCP],
            "pinky_tip": normalized_points[PINKY_TIP],
        }

    def _forward_sparse_points(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        self._set_pose_from_action(action)
        world_points = {
            label: self.env.data.xpos[body_id].copy()
            for label, body_id in self._body_ids.items()
        }

        wrist = world_points["wrist"]
        translated = {label: value - wrist for label, value in world_points.items()}
        palm_width = np.linalg.norm(translated["index_mcp"] - translated["pinky_mcp"])
        palm_length = np.linalg.norm(translated["middle_mcp"])
        scale = max(float(palm_width), float(palm_length), 1e-6)
        normalized = {label: value / scale for label, value in translated.items()}

        palm_across = normalized["index_mcp"] - normalized["pinky_mcp"]
        palm_forward = normalized["middle_mcp"] - normalized["wrist"]
        palm_normal = np.cross(palm_across, palm_forward)
        palm_norm = np.linalg.norm(palm_normal)
        if palm_norm > 1e-8:
            palm_normal = palm_normal / palm_norm
        else:
            palm_normal = np.zeros(3, dtype=np.float64)
        return normalized, palm_normal

    def _set_pose_from_action(self, action: np.ndarray) -> None:
        mujoco.mj_resetData(self.env.model, self.env.data)
        clipped = np.clip(np.asarray(action, dtype=np.float64), self.env.action_low, self.env.action_high)
        for actuator_id, qpos_index in self._joint_qpos_indices.items():
            self.env.data.qpos[qpos_index] = clipped[actuator_id]
        self.env.data.ctrl[:] = clipped
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def _landmark_loss(
        self,
        predicted: dict[str, np.ndarray],
        target: dict[str, np.ndarray],
    ) -> float:
        total = 0.0
        for label in self._POINT_LABELS:
            total += float(np.sum((predicted[label] - target[label]) ** 2))
        return total
