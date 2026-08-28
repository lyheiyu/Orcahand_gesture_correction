from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generate_joint_angle_baseline import JOINT_DEFINITIONS, joint_angle_vector


FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}


@dataclass(frozen=True)
class ConfidenceParameters:
    jump_scale: np.ndarray
    prediction_scale: np.ndarray
    geometry_scale: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    strength: float = 1.0


def angle_sequence(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angles = []
    validity = []
    for frame in np.asarray(points):
        frame_angles, frame_validity = joint_angle_vector(frame)
        angles.append(frame_angles)
        validity.append(frame_validity)
    return np.asarray(angles, dtype=np.float64), np.asarray(validity, dtype=bool)


def geometry_inconsistency(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    lengths = np.zeros((len(points), len(JOINT_DEFINITIONS), 2), dtype=np.float64)
    valid = np.ones((len(points), len(JOINT_DEFINITIONS)), dtype=bool)
    for joint_index, (_, proximal, joint, distal) in enumerate(JOINT_DEFINITIONS):
        lengths[:, joint_index, 0] = np.linalg.norm(points[:, proximal] - points[:, joint], axis=1)
        lengths[:, joint_index, 1] = np.linalg.norm(points[:, distal] - points[:, joint], axis=1)
        valid[:, joint_index] = np.all(lengths[:, joint_index] > 1e-8, axis=1)

    error = np.zeros((len(points), len(JOINT_DEFINITIONS)), dtype=np.float64)
    if len(points) > 1:
        ratio = np.log(
            np.maximum(lengths[1:], 1e-8) / np.maximum(lengths[:-1], 1e-8)
        )
        error[1:] = np.max(np.abs(ratio), axis=2)
    error[~valid] = 20.0
    return error, valid


def affected_joint_mask(landmark_mask: np.ndarray) -> np.ndarray:
    landmark_mask = np.asarray(landmark_mask, dtype=bool)
    output = np.zeros((len(landmark_mask), len(JOINT_DEFINITIONS)), dtype=bool)
    for joint_index, (_, proximal, joint, distal) in enumerate(JOINT_DEFINITIONS):
        output[:, joint_index] = np.any(landmark_mask[:, (proximal, joint, distal)], axis=1)
    return output


def estimate_confidence_parameters(
    development_points: list[np.ndarray], *, strength: float = 1.0
) -> ConfidenceParameters:
    angle_values = []
    jumps = [[] for _ in JOINT_DEFINITIONS]
    prediction_residuals = [[] for _ in JOINT_DEFINITIONS]
    geometry_errors = [[] for _ in JOINT_DEFINITIONS]

    for points in development_points:
        angles, valid = angle_sequence(points)
        geometry, geometry_valid = geometry_inconsistency(points)
        angle_values.append(angles)
        for joint_index in range(len(JOINT_DEFINITIONS)):
            valid_jump = valid[1:, joint_index] & valid[:-1, joint_index]
            jumps[joint_index].extend(np.abs(np.diff(angles[:, joint_index]))[valid_jump])
            if len(angles) > 2:
                valid_prediction = (
                    valid[2:, joint_index]
                    & valid[1:-1, joint_index]
                    & valid[:-2, joint_index]
                )
                prediction = 2.0 * angles[1:-1, joint_index] - angles[:-2, joint_index]
                residual = np.abs(angles[2:, joint_index] - prediction)
                prediction_residuals[joint_index].extend(residual[valid_prediction])
            valid_geometry = geometry_valid[:, joint_index] & np.isfinite(geometry[:, joint_index])
            geometry_errors[joint_index].extend(geometry[valid_geometry, joint_index])

    stacked_angles = np.concatenate(angle_values, axis=0)

    def robust_scale(collections: list[list[float]], floor: float) -> np.ndarray:
        values = []
        for collection in collections:
            array = np.asarray(collection, dtype=float)
            values.append(max(float(np.percentile(array, 95)) if len(array) else floor, floor))
        return np.asarray(values, dtype=np.float64)

    lower = np.maximum(np.percentile(stacked_angles, 0.5, axis=0) - 5.0, 0.0)
    upper = np.minimum(np.percentile(stacked_angles, 99.5, axis=0) + 5.0, 180.0)
    return ConfidenceParameters(
        jump_scale=robust_scale(jumps, 2.0),
        prediction_scale=robust_scale(prediction_residuals, 2.0),
        geometry_scale=robust_scale(geometry_errors, 0.02),
        lower_bound=lower,
        upper_bound=upper,
        strength=float(strength),
    )


def recover_confidence_weighted(
    observed_angles: np.ndarray,
    observed_points: np.ndarray,
    parameters: ConfidenceParameters,
    *,
    oracle_joint_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    observed_angles = np.asarray(observed_angles, dtype=np.float64)
    geometry_error, geometry_valid = geometry_inconsistency(observed_points)
    recovered = np.zeros_like(observed_angles)
    confidence = np.zeros_like(observed_angles)

    for frame_index in range(len(observed_angles)):
        if frame_index == 0:
            prediction = observed_angles[frame_index].copy()
        elif frame_index == 1:
            prediction = recovered[frame_index - 1].copy()
        else:
            prediction = 2.0 * recovered[frame_index - 1] - recovered[frame_index - 2]

        if oracle_joint_mask is not None:
            weight = (~oracle_joint_mask[frame_index]).astype(np.float64)
        elif frame_index == 0:
            weight = geometry_valid[frame_index].astype(np.float64)
        else:
            angle_jump = np.abs(observed_angles[frame_index] - recovered[frame_index - 1])
            prediction_disagreement = np.abs(observed_angles[frame_index] - prediction)
            score = parameters.strength * (
                0.35 * angle_jump / parameters.jump_scale
                + 0.30 * geometry_error[frame_index] / parameters.geometry_scale
                + 0.35 * prediction_disagreement / parameters.prediction_scale
            )
            weight = np.exp(-np.clip(score, 0.0, 50.0))
            weight[~geometry_valid[frame_index]] = 0.0

        confidence[frame_index] = np.clip(weight, 0.0, 1.0)
        fused = confidence[frame_index] * observed_angles[frame_index] + (
            1.0 - confidence[frame_index]
        ) * prediction
        recovered[frame_index] = np.clip(
            fused, parameters.lower_bound, parameters.upper_bound
        )
    return recovered, confidence


def corrupt_landmarks(
    clean: np.ndarray,
    corruption_type: str,
    *,
    seed: int,
    finger_group: str = "index",
    duration: int = 0,
    gaussian_sigma: float = 0.03,
    spike_magnitude: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    clean = np.asarray(clean, dtype=np.float64)
    corrupted = clean.copy()
    mask = np.zeros((len(clean), 21), dtype=bool)
    manifest = []
    rng = np.random.default_rng(seed)

    if corruption_type == "gaussian":
        noise = rng.normal(0.0, gaussian_sigma, size=clean.shape)
        corrupted += noise
        mask[:] = True
        for frame_index in range(len(clean)):
            for landmark_id in range(21):
                manifest.append(
                    {
                        "frame_index": frame_index,
                        "landmark_id": landmark_id,
                        "finger_group": "all",
                        "corruption_type": "gaussian",
                        "severity": "medium",
                        "duration": 0,
                        "random_seed": seed,
                    }
                )
        return corrupted, mask, manifest

    if duration <= 0:
        raise ValueError("A positive duration is required for spike/occlusion corruption.")
    start_max = max(len(clean) - duration, 0)
    start = int(rng.integers(1 if start_max >= 1 else 0, start_max + 1))

    if corruption_type == "spike":
        finger_group = sorted(FINGER_LANDMARKS)[int(rng.integers(0, len(FINGER_LANDMARKS)))]
        # Choose a landmark that directly participates in JointAngle-11.
        candidates = FINGER_LANDMARKS[finger_group][:-1]
        landmark_ids = (int(candidates[int(rng.integers(0, len(candidates)))]),)
        direction = rng.normal(size=3)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        offset = direction * spike_magnitude
        for frame_index in range(start, min(start + duration, len(clean))):
            for landmark_id in landmark_ids:
                corrupted[frame_index, landmark_id] += offset
                mask[frame_index, landmark_id] = True
    elif corruption_type == "occlusion":
        landmark_ids = FINGER_LANDMARKS[finger_group]
        source_index = max(start - 1, 0)
        for frame_index in range(start, min(start + duration, len(clean))):
            for landmark_id in landmark_ids:
                corrupted[frame_index, landmark_id] = clean[source_index, landmark_id]
                mask[frame_index, landmark_id] = True
    else:
        raise ValueError(f"Unsupported corruption type: {corruption_type}")

    for frame_index, landmark_id in np.argwhere(mask):
        manifest.append(
            {
                "frame_index": int(frame_index),
                "landmark_id": int(landmark_id),
                "finger_group": finger_group,
                "corruption_type": corruption_type,
                "severity": "fixed" if corruption_type == "spike" else "short",
                "duration": duration,
                "random_seed": seed,
            }
        )
    return corrupted, mask, manifest
