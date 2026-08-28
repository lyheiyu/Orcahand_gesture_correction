import numpy as np

from joint_angle_recovery import (
    ConfidenceParameters,
    affected_joint_mask,
    recover_confidence_weighted,
)


def _parameters() -> ConfidenceParameters:
    return ConfidenceParameters(
        jump_scale=np.full(11, 10.0),
        prediction_scale=np.full(11, 10.0),
        geometry_scale=np.full(11, 0.1),
        lower_bound=np.zeros(11),
        upper_bound=np.full(11, 180.0),
        strength=1.0,
    )


def test_landmark_mask_maps_only_to_incident_joints() -> None:
    mask = np.zeros((2, 21), dtype=bool)
    mask[1, 7] = True
    joint_mask = affected_joint_mask(mask)
    assert joint_mask[1, 4]
    assert not joint_mask[1, 3]
    assert int(joint_mask.sum()) == 1


def test_recovery_is_causal_and_confidence_is_bounded() -> None:
    points = np.zeros((6, 21, 3), dtype=float)
    for landmark in range(21):
        points[:, landmark, 0] = landmark + 1.0
        points[:, landmark, 1] = np.arange(6) * 0.01
    observed = np.full((6, 11), 90.0)
    first, confidence = recover_confidence_weighted(observed, points, _parameters())
    changed = observed.copy()
    changed[4:] += 50.0
    second, _ = recover_confidence_weighted(changed, points, _parameters())
    np.testing.assert_allclose(first[:4], second[:4])
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))
