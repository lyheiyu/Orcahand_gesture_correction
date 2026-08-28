import numpy as np

from generate_joint_angle_baseline import JOINT_DEFINITIONS, joint_angle_vector


def test_joint_angle_vector_returns_expected_straight_angles() -> None:
    points = np.zeros((21, 3), dtype=np.float64)
    chains = (
        ((1, 2, 3, 4), np.array((-0.7, 0.7, 0.0))),
        ((5, 6, 7, 8), np.array((-0.3, 1.0, 0.0))),
        ((9, 10, 11, 12), np.array((0.0, 1.0, 0.0))),
        ((13, 14, 15, 16), np.array((0.3, 1.0, 0.0))),
        ((17, 18, 19, 20), np.array((0.6, 0.8, 0.0))),
    )
    for chain, direction in chains:
        direction = direction / np.linalg.norm(direction)
        for distance, landmark in enumerate(chain, start=1):
            points[landmark] = distance * direction
    angles, valid = joint_angle_vector(points)
    assert valid.all()
    np.testing.assert_allclose(angles, 180.0, atol=1e-5)


def test_joint_angle_vector_handles_zero_length_vectors() -> None:
    points = np.zeros((21, 3), dtype=np.float64)
    angles, valid = joint_angle_vector(points)
    assert not valid.any()
    np.testing.assert_allclose(angles, 0.0)
    assert np.isfinite(angles).all()
