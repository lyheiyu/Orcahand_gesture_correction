import numpy as np

from generate_occlusion_dataset import _apply_corruption


def _clean_sequence(length: int = 12) -> np.ndarray:
    points = np.zeros((length, 21, 3), dtype=np.float64)
    for frame in range(length):
        points[frame, :, 0] = np.linspace(0.0, 1.0, 21)
        points[frame, :, 1] = frame * 0.01
        points[frame, :, 2] = np.linspace(0.0, 0.5, 21)
    return points


def test_corruption_is_deterministic_and_local_to_selected_window() -> None:
    clean = _clean_sequence()
    landmarks = (5, 6, 7, 8)
    first = _apply_corruption(
        clean,
        start=3,
        end=8,
        landmark_indices=landmarks,
        mode="drift",
        noise_std=0.05,
        rng=np.random.default_rng(42),
    )
    second = _apply_corruption(
        clean,
        start=3,
        end=8,
        landmark_indices=landmarks,
        mode="drift",
        noise_std=0.05,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[:3], clean[:3])
    np.testing.assert_allclose(first[8:], clean[8:])
    unaffected = sorted(set(range(21)) - set(landmarks))
    np.testing.assert_allclose(first[3:8, unaffected], clean[3:8, unaffected])
    assert np.any(np.abs(first[3:8, list(landmarks)] - clean[3:8, list(landmarks)]) > 1e-9)
    assert np.isfinite(first).all()


def test_freeze_uses_last_visible_frame_before_window() -> None:
    clean = _clean_sequence()
    landmarks = (17, 18, 19, 20)
    corrupted = _apply_corruption(
        clean,
        start=4,
        end=7,
        landmark_indices=landmarks,
        mode="freeze",
        noise_std=0.0,
        rng=np.random.default_rng(1),
    )

    expected = np.repeat(clean[3, list(landmarks)][None, :, :], 3, axis=0)
    np.testing.assert_allclose(corrupted[4:7, list(landmarks)], expected)

