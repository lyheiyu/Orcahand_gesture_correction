import numpy as np

from evaluate_sequence_encodings import encode_sequence


def test_sequence_encoding_dimensions() -> None:
    sequence = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
    assert encode_sequence(sequence, "global4").shape == (12,)
    assert encode_sequence(sequence, "global5").shape == (15,)
    assert encode_sequence(sequence, "pyramid").shape == (45,)
    assert encode_sequence(sequence, "resample16").shape == (48,)


def test_order_aware_encodings_change_when_time_is_reversed() -> None:
    sequence = np.asarray([[0.0], [1.0], [4.0], [9.0]], dtype=np.float32)
    reversed_sequence = sequence[::-1]
    assert not np.allclose(
        encode_sequence(sequence, "pyramid"),
        encode_sequence(reversed_sequence, "pyramid"),
    )
    assert not np.allclose(
        encode_sequence(sequence, "resample16"),
        encode_sequence(reversed_sequence, "resample16"),
    )
