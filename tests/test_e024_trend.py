import numpy as np

from src.scripts.run_e024_trend import sparse_shift, strongest_changes


def test_shift_preserves_zeros_and_input_and_clips_negative_values():
    original = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.float32)
    unchanged = original.copy()
    result = sparse_shift(original, np.array([1, -2, .5], dtype=np.float32))
    np.testing.assert_array_equal(result, [[0, 0, 2.5], [4, 0, 1.5]])
    np.testing.assert_array_equal(original, unchanged)
    assert result.dtype == np.float32 and np.isfinite(result).all()


def test_top_changes_are_signed_stable_and_do_not_modify_input():
    delta = np.array([1., -3., 3., .1], dtype=np.float32)
    np.testing.assert_array_equal(strongest_changes(delta, 1), [0, -3, 0, 0])
    np.testing.assert_array_equal(strongest_changes(delta, 2), [0, -3, 3, 0])
    np.testing.assert_array_equal(strongest_changes(delta, 0), delta)
    np.testing.assert_array_equal(delta, [1, -3, 3, np.float32(.1)])
    with np.testing.assert_raises(ValueError):
        strongest_changes(delta, -1)
