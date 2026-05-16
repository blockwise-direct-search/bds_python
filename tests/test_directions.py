import numpy as np
import unittest

from bds._directions import cycling, divide_direction_set
from bds._options import make_options


class DirectionTests(unittest.TestCase):
    def test_cycling_strategies_match_reference_examples(self):
        array = np.array([1, 2, 3, 4, 5])

        np.testing.assert_array_equal(cycling(array, 2, 0), [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(cycling(array, 2, 1), [3, 1, 2, 4, 5])
        np.testing.assert_array_equal(cycling(array, 2, 2), [3, 4, 5, 1, 2])
        np.testing.assert_array_equal(cycling(array, 2, 3), [4, 5, 1, 2, 3])

    def test_divide_direction_set_default_groups_evenly(self):
        opts = make_options({"num_blocks": 3}, 11, np.ones(11))

        groups = divide_direction_set(11, 3, opts)

        np.testing.assert_array_equal(groups[0] + 1, [1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(groups[1] + 1, [9, 10, 11, 12, 13, 14, 15, 16])
        np.testing.assert_array_equal(groups[2] + 1, [17, 18, 19, 20, 21, 22])

    def test_divide_direction_set_respects_user_dimension_groups(self):
        opts = make_options(
            {
                "num_blocks": 3,
                "grouped_direction_indices": [[1, 3, 5, 7], [2, 4, 8], [6, 9, 10, 11]],
            },
            11,
            np.ones(11),
        )

        groups = divide_direction_set(11, 3, opts)

        np.testing.assert_array_equal(groups[0] + 1, [1, 2, 5, 6, 9, 10, 13, 14])
        np.testing.assert_array_equal(groups[1] + 1, [3, 4, 7, 8, 15, 16])
        np.testing.assert_array_equal(groups[2] + 1, [11, 12, 17, 18, 19, 20, 21, 22])


if __name__ == "__main__":
    unittest.main()
