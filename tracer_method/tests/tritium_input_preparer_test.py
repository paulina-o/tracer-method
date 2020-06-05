import unittest

import numpy as np

from tracer_method.core.tritium.tritium_input_preparer import TritiumInputPreparer


class TestClass(unittest.TestCase):
    def setUp(self):
        alpha = 0.5
        time = np.arange(1, 2)
        concentration = np.arange(10, 22)
        precipitation = np.arange(1, 13)
        self.preparer = TritiumInputPreparer(time, concentration, precipitation, alpha)

    def test_calculate_input(self):
        input = self.preparer.calculate_input()
        self.assertListEqual([0.0, 17.794871794871796], [j for i in input for j in i],
                             'Check if input is calculated correct')


if __name__ == '__main__':
    unittest.main()
