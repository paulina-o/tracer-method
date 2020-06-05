import unittest

from tracer_method.core.fitting_result import FittingResult
import numpy as np


class TestClass(unittest.TestCase):
    def setUp(self):
        self.fitting = FittingResult('DM', np.arange(1, 4), 0.4)

    def test_initial_values(self):
        self.assertEqual('DM', self.fitting.model_type, 'Checking if model type is set')

        self.assertListEqual([1, 2, 3], [i for i in self.fitting.observations], 'Checking if observations are set')

        self.assertEqual(0.4, self.fitting.beta, 'Checking if beta parameter is set')

    def test_set_params(self):
        self.fitting.set_params(np.array([0.1, 0.9]))
        self.assertListEqual([0.1, 0.9], [i for i in self.fitting.params], 'Checking if params are set')

    def test_set_response_function(self):
        self.fitting.set_response_function(np.array([1, 9]), np.array([1, 19]))
        self.assertListEqual([1, 9, 1, 19], [j for i in self.fitting.response_function for j in i],
                             'Checking if response function is set')

    def test_set_output(self):
        self.fitting.set_output(np.array([1, 9]), np.array([1, 19]))
        self.assertListEqual([1, 9, 1, 19], [j for i in self.fitting.output for j in i], 'Checking if output is set')

    def test_set_mse(self):
        self.fitting.set_mse(92)
        self.assertEqual(92, self.fitting.mse, 'Checking if mse are set')

    def test_model_efficiency(self):
        self.fitting.set_model_efficiency(111)
        self.assertEqual(111, self.fitting.model_efficiency, 'Checking if model efficiency are set')


if __name__ == '__main__':
    unittest.main()
