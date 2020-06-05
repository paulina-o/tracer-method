import unittest

import numpy as np

from tracer_method.core.config.config_model import ConfigModel
from tracer_method.core.curve_fitter.params_fitter import ParamsFitter


class TestClass(unittest.TestCase):
    def setUp(self):
        input_data = np.array([np.arange(0.01, 12, dtype='float64'),
                               np.arange(0.01 + 10, 12 + 10, dtype='float64')])
        obs = np.array([np.arange(0, 4), np.arange(11, 15)])
        cfg = ConfigModel(['DM', ((20.0, 90.0), (0.01, 1)), 0.4])

        self.fitter = ParamsFitter(input_data, obs, 1950, cfg, 0.06)

    def test_run_fitting(self):
        solution = self.fitter.run_algorithm()
        fitting_results = self.fitter.get_data_solution(solution, '')

        self.assertEqual(-0.008, fitting_results.model_efficiency,
                         'Checking if model Model Efficacy is calculated in the right way')

        self.assertEqual(157.5, fitting_results.mse, 'Checking if Mean Squared Error is equal to the right value')

        self.assertListEqual([55, 0.5], [i for i in fitting_results.params], 'Checking model parameters')

        self.assertListEqual([], [i for i in fitting_results.params_accuracy], 'Checking model parameters accuracy')

        self.assertListEqual([0.001, 1.001, 2.001, 3.001, 4.001],
                             [i for i in fitting_results.response_function[0]][:5],
                             'Checking arguments of response function')

        self.assertListEqual([0.0, 0.0, 0.0, 0.0002, 0.001],
                             [i for i in fitting_results.response_function[1]][:5],
                             'Checking values of response function')


if __name__ == '__main__':
    unittest.main()
