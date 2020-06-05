import unittest

import numpy as np

from tracer_method.core.config.config_model import ConfigModel
from tracer_method.core.curve_fitter.params_fitter import ParamsFitter
from tracer_method.core.run import run, get_params_accuracy, run_method


class TestClass(unittest.TestCase):
    def setUp(self):
        self.observations_data = np.array([np.arange(1, 4), np.arange(1, 4)])
        self.input_data = np.array([np.arange(1, 13), np.arange(1, 13), np.arange(1, 13)])

        self.decay = 0.12
        self.config = ConfigModel(['DM', ((20.0, 90.0), (0.01, 0.1))])
        self.fitter = ParamsFitter

    def test_run(self):
        results = run(self.input_data, self.observations_data, 1, self.config, self.decay, self.fitter)

        self.assertListEqual([55, 0.06], [i for i in results.params],
                             'Checking if parameters is calculated in right way')

        self.assertListEqual([0, 6.938893903907228e-18], [i for i in results.params_accuracy],
                             'Checking if params accuracy is calculated in right way')

    def test_get_params_accuracy(self):
        params_accuracy = get_params_accuracy(self.input_data, self.observations_data, 1, self.config,
                                              self.decay, self.fitter)

        self.assertListEqual([0.0, 6.938893903907228e-18], [i for i in params_accuracy],
                             'Checking if params accuracy is calculated')

    def test_run_method(self):
        solution = run_method(self.input_data, self.observations_data, 1, self.config, self.decay, self.fitter)

        self.assertListEqual([55.0, 0.055], [i for i in solution],
                             'Checking if parameters is calculated in right way')


if __name__ == '__main__':
    unittest.main()
