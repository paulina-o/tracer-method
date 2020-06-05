import unittest

import numpy as np

from tracer_method.core.config.config_model import ConfigModel
from tracer_method.core.curve_fitter.pfm_params_fitter import PFMParamsFitter


class TestClass(unittest.TestCase):
    def setUp(self):
        input_data = np.array([np.arange(0.01, 12, dtype='float64'),
                               np.arange(0.01 + 10, 12 + 10, dtype='float64')])
        obs = np.array([np.arange(0, 4), np.arange(11, 15)])
        cfg = ConfigModel(['DM', ((20.0, 90.0), (0.01, 1)), 0.4])

        self.fitter = PFMParamsFitter(input_data, obs, 1950, cfg, 0.06)

    def test_get_predictions(self):
        predictions = self.fitter._get_predictions(np.array([5, ]))

        self.assertListEqual([1955.01, 1956.01, 1957.01, 1958.01], [i for i in predictions[0][:4]],
                             'Checking arguments of Piston Flow response function')

        self.assertListEqual([4.449354233414397, 4.8938451658234285, 5.338336098232459, 5.7828270306414895],
                             [i for i in predictions[1][:4]], 'Checking values of Piston Flow response function')


if __name__ == '__main__':
    unittest.main()
