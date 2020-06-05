import unittest

import numpy as np

from tracer_method.core.curve_fitter.response_functions import dispersion, exponential_piston_flow, exponential


class TestClass(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(1, 5)

    def test_exponential_model(self):
        parameters = np.array([5, ])
        response_function = exponential(self.time, parameters)
        self.assertListEqual([0.1637461506155964, 0.13406400920712788, 0.1097623272188053, 0.08986579282344431],
                             [i for i in response_function],
                             'Checking values of exponential response function')

    def test_dispersion_model(self):
        parameters = np.array([30, 0.05])
        response_function = dispersion(self.time, parameters)
        self.assertListEqual([9.244018651465891e-61, 1.0328078939613802e-28, 3.426591190556329e-18,
                              5.055368281422804e-13], [i for i in response_function],
                             'Checking values of dispersion response function')

    def test_exponential_piston_flow_model(self):
        parameters = np.array([30, 1.05])
        response_function = exponential_piston_flow(self.time, parameters)
        self.assertListEqual([0.0, 0.03430695356573644, 0.03312698017837194, 0.03198759148449299],
                             [i for i in response_function],
                             'Checking values of exponential piston flow response function')


if __name__ == '__main__':
    unittest.main()
