import unittest

import numpy as np

from tracer_method.core.tritium.tritium_method import tritium_method


class TestClass(unittest.TestCase):
    def setUp(self):
        observations_data = np.array([np.arange(1, 10), np.arange(1, 10)])
        input_data = (np.arange(1, 1.1), np.arange(1, 10), np.arange(1, 10))

        model_a = [['DM', ((20.0, 90.0), (0.01, 0.1)), 0.4]]
        alpha_a = 0.5
        self.results = tritium_method(input_data, observations_data, alpha_a, model_a)

    def test_tritium_method(self):
        self.assertEqual(1, len(self.results), 'Checking if one result is returned')

        self.assertListEqual([55, 0.06], [i for i in self.results[0].params],
                             'Checking if parameters is calculated in right way')

        self.assertListEqual([0, 6.938893903907228e-18], [i for i in self.results[0].params_accuracy],
                             'Checking if params accuracy is calculated in right way')

        self.assertTrue(self.results[0].output, 'Checking if output is calculated')

        self.assertTrue(self.results[0].response_function, 'Checking if response function is calculated')


if __name__ == '__main__':
    unittest.main()
