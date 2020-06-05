import unittest

from tracer_method.core.config.config_model import ConfigModel


class TestClass(unittest.TestCase):
    def setUp(self):
        self.cfg = ConfigModel(['EM', ((20.0, 90.0), )])

    def test_run_fitting(self):
        self.assertEqual(110, self.cfg.initial_values[0], 'Checking initial values of parameters')

        self.assertEqual('EM', self.cfg.type, 'Checking model type')

        self.assertEqual(0, self.cfg.beta, 'Checking if beta equals 0')

        self.assertTupleEqual((20, 90), self.cfg.params_range[0], 'Checking params range')


if __name__ == '__main__':
    unittest.main()
