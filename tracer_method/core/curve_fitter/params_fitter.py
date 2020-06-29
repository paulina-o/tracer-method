from copy import deepcopy
from typing import Tuple

import numpy as np
from scipy import integrate
from scipy.optimize import minimize, OptimizeResult

from tracer_method.core.config.config_model import ConfigModel
from tracer_method.core.curve_fitter.response_functions import dispersion, exponential, exponential_piston_flow
from tracer_method.core.fitting_result import FittingResult


class ParamsFitter:
    """ Get the output which provides the best fit to observations data points. It depends on input, selected model
    (EM, EPM and DM), its parameters range and beta (if provided). """

    def __init__(self, input: np.ndarray, obs: np.ndarray, start_year: int, cfg: ConfigModel, decay: float):
        self.input = deepcopy(input)
        self.cfg = cfg
        self.decay = decay
        self.fit_data = FittingResult(cfg.type, obs, self.cfg.beta)
        self.start_year = start_year
        self.response_functions = {
            'DM': dispersion,
            'EM': exponential,
            'EPM': exponential_piston_flow,
        }

        if self.cfg.beta:
            self.input[1] *= (1 - self.cfg.beta)

    def __calculate_mse(self, params: np.ndarray) -> float:
        """
        Get calculated MSE (mean-square error) for predictions and observations.

        :param params: model's parameters used in order to calculate response function - g(t)
        :return: value of calculated MSE
        """
        interpolated_y_predictions = self.__get_interpolated_y_predictions(params)

        return ((interpolated_y_predictions - self.fit_data.observations[1]) ** 2).mean()

    def __calculate_me(self, params: np.ndarray) -> float:
        """
        Get calculated Model Efficiency (ME)

        :param params: model's parameters used in order to calculate response function - g(t)
        :return: value of calculated ME
        """
        interpolated_y_predictions = self.__get_interpolated_y_predictions(params)

        a = np.sum((interpolated_y_predictions - self.fit_data.observations[1]) ** 2)
        b = np.sum((interpolated_y_predictions - self.fit_data.observations[1].mean()) ** 2)

        return 1 - (a / b)

    def __get_interpolated_y_predictions(self, params: np.array):
        """
        Observations are usually only a few points so predictions points need to be interpolated (interpolation
        ia a method of constructing new data points within the range of a discrete set of known data points).

        :param params: model's parameters used in order to calculate response function - g(t)
        :return: interpolated y predictions
        """
        x_predictions, y_predictions = self._get_predictions(params)
        return np.interp(self.fit_data.observations[0], x_predictions, y_predictions)

    def _get_predictions(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions points which can be after used in order to calculate MSE and minimize it to get the best fit
        to observations.

        :param params: model's parameters
        :return: new x range points based on lengths of input and response function g(t) and calculated predictions
        points based on convolution of input and g(t)
        """
        response_function = self.response_functions[self.cfg.type]
        t = np.arange(0.001, 2, 1)
        g_t = response_function(t, params)

        while 1 - integrate.quad(response_function, min(t), max(t), args=params)[0] >= 0.01:
            t = np.arange(0.001, (len(t) + params[0]) + 1, 1)
            g_t = response_function(t, params)

        # print(params)

        x = np.round(np.arange(min(t) + min(self.input[0]), max(t) + max(self.input[0]) + 1, 1), 2) + self.start_year
        y = self.__calculate_convolution(self.input[1], g_t, t)
        self.time = t

        self.fit_data.set_output(np.round(x, 4), np.round(y, 4))
        self.fit_data.set_response_function(np.round(self.time, 4), np.round(g_t, 4))

        return x, y

    def __calculate_convolution(self, vector_a: np.ndarray, vector_b: np.ndarray, t) -> np.ndarray:
        """
        Calculate convolution of two vectors (full mode), include radioactive decay constant if tracer
        is a radionuclide.

        :param vector_a: first vector (input)
        :param vector_b: second vector (response function)
        :return: calculated convolution of two vectors
        """
        if self.decay is not None:
            return np.convolve(vector_a, vector_b * np.exp(-t * self.decay), mode="full")

        return np.convolve(vector_a, vector_b, mode="full")

    def get_data_solution(self, solution: OptimizeResult, params_accuracy=None):
        """
        Get all solution information - output data, response function data and model efficiency based on chosen
        parameters.

        :param solution: solution of the parameters minimization
        :param params_accuracy: model's parameters accuracy
        """
        params = np.round(solution.x, 2)
        self.fit_data.params = params

        self.fit_data.set_mse(round(solution.fun, 3))
        self.fit_data.set_model_efficiency(round(self.__calculate_me(params), 3))

        if params_accuracy is not None:
            self.fit_data.confidence_level = params_accuracy[0]
            self.fit_data.confidence_interval = params_accuracy[1]

        return self.fit_data

    def run_algorithm(self):
        """
        Run whole algorithm for finding the best model parameters.

        :return: solution of the minimization
        """
        solution = minimize(self.__calculate_mse, self.cfg.initial_values, method='TNC', options={'maxiter': 200},
                            bounds=self.cfg.params_range)

        return solution
