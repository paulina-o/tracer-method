import multiprocessing
from functools import partial
from typing import Callable

import numpy as np

from tracer_method.core.config.config_model import ConfigModel
from tracer_method.core.fitting_result import FittingResult


def get_params_accuracy(params, input, obs, start_year, config, decay, fitting_method):
    """
    Calculate measurement uncertainties of calculated parameters as standard deviations based on multiple
    observations data set which are randomly obtained within one sigma.

    :return: params accuracy (standard deviations)
    """
    std = np.std(obs[1])
    obs_sets = [np.array([obs[0], np.array([np.random.uniform(i - std, i + std) for i in obs[1]])])
                for i in range(100)]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(partial(run_method, input=input, start_year=start_year, config=config,
                                   decay=decay, fitting_method=fitting_method), obs_sets)

    return get_params_interval_confidence(params, np.round(results, 4))


def get_params_interval_confidence(params: np.ndarray, results: np.ndarray):
    """
    Get params confidence level and confidence interval.

    :param params: obtained parameters
    :param results: calculated parameters based on multiple observations sets
    :return: calculated confidence level and confidence interval
    """
    std = np.std(results, axis=0)
    results_params = results.transpose()
    ranges = [(i * 0.9, i * 1.1) for i in params]

    is_any_value = [len(value[np.logical_or(value < condition[0], value > condition[1])])
                    for value, condition in zip(results_params, ranges)]

    confidence_level = [1 - (is_any_value[i] / len(results_params[i])) if is_any_value[i] else 0.95
                        for i in range(0, len(params))]

    confidence_interval = [ranges[i] if is_any_value[i] else (params[i] - std[i] * 1.96, params[i] + std[i] * 1.96)
                           for i in range(0, len(params))]

    return confidence_level, confidence_interval


def run_method(obs, input, start_year, config, decay, fitting_method) -> np.ndarray:
    """ Run fitting method and return the best model parameters. """
    solution = fitting_method(input, obs, start_year, config, decay).run_algorithm()

    return solution.x


def run(input: np.ndarray, obs: np.ndarray, start_year: int, config: ConfigModel, decay: float,
        fitting_method: Callable, calculate_params_accuracy: bool) -> FittingResult:
    """
    Run whole simulation and get all fitting data, calculate parameters accuracy

    :return:
    :param input: input data
    :param obs: observations
    :param start_year: start year for which calculations begin
    :param config: configuration
    :param decay: decay constant
    :param fitting_method: fitting method (different type for PFM)
    :param calculate_params_accuracy: True if accuracy of params should be included, False otherwise
    :return: Fit Data which includes observations, model type, calculated parameters, beta value and final output

    """
    base_fitter = fitting_method(input, obs, start_year, config, decay)

    solution = base_fitter.run_algorithm()
    params = solution.x

    if calculate_params_accuracy:
        if params.size == 2:
            params_range = tuple([(i - 0.1 * i, i + 0.1 * i) for i in params])
        else:
            params_range = ((params[0] - 0.1 * params[0], params[0] + 0.1 * params[0]), )

        new_config = ConfigModel([config.type, params_range, config.beta])

        params_accuracy = get_params_accuracy(params, input, obs, start_year, new_config, decay, fitting_method)
        return base_fitter.get_data_solution(solution, params_accuracy)

    return base_fitter.get_data_solution(solution)
