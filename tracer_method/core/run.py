import multiprocessing
from functools import partial
from typing import Callable

import numpy as np

from tracer_method.core.config.config_model import ConfigModel
from tracer_method.core.fitting_result import FittingResult


def get_params_accuracy(input, obs, start_year, config, decay, fitting_method) -> np.ndarray:
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

    return np.std(results, axis=0)


def run_method(obs, input, start_year, config, decay, fitting_method) -> np.ndarray:
    """ Run fitting method and return the best model parameters. """
    solution = fitting_method(input, obs, start_year, config, decay).run_algorithm()

    return solution.x


def run(input: np.ndarray, obs: np.ndarray, start_year: int, config: ConfigModel, decay: float,
        fitting_method: Callable) -> FittingResult:
    """
    Run whole simulation and get all fitting data, calculate parameters accuracy

    :return:
    :param input: input data
    :param obs: observations
    :param start_year: start year for which calculations begin
    :param config: configuration
    :param decay: decay constant
    :param fitting_method: fitting method (different type for PFM)
    :return: Fit Data which includes observations, model type, calculated parameters, beta value and final output

    """
    base_fitter = fitting_method(input, obs, start_year, config, decay)

    solution = base_fitter.run_algorithm()
    params_accuracy = get_params_accuracy(input, obs, start_year, config, decay, fitting_method)

    results = base_fitter.get_data_solution(solution, params_accuracy)

    return results
