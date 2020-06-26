from typing import List, Union
from typing import Tuple

import numpy as np
from tracer_method.core.read_data.read_input_file import read_tritium_file

from tracer_method.core.read_data.read_observations_file import read_observations

import tracer_method.core.constans as const
from tracer_method.core.config.config_model import ConfigModel
from tracer_method.core.curve_fitter.params_fitter import ParamsFitter
from tracer_method.core.curve_fitter.pfm_params_fitter import PFMParamsFitter
from tracer_method.core.run import run
from tracer_method.core.tritium.tritium_input_preparer import TritiumInputPreparer
from pathlib import Path

FITTING_METHODS = {
    'DM': ParamsFitter,
    'EM': ParamsFitter,
    'EPM': ParamsFitter,
    'PFM': PFMParamsFitter,
}


def tritium_method(input: Tuple[np.ndarray, np.ndarray, np.ndarray], obs: np.ndarray, alpha: float,
                   model_configs: List[List[Union[str, float]]], calculate_params_accuracy=False):
    """
    Calculate output concentration based on provided data, calculations are obtained for each model configuration.

    :param input: input data with dates and monthly h3 concentration and precipitation (data must be provided
    for whole year)
    :param obs: observations data with date and h3 concentration
    :param alpha: infiltration rate (from 0.01 to 1)
    :param model_configs: models configuration for which the output concentration should be calculated
    :param calculate_params_accuracy: True if accuracy of params should be included, False otherwise
    :return: the list with calculated output concentration with the best fit for each provided model
    """
    dates, concentration, precipitation = input
    start_year = int(str(min(dates.astype('datetime64[Y]'))))
    input_data = TritiumInputPreparer(dates, concentration, precipitation, alpha).calculate_input()

    decay = np.log(2) / const.DECAY_CONSTANTS['tritium']

    output_data = []
    for model_cfg in model_configs:
        config = ConfigModel(model_cfg)

        try:
            fitting_method = FITTING_METHODS[config.type]
        except KeyError:
            raise Exception('Model type not found: PFM, EM, EPM or DM')

        fitting_result = run(input_data, obs, start_year, config, decay, fitting_method, calculate_params_accuracy)

        output_data.append(fitting_result)

    return output_data
