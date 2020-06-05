from typing import Tuple

import numpy as np

from tracer_method.core.config.config_model import ConfigModel
from tracer_method.core.curve_fitter.params_fitter import ParamsFitter


class PFMParamsFitter(ParamsFitter):
    """ Get the output data which gets the best fit to observations data points. It depends on PFM model params
    and beta (if provided). """

    def __init__(self, input: np.ndarray, obs: np.ndarray, start_year: int, cfg: ConfigModel, decay: float):
        super().__init__(input, obs, start_year, cfg, decay)

    def _get_predictions(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions points which can be after used in order to calculate RMSE and minimize it to get the best fit
        to observations.

        :param params: model's parameters
        :return: predictions (if tracer is a radionuclide radioactive decay constant is included)
        """
        t_t, = params

        x = self.input[0] + t_t + self.start_year
        y = self.input[1] * np.exp(-t_t * self.decay) if self.decay is not None else self.input[1]

        self.fit_data.set_output(x, y)

        return x, y
