from enum import Enum
from typing import List, Union

import numpy as np


class Model(Enum):
    """ Represents model configuration items. """
    TYPE = 0
    PARAMS_RANGE = 1
    BETA = 2


class ConfigModel:
    """ Holds model configuration with all the necessary data like model type, parameters range, beta value
    and start values for each model parameter. """

    def __init__(self, model_data: List[Union[str, float]]):
        self.type = model_data[Model.TYPE.value]
        self.params_range = model_data[Model.PARAMS_RANGE.value]
        self.beta = model_data[Model.BETA.value] if len(model_data) > Model.BETA.value else 0
        self.initial_values = self.__calculate_initial_values()

    def __calculate_initial_values(self) -> np.ndarray:
        """
        Get initial guess values which are calculated based on parameters ranges as the middle value of interval.

        :return: array of elements of size (n,), where ‘n’ is the number of model's parameters
        """
        if len(self.params_range) == 1:
            return np.array([np.mean(self.params_range[0][0] + self.params_range[0][1]), ])

        return np.array([(param[0] + param[1]) / 2 for param in self.params_range])
