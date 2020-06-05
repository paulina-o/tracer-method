import numpy as np


class FittingResult:
    """ Class which holds output data(observations, model type, model params, beta param)  regarding obtaining the
    best fit for specified observations. """

    def __init__(self, model_type: str, observations: np.ndarray, beta: float=0):
        self.model_type = model_type
        self.observations = observations
        self.output: np.ndarray
        self.params: np.ndarray
        self.response_function: np.array
        self.beta = beta
        self.mse = 0
        self.model_efficiency = 0
        self.params_accuracy = []

    def set_params(self, params: np.ndarray):
        self.params = params

    def set_response_function(self, time: np.ndarray, g_t: np.ndarray):
        self.response_function = np.array([time, g_t])

    def set_output(self, dates: np.ndarray, concentration: np.ndarray):
        self.output = np.array([dates, concentration])

    def set_mse(self, mse: float):
        self.mse = mse

    def set_model_efficiency(self, model_efficiency: float):
        self.model_efficiency = model_efficiency
