import numpy as np


def exponential(t: np.ndarray, params: np.ndarray):
    """
    Calculate response function for EM

    :param t: time range
    :param params: transit time
    :return: calculated response function g(t)
    """
    t_t, = params

    return 1/t_t * np.exp(-t / t_t)


def exponential_piston_flow(t: np.ndarray, params: np.ndarray):
    """
    Calculate response function for EPM

    :param t: time
    :param params: transit time and η (he ratio of the total volume to the volume with the exponential distribution
    of transit times)
    :return: calculated response function g(t)
    """
    t_t, n = params

    return np.where(t >= t_t * (1 - 1 / n), n / t_t * np.exp(-n * t / t_t + n - 1), 0)


def dispersion(t: np.ndarray, params: np.ndarray):
    """
    Calculate response function for DM

    :param t: time range
    :param params: transit time and dispersion parameter (Pd)
    :return: calculated response function g(t)
    """
    t_t, p_d = params

    return ((4 * np.pi * p_d * t / t_t) ** -0.5) * (1 / t) * np.exp(-((1 - (t / t_t)) ** 2) / (4 * p_d * t / t_t))
