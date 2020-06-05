from pathlib import Path

import numpy as np
import pandas as pd

from tracer_method.core.exceptions import FileException


def read_observations(obs_file: Path) -> np.ndarray:
    """
    Read csv file with observations data.

    :param obs_file: csv file with observations
    :return: array with observations data (dates and h3 concentration)
    """
    if obs_file.suffix == '.csv':
        data = pd.read_csv(obs_file, usecols=[0, 1], header=0, names=['Date', 'Concentration'])
    elif obs_file.suffix in ['.xlsx', '.xls']:
        data = pd.read_excel(obs_file, usecols='A,B', header=0, names=['Date', 'Concentration'])
    else:
        raise FileException(f'Not supported extension: {obs_file.suffix}')

    try:
        dates = pd.to_datetime(data['Date']).to_numpy(dtype='datetime64[D]')
        h3_concentration = data['Concentration'].to_numpy(dtype='float')
    except ValueError:
        raise FileException('Invalid data')
    except KeyError:
        raise FileException('Missing data')

    if data.isnull().sum().sum():
        raise FileException('Missing data')

    return np.array([convert_date_to_year(dates), h3_concentration])


def convert_date_to_year(dates: np.ndarray):
    """
    Convert dates eg. ['1974-05-30' '1975-10-23' '1976-04-22'] to year and part of year eg. [1974.41, 1975.81, 1976.31].

    :param dates: array with dates in format '%Y-%m-%d'
    :return: array with coverted dates into years and part of it eg. 1974-05-30 -> 1974 + 151/365 (30th May is 151st
    day pf year) = 1974.41
    """
    years = dates.astype(dtype="datetime64[Y]")

    days = dates - np.array([np.datetime64(f'{year}-01-01') for year in years])
    days = days / np.timedelta64(1, 'D') + 1

    year_part = days / 365

    return np.round(years.astype(str).astype(float) + year_part, 2)

