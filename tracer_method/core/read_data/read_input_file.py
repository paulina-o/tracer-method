from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from tracer_method.core.exceptions import FileException


def read_tritium_file(input_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read file with input data.

    :param input_file: file with input data
    :return: arrays with dates(year-month), tritium concentration and precipitation
    """
    if input_file.suffix == '.csv':
        data = pd.read_csv(input_file, usecols=[0, 2], header=0, names=['Date', 'Concentration', 'Precipitation'])
    elif input_file.suffix in ['.xlsx', '.xls']:
        data = pd.read_excel(input_file, usecols='A,B,C', header=0, names=['Date', 'Concentration', 'Precipitation'])
    else:
        raise FileException(f'Not supported extension: {input_file.suffix}')

    try:
        dates = pd.to_datetime(data['Date']).to_numpy(dtype='datetime64[M]')
        h3_concentration = data['Concentration'].to_numpy(dtype='float')
        precipitation = data['Precipitation'].to_numpy(dtype='float')
    except ValueError:
        raise FileException('Invalid data')

    except KeyError:
        raise FileException('Missing data')

    if data[data.columns[0:3]].isnull().sum().sum():
        raise FileException('Invalid data')

    if not np.all(np.diff(dates) == np.timedelta64('1', 'M')) or dates.size % 12:
        raise FileException('Missing data for every month. Make sure that your file has header.')

    return dates, h3_concentration, precipitation
