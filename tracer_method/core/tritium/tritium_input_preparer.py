import numpy as np


class TritiumInputPreparer:
    """ Read input file with tritium concentration and precipitation and calculate the input for each calendar year. """

    def __init__(self, dates: np.ndarray, concentration: np.ndarray, precipitation: np.ndarray, alpha: float):
        self.alpha = alpha
        self.dates = dates
        self.concentration = concentration
        self.precipitation = precipitation
        self.dates_range = 0

        self.dates_range = int(max(self.dates.astype('datetime64[Y]')) - min(self.dates.astype('datetime64[Y]')) + 1)

    def __calculate_infiltration_rated(self) -> np.ndarray:
        """ Calculate infiltration rates for each year which was provided, summer moths (from april to september)
        has different infiltration rate that winter months (from october to march)

        :return: infiltration rates for each year
        """
        infiltration_rates = np.ones(12)
        infiltration_rates[3:9] = self.alpha  # summer months has smaller infiltration rates

        return np.tile(infiltration_rates, self.dates_range)

    def calculate_input(self) -> np.ndarray:
        """
        Calculate input for each year based on tritium concentration, precipitation and infiltration rates

        :return: input data for tritium method
        """
        infiltration_rates = self.__calculate_infiltration_rated()

        concentrations = self.precipitation * infiltration_rates * self.concentration
        precipitations = self.precipitation * infiltration_rates

        year_concentration = np.add.reduceat(concentrations, np.arange(0, len(concentrations), 12)) / \
            np.add.reduceat(precipitations, np.arange(0, len(precipitations), 12))

        return np.array([np.round(np.arange(0.00001, self.dates_range, 1), 2), year_concentration])
