"""This file contains a class that represents a photovoltaic system with its properties."""

import preprocessing

class PhotoVoltaicSystem:
    """Class that models a photovoltaic system."""

    def __init__(self, area_in_square_meters=12):
        self.AREA_IN_SQUARE_METERS = area_in_square_meters        
        self.EFFICIENCY = 0.1

    def compute_electricity_generation_in_W(self, global_radiation_in_W_per_square_meter):
        """Computes the electricty generation of the solar array based on 
           the global solar radiation.

        Keyword arguments:
        global_radiation_in_W_per_square_meter -- the amount of global solar radiation in Watts / m^2
        """
        total_radiation_in_W = global_radiation_in_W_per_square_meter * self.AREA_IN_SQUARE_METERS
        generation_in_W = self.EFFICIENCY * total_radiation_in_W

        return generation_in_W