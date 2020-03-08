"""This file contains a class that represents a battery with its properties."""

import tensorflow as tf
import numpy as np

class Battery:
    """Class that models a battery."""

    def __init__(self, capacity_in_kWh=8.0):
        self.stored_energy_in_kWh = None
        self.current_soc_in_percent = None # ranges from 0.0 to 100.0 (0-100%)
        self.CAPACITY_IN_kWh = capacity_in_kWh
        self.MAX_CHARGE_RATE_IN_W = 3000.0
        self.MAX_DISCHARGE_RATE_IN_W = 3000.0

    def update_soc_with_charge_rate_and_constraints(self, current_charge_rate_in_W, charging_time_in_mins):
        """Simulates charging the battery and updates the battery SoC and charge.

        Keyword arguments:
        current_charge_rate_in_W -- rate at which the battery is charged in W
        charging_time_in_mins -- how long the battery is charged in minutes
        """
        current_charge_rate_in_W = self.enforce_charge_rate_contraints(current_charge_rate_in_W, charging_time_in_mins)
        current_charge_rate_in_kW = current_charge_rate_in_W / 1000.0
        charging_time_in_hours = charging_time_in_mins / 60.0
        self.stored_energy_in_kWh += current_charge_rate_in_kW * charging_time_in_hours
        self.stored_energy_in_kWh = self.enforce_battery_energy_storage_constraints()
        self.current_soc_in_percent = (self.stored_energy_in_kWh / self.CAPACITY_IN_kWh) * 100.0

        return self.current_soc_in_percent

    def enforce_charge_rate_contraints(self, charge_rate_in_W, charging_time_in_mins):
        """Changes the value of the charging rate if necessary in order to comply with constraints.

        Keyword arguments:
        charge_rate_in_W -- the battery charging rate in Watts
        charging_time_in_mins -- how long the battery is charged with the given charge rate in minutes
        """
        charge_rate_in_W = self._ensure_compliance_with_battery_storage_constraints(charge_rate_in_W, charging_time_in_mins)
        charge_rate_in_W = self._ensure_value_stays_within_min_max_limit(charge_rate_in_W)

        return charge_rate_in_W

    def _ensure_compliance_with_battery_storage_constraints(self, charge_rate_in_W, charging_time_in_mins):
        """Corrects the charge rate value if it would result in a state of charge that is negative or greater than 100.

        Keyword arguments:
        charge_rate_in_W -- the charge rate chosen by the policy in Watts
        charging_time_in_mins -- how long the battery is charged with the given charge rate in minutes
        """
        charging_time_in_hours = charging_time_in_mins / 60.0
        resulting_soc_in_percent = self.get_soc_resulting_from_charging_in_percent(self.current_soc_in_percent, charge_rate_in_W, charging_time_in_mins)
        resulting_stored_energy_in_kWh = (resulting_soc_in_percent / 100.0) * self.CAPACITY_IN_kWh

        if isinstance(charge_rate_in_W, float) or isinstance(charge_rate_in_W, int):
            if resulting_stored_energy_in_kWh > self.CAPACITY_IN_kWh:
                max_additional_energy_to_store_in_Wh = (self.CAPACITY_IN_kWh - self.stored_energy_in_kWh) * 1000.0
                charge_rate_in_W = max_additional_energy_to_store_in_Wh / charging_time_in_hours
            elif resulting_stored_energy_in_kWh < 0.0:
                charge_rate_in_W = -(self.stored_energy_in_kWh * 1000.0 / charging_time_in_hours)
        elif isinstance(charge_rate_in_W, tf.Tensor):
            charge_rate_in_W = tf.cond(resulting_stored_energy_in_kWh > self.CAPACITY_IN_kWh,
                                       true_fn=lambda: tf.cast((self.CAPACITY_IN_kWh - self.stored_energy_in_kWh) * 1000.0 / charging_time_in_hours, tf.float32),
                                       false_fn=lambda: charge_rate_in_W)
            charge_rate_in_W = tf.cond(resulting_stored_energy_in_kWh < 0.0,
                                       true_fn=lambda: tf.cast(-(self.stored_energy_in_kWh * 1000.0 / charging_time_in_hours), tf.float32),
                                       false_fn=lambda: charge_rate_in_W)
        else:
            raise ValueError("Unexpected input type:" + str(type(charge_rate_in_W)))

        return charge_rate_in_W

    def _ensure_value_stays_within_min_max_limit(self, charge_rate_in_W):
        """Corrects the given charge rate if it exceeds the maximum allowed charge or discharge rate.

        Keyword arguments:
        charge_rate_in_W -- the charge rate chosen by the policy in Watts
        """
        if isinstance(charge_rate_in_W, float) or isinstance(charge_rate_in_W, int):
            if charge_rate_in_W > self.MAX_CHARGE_RATE_IN_W:
                charge_rate_in_W = self.MAX_CHARGE_RATE_IN_W
            elif charge_rate_in_W < -self.MAX_DISCHARGE_RATE_IN_W:
                charge_rate_in_W = -self.MAX_DISCHARGE_RATE_IN_W
        elif isinstance(charge_rate_in_W, tf.Tensor):
            charge_rate_in_W = tf.cond(charge_rate_in_W > self.MAX_CHARGE_RATE_IN_W,
                                       true_fn=lambda: self.MAX_CHARGE_RATE_IN_W,
                                       false_fn=lambda: charge_rate_in_W)
            charge_rate_in_W = tf.cond(charge_rate_in_W < -self.MAX_DISCHARGE_RATE_IN_W,
                                       true_fn=lambda: -self.MAX_DISCHARGE_RATE_IN_W,
                                       false_fn=lambda: charge_rate_in_W)
        else:
            raise ValueError("Unexpected input type:" + str(type(charge_rate_in_W)))

        return charge_rate_in_W

    def enforce_battery_energy_storage_constraints(self):
        """Changes the value of the electrical energy stored in the battery to comply with physical constraints."""
        if isinstance(self.stored_energy_in_kWh, float) or isinstance(self.stored_energy_in_kWh, int):
            if self.stored_energy_in_kWh > self.CAPACITY_IN_kWh:
                self.stored_energy_in_kWh = self.CAPACITY_IN_kWh
            elif self.stored_energy_in_kWh < 0.0:
                self.stored_energy_in_kWh = 0.0
        elif isinstance(self.stored_energy_in_kWh, tf.Tensor):
            self.stored_energy_in_kWh = tf.cond(self.stored_energy_in_kWh > self.CAPACITY_IN_kWh,
                                                true_fn=lambda: self.CAPACITY_IN_kWh,
                                                false_fn=lambda: self.stored_energy_in_kWh)
            self.stored_energy_in_kWh = tf.cond(self.stored_energy_in_kWh < 0.0,
                                                true_fn=lambda: 0.0,
                                                false_fn=lambda: self.stored_energy_in_kWh)
        else:
            raise ValueError("Unexpected input type:" + str(type(self.stored_energy_in_kWh)))

        return self.stored_energy_in_kWh

    def update_soc_with_given_soc_and_constraints(self, state_of_charge_in_percent):
        """Updates the state of charge and the stored energy of the battery while making sure that the 
           state of charge stays within the acceptable range of [0, 100], and therefore the stored energy too.

        Keyword arguments:
        state_of_charge_in_percent -- value for the SoC that will be the new value (after applying constraints)
        """
        if isinstance(state_of_charge_in_percent, float) or isinstance(state_of_charge_in_percent, int):
            if state_of_charge_in_percent > 100.0:
                state_of_charge_in_percent = 100.0
            elif state_of_charge_in_percent < 0.0:
                state_of_charge_in_percent = 0.0
        elif isinstance(state_of_charge_in_percent, tf.Tensor):
            state_of_charge_in_percent = tf.cond(state_of_charge_in_percent > 100.0,
                                                 true_fn=lambda: 100.0,
                                                 false_fn=lambda: state_of_charge_in_percent)
            state_of_charge_in_percent = tf.cond(state_of_charge_in_percent < 0.0,
                                                 true_fn=lambda: 0.0,
                                                 false_fn=lambda: state_of_charge_in_percent)
        else:
            raise ValueError("Unexpected input type:" + str(type(state_of_charge_in_percent)))

        self.current_soc_in_percent = state_of_charge_in_percent
        self.stored_energy_in_kWh = (state_of_charge_in_percent / 100.0) * self.CAPACITY_IN_kWh

        return state_of_charge_in_percent
    
    def get_soc_resulting_from_charging_in_percent(self, soc_in_percent, charge_rate_in_W, charging_time_in_mins):
        """Returns the SoC that would result from charging the battery with the given charge rate for the
           given amount of time.
        
        Keyword arguments:
        soc_in_percent -- battery SoC (does not have to be the current one saved by the instance variable current_soc_in_percent)
        charge_rate_in_W -- the battery charge rate in Watts
        charging_time_in_mins -- how long the battery is charged with the given charge rate in minutes
        """
        charging_time_in_hours = charging_time_in_mins / 60.0
        charge_rate_in_kW = charge_rate_in_W / 1000.0
        stored_energy_in_kWh = (soc_in_percent / 100.0) * self.CAPACITY_IN_kWh
        resulting_stored_energy_in_kWh = stored_energy_in_kWh + charge_rate_in_kW * charging_time_in_hours
        resulting_soc_in_percent = (resulting_stored_energy_in_kWh / self.CAPACITY_IN_kWh) * 100.0

        return resulting_soc_in_percent

    def get_random_charge_rate_value_within_limits(self, sample_either_limit_probability=0.1):
        """Returns a random value for the battery charge rate that is within the limits the battery allows,
           where the probability of returning either the minimum or the maximum value is set by sample_limit_probability.
           The probability of sampling other values is uniformly distributed.
           The function assumes that maximum_value = - minimum_value.

        Keyword arguments:
        sample_either_limit_probability -- the probability that the sampled value the minimum or maximum value
        """
        factor = 1 / (1 - sample_either_limit_probability)
        lower_limit = -self.MAX_DISCHARGE_RATE_IN_W * factor
        upper_limit = self.MAX_CHARGE_RATE_IN_W * factor

        value = np.random.uniform(lower_limit, upper_limit)

        if value < -self.MAX_DISCHARGE_RATE_IN_W:
            return -self.MAX_DISCHARGE_RATE_IN_W
        elif value > self.MAX_CHARGE_RATE_IN_W:
            return self.MAX_CHARGE_RATE_IN_W
        else:
            return value
