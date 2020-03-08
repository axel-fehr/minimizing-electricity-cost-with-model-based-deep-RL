"""This file contains a class of a an energy system with PV panels and a battery."""

import numpy as np
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from preprocessing import preprocessing as pp
from simulation import battery
from simulation import pv_system
from simulation import electricity_pricing as ep

class EnergySystem:
    """Class that simulates an energy system with PV panels and a battery."""
    def __init__(self, simulation_data_folder_path, electricity_pricing_model, time_step_length_in_mins,
                 num_samples_in_test_set, length_of_action_interval_in_time_steps=1, price_update_interval_in_mins=15,
                 policy_type=None, policy_network=None):
        self.policy_network = policy_network
        self.policy_type = policy_type
        self.battery = battery.Battery()
        self.pv_system = pv_system.PhotoVoltaicSystem()
        self.electricity_bill_in_euros = 0.0
        self.LENGTH_OF_ACTION_INTERVAL_IN_TIME_STEPS = length_of_action_interval_in_time_steps
        self.TIME_STEP_LENGTH_IN_MINS = time_step_length_in_mins
        self.PRICE_UPDATE_INTERVAL_IN_MINS = price_update_interval_in_mins
        self.electricity_pricing_model = electricity_pricing_model
        self._create_instance_variables_from_simulation_data(simulation_data_folder_path, num_samples_in_test_set)
        self._test_for_consistent_time_stamp_difference()

        if (price_update_interval_in_mins < time_step_length_in_mins
            or price_update_interval_in_mins % time_step_length_in_mins != 0):
            raise ValueError("Price update interval has to be multiple of the time step length.")

    def _create_instance_variables_from_simulation_data(self, folder_path, num_samples_in_test_set):
        """Creates instance variables corresponding to the states used in the simulation, their time
           stamps and also a list of the state variables in the order they were added.

        Keyword arguments:
        folder_path -- file path of the folder that contains the data that will be loaded
        """
        with open(folder_path + '/longest_sequence_of_state_action_pairs_with_time_stamps.pkl', 'rb') as f:
            state_action_pairs_sequence_1_min_res, s_a_pairs_time_stamps_sequence_1_min_res = pickle.load(f)
        with open(folder_path + '/state_action_pair_feature_order.pkl', 'rb') as f:
            self.s_a_pair_feature_order = pickle.load(f)

        (s_a_pairs_with_electricity_prices_1_min_res, 
         self.s_a_pair_feature_order) = ep.add_electricity_buy_and_sell_prices_to_s_a_pairs(state_action_pairs_sequence_1_min_res,
                                                                                            s_a_pairs_time_stamps_sequence_1_min_res,
                                                                                            self.s_a_pair_feature_order,
                                                                                            self.electricity_pricing_model,
                                                                                            self.PRICE_UPDATE_INTERVAL_IN_MINS)

        charge_rate_action_idx = self.s_a_pair_feature_order.index('charge_rate_in_W')
        self.state_feature_order = self.s_a_pair_feature_order[:charge_rate_action_idx] + self.s_a_pair_feature_order[charge_rate_action_idx+1:]
        self.num_state_variables = len(self.state_feature_order)

        (s_a_pairs_with_electricity_prices_correct_res,
         s_a_pairs_time_stamps_sequence_correct_res) = pp.subsample_data(s_a_pairs_with_electricity_prices_1_min_res,
                                                                         s_a_pairs_time_stamps_sequence_1_min_res,
                                                                         self.TIME_STEP_LENGTH_IN_MINS)
        state_sequence_with_prices_correct_res = np.concatenate((s_a_pairs_with_electricity_prices_correct_res[:,:charge_rate_action_idx],
                                                                 s_a_pairs_with_electricity_prices_correct_res[:,charge_rate_action_idx+1:]),
                                                                axis=1)

        self.test_s_a_pairs = s_a_pairs_with_electricity_prices_correct_res[:num_samples_in_test_set,:]
        self.test_states = state_sequence_with_prices_correct_res[:num_samples_in_test_set,:]
        self.test_time_stamps = s_a_pairs_time_stamps_sequence_correct_res[:num_samples_in_test_set]
        self.stm_train_s_a_pairs = None # s_a_pairs_with_electricity_prices_correct_res[num_samples_in_test_set:,:]
        self.stm_train_subsequent_states = None # state_sequence_with_prices_correct_res[num_samples_in_test_set:,:]
        self.stm_train_time_stamps = s_a_pairs_time_stamps_sequence_correct_res[num_samples_in_test_set:]

        with open(folder_path + '/stm_train_state_action_pairs_and_subsequent_states_normalized_temp_res_' 
                  + str(self.TIME_STEP_LENGTH_IN_MINS) + '_' + self.electricity_pricing_model + '.pkl', 'rb') as f:
            self.stm_train_s_a_pairs_normalized, self.stm_train_subsequent_states_normalized = pickle.load(f)
        with open(folder_path + '/stm_train_state_action_pairs_means_and_stds_temp_res_' 
                  + str(self.TIME_STEP_LENGTH_IN_MINS) + '_' + self.electricity_pricing_model + '.pkl', 'rb') as f:
            self.stm_train_s_a_pairs_means, self.stm_train_s_a_pairs_stds = pickle.load(f)
        self.stm_train_subsequent_states_means = np.delete(self.stm_train_s_a_pairs_means, charge_rate_action_idx)
        self.stm_train_subsequent_states_stds = np.delete(self.stm_train_s_a_pairs_stds, charge_rate_action_idx)
        '''
        # used to create the STM training set and save it to disk
        self.stm_train_s_a_pairs, self.stm_train_subsequent_states = self.create_stm_training_set()
        (self.stm_train_s_a_pairs_normalized,
         self.stm_train_s_a_pairs_means,
         self.stm_train_s_a_pairs_stds) = pp.get_z_scores_with_means_and_stds(self.stm_train_s_a_pairs)
        self.stm_train_subsequent_states_means = np.delete(self.stm_train_s_a_pairs_means, charge_rate_action_idx)
        self.stm_train_subsequent_states_stds = np.delete(self.stm_train_s_a_pairs_stds, charge_rate_action_idx)
        self.stm_train_subsequent_states_normalized = pp.normalize_with_given_means_and_stds(self.stm_train_subsequent_states,
                                                                                             self.stm_train_subsequent_states_means,
                                                                                             self.stm_train_subsequent_states_stds)
        with open(folder_path + '/stm_train_state_action_pairs_and_subsequent_states_temp_res_' 
                  + str(self.TIME_STEP_LENGTH_IN_MINS) + '_' + self.electricity_pricing_model + '.pkl', 'wb') as f:
            pickle.dump([self.stm_train_s_a_pairs, self.stm_train_subsequent_states], f)
        with open(folder_path + '/stm_train_state_action_pairs_and_subsequent_states_normalized_temp_res_' 
                  + str(self.TIME_STEP_LENGTH_IN_MINS) + '_' + self.electricity_pricing_model + '.pkl', 'wb') as f:
            pickle.dump([self.stm_train_s_a_pairs_normalized, self.stm_train_subsequent_states_normalized], f)
        with open(folder_path + '/stm_train_state_action_pairs_means_and_stds_temp_res_' 
                  + str(self.TIME_STEP_LENGTH_IN_MINS) + '_' + self.electricity_pricing_model + '.pkl', 'wb') as f:
            pickle.dump([self.stm_train_s_a_pairs_means, self.stm_train_s_a_pairs_stds], f)
        '''

    def _test_for_consistent_time_stamp_difference(self):
        """Tests if the the number of minutes between the time stamps in all pairs of time stamps 
          corresponding to two subsequent states in the data is the same as the specified length of
          each time step.
        """
        for i in range(len(self.stm_train_time_stamps)-1):
            expected_time_stamp = pp.get_corresponding_time_stamp_string(self.stm_train_time_stamps[i], self.TIME_STEP_LENGTH_IN_MINS)
            if self.stm_train_time_stamps[i+1] != expected_time_stamp:
                raise ValueError("Not all time stamps have the expected temporal difference (the specified time step length).")

        for i in range(len(self.test_time_stamps)-1):
            expected_time_stamp = pp.get_corresponding_time_stamp_string(self.test_time_stamps[i], self.TIME_STEP_LENGTH_IN_MINS)
            if self.test_time_stamps[i+1] != expected_time_stamp:
                raise ValueError("Not all time stamps have the expected temporal difference (the specified time step length).")

    def create_stm_training_set(self):
        """Returns a sequence of state transitions for the state transition model corresponding to the chosen properties of the energy system."""
        state_action_pairs = np.empty(self.stm_train_s_a_pairs[:-1].shape)
        subsequent_states = np.empty(self.stm_train_subsequent_states[:-1].shape)

        state = self.stm_train_subsequent_states[0]
        for time_step in range(self.stm_train_s_a_pairs.shape[0]-1):
            self.battery.update_soc_with_given_soc_and_constraints(np.random.uniform(0, 100))
            state[self.state_feature_order.index('SoC')] = self.battery.current_soc_in_percent
            charge_rate_action_in_W = self.battery.get_random_charge_rate_value_within_limits()
            next_state = self.compute_next_state_with_constraints(state, charge_rate_action_in_W, time_step, train_mode=True)
            state_action_pair = np.insert(state, self.s_a_pair_feature_order.index('charge_rate_in_W'), charge_rate_action_in_W)

            state_action_pairs[time_step] = state_action_pair
            subsequent_states[time_step] = next_state

            state = next_state

        return state_action_pairs, subsequent_states

    def run_test_simulation(self):
        """Runs a simulation with the provided policy and simulation data."""
        self.battery.update_soc_with_given_soc_and_constraints(self.test_states[0, self.state_feature_order.index('SoC')])
        self.electricity_bill_in_euros = 0.0

        if self.policy_type == 'neural_network':
            self.policy_network.model.load_weights('./stored_data/weights/policy_network/policy_network_' + self.electricity_pricing_model + '.h5')

        state = self.test_states[0]
        for time_step, time_stamp in enumerate(self.test_time_stamps[:-1]):
            if time_step % self.LENGTH_OF_ACTION_INTERVAL_IN_TIME_STEPS == 0:
                action = self._get_policy_action(state)
            next_state = self.compute_next_state_with_constraints(state, action, time_step, train_mode=False)
            self.electricity_bill_in_euros = self.update_electricity_bill(state, action, time_stamp, time_step)
            state = next_state

        print(self.policy_type + ":", "Accumulated electricity bill in euros (lower is better): %.4f" % self.electricity_bill_in_euros)

        return self.electricity_bill_in_euros

    def _get_policy_action(self, state):
        """Returns the action that would be taken by the used policy in the given state.

        Keyword arguments:
        state -- state based on which the policy chooses an action
        """
        if self.policy_type == 'neural_network':
            action = self._choose_action_with_policy_network(state)
        elif self.policy_type == 'rule_based':
            action = self._choose_action_with_rule(state)
        elif self.policy_type == 'random':
            action = self._choose_random_action()
        else:
            raise ValueError("Unexpected policy type: " + self.policy_type)

        corrected_action = self.battery.enforce_charge_rate_contraints(float(action), self.TIME_STEP_LENGTH_IN_MINS)

        return corrected_action

    def compute_next_state_with_constraints(self, state, action, time_step, train_mode,
                                            compute_battery_soc_with_battery_properties=True,
                                            compute_electricity_generation_with_solar_array_properties=False):
        """Returns the next state in the test dataset based on the action taken in the given state
           while taking into account physical constraints.

        Keyword arguments:
        state -- state of the environment in which the given action is taken
        action -- action chosen by the policy in the given state (charge rate in Watts)
        time_step -- current time step of the simulation (starting from 0)
        train_mode -- Boolean variable that indicates whether the next state is computed for training or for testing
        compute_battery_soc_with_battery_properties -- Boolean value that determines whether the given value for the battery SoC will
                                                       be subsituted by one computed based on a formula that takes into account the chosen
                                                       battery capacity (member variable of the battery object)
        compute_electricity_generation_with_solar_array_properties -- Boolean value that determines whether the given value for
                                                                      the electricity generation will be substituted by one computed based on
                                                                      a rule and the size of the solar array in the simulation
        """
        if train_mode:
            next_state = self.stm_train_subsequent_states[time_step+1]
        else:
            next_state = self.test_states[time_step+1]

        if compute_battery_soc_with_battery_properties:
            battery_soc_in_percent = self.battery.update_soc_with_charge_rate_and_constraints(action, self.TIME_STEP_LENGTH_IN_MINS)
            next_state[self.state_feature_order.index('SoC')] = battery_soc_in_percent

        if compute_electricity_generation_with_solar_array_properties:
            global_radiation_in_W_per_square_meter = next_state[self.state_feature_order.index('global_radiation_in_W_per_square_meter')]
            current_generation_in_W = self.pv_system.compute_electricity_generation_in_W(global_radiation_in_W_per_square_meter)
            next_state[self.state_feature_order.index('generation_in_W')] = current_generation_in_W

        return next_state

    def compute_next_training_state_with_constraints(self, normalized_state, normalized_action, state_transition_model):
        """Returns the next state in the training dataset based on the action taken in the given state while taking into account
           physical constraints.

        Keyword arguments:
        normalized_state -- normalized state of the environment in which the given action is taken
        normalized_action -- normalized action chosen by the policy in the given state (charge rate in Watts)
        state_transition_model -- model that simulates state transitions
        """
        normalized_state = tf.reshape(normalized_state, [1, len(self.state_feature_order)])
        normalized_action = tf.reshape(normalized_action, [1, 1])
        action_idx_in_s_a_pair = self.s_a_pair_feature_order.index('charge_rate_in_W')
        normalized_state_action_pair = tf.concat((normalized_state[:, :action_idx_in_s_a_pair], 
                                                  normalized_action, 
                                                  normalized_state[:, action_idx_in_s_a_pair:]), 
                                                 axis=1)

        normalized_input_soc = normalized_state[0, self.state_feature_order.index('SoC')]
        tf.summary.scalar('denormalized_STM_input_SoC', tf.reshape(self._denormalize_variable_with_training_data(normalized_input_soc, 'SoC'), []))

        stm_output_distribution = tfd.Normal(loc=state_transition_model.model(normalized_state_action_pair),
                                             scale=state_transition_model.output_stds)
        next_state_normalized = stm_output_distribution.sample()

        normalized_next_soc = next_state_normalized[0, self.state_feature_order.index('SoC')]
        denormalized_next_soc = self._denormalize_variable_with_training_data(normalized_next_soc, 'SoC')
        tf.summary.scalar('denormalized_SoC_from_STM_without_constraints', tf.reshape(denormalized_next_soc, []))

        # substituting the SoC from the STM with one computed with a formula
        charge_rate_in_W = self._denormalize_variable_with_training_data(normalized_action, 'charge_rate_in_W')
        next_soc_in_percent = self.battery.get_soc_resulting_from_charging_in_percent(self.battery.current_soc_in_percent,
                                                                                      tf.reshape(charge_rate_in_W, []),
                                                                                      self.TIME_STEP_LENGTH_IN_MINS)
        constrained_next_soc = self.battery.update_soc_with_given_soc_and_constraints(tf.reshape(next_soc_in_percent, []))
        tf.summary.scalar('constrained_next_SoC_from_computation_formula', tf.reshape(constrained_next_soc, []))
        normalized_constrained_next_soc = self.normalize_variable_with_training_data(constrained_next_soc, 'SoC')
        next_state_normalized = self.substitute_old_value_of_state_variable(next_state_normalized, normalized_constrained_next_soc, 'SoC')

        # substituting the time feature value with one computed with a formula
        normalized_time_stamp = normalized_state[0, self.state_feature_order.index('time_stamps')]
        denormalized_time_stamp = self._denormalize_variable_with_training_data(normalized_time_stamp, 'time_stamps')
        tf.summary.scalar('current_time_stamp', tf.reshape(denormalized_time_stamp, []))
        next_time_stamp = denormalized_time_stamp + self.TIME_STEP_LENGTH_IN_MINS
        normalized_next_time_stamp = self.normalize_variable_with_training_data(next_time_stamp, 'time_stamps')
        next_state_normalized = self.substitute_old_value_of_state_variable(next_state_normalized, normalized_next_time_stamp, 'time_stamps')

        return next_state_normalized

    def _choose_action_with_policy_network(self, state):
        """Returns the action (the battery charge rate) that the policy network chooses in
           the given state.

        Keyword arguments:
        state -- state based on which the policy network chooses an action
        """
        normalized_state = (state - self.stm_train_subsequent_states_means) / self.stm_train_subsequent_states_stds
        normalized_charge_rate = self.policy_network.model.predict(normalized_state.reshape((1,1,len(state))))
        denormalized_charge_rate_in_W = self.policy_network.denormalize_network_output(normalized_charge_rate)

        return denormalized_charge_rate_in_W

    def _choose_action_with_rule(self, state):
        """Returns the action (the battery charge rate) that the rule-based baseline policy 
           chooses in the given state.

        Keyword arguments:
        state -- state based on which the policy chooses an action
        """
        generation_in_W = state[self.state_feature_order.index('generation_in_W')]
        load_in_W = state[self.state_feature_order.index('load_in_W')]
        charge_rate_in_W = generation_in_W - load_in_W

        return charge_rate_in_W

    def _choose_random_action(self):
        """Returns a random action that is drawn from a unifrom distribution covering the
           interval from the maximum discharge rate of the battery to the maximum charge rate.
        """
        charge_rate_in_W = np.random.uniform(low=-self.battery.MAX_DISCHARGE_RATE_IN_W, high=self.battery.MAX_CHARGE_RATE_IN_W)

        return charge_rate_in_W

    def update_electricity_bill(self, state, charge_rate_in_W, time_stamp_string, time_step):
        """Updates the electricity bill.

        Keyword arguments:
        state -- state of the environment in which the given charge rate is used
        charge_rate_in_W -- rate at which the battery is charged in Watts
        time_stamp_string -- time stamp string of the given state in the format of "dd.mm.yyyy hh:mm"
        time_step -- current time step of the test simulation (starting from 0)
        """
        grid_drawn_electricity_in_kWh = self.compute_drawn_electricity_amount_in_kWh(state, charge_rate_in_W)
        current_electricity_buy_price_in_euros_per_kWh = state[self.state_feature_order.index('buy_price_in_euros_per_kWh')]
        current_electricity_sell_price_in_euros_per_kWh = state[self.state_feature_order.index('sell_price_in_euros_per_kWh')]

        if grid_drawn_electricity_in_kWh > 0:
            buy_price_in_euros = grid_drawn_electricity_in_kWh * current_electricity_buy_price_in_euros_per_kWh
            return self.electricity_bill_in_euros + buy_price_in_euros
        else:
            sell_price_in_euros = abs(grid_drawn_electricity_in_kWh * current_electricity_sell_price_in_euros_per_kWh)
            return self.electricity_bill_in_euros - sell_price_in_euros

    def compute_drawn_electricity_amount_in_kWh(self, state, charge_rate_in_W):
        """Computes the amount of electricity (in kWh) drawn from the grid, which 
           is negative if electricity is fed into the grid.

        Keyword arguments:
        state -- state of the environment in which the given charge rate is used
        charge_rate_in_W -- rate at which the battery is charged (or discharged if < 0) in W
        """
        electric_load_in_W = state[self.state_feature_order.index('load_in_W')]
        electricity_generation_in_W = state[self.state_feature_order.index('generation_in_W')]
        drawn_power_in_W = charge_rate_in_W + electric_load_in_W - electricity_generation_in_W
        time_step_length_in_hours = self.TIME_STEP_LENGTH_IN_MINS / 60.0
        grid_drawn_electricity_in_kWh = drawn_power_in_W * time_step_length_in_hours / 1000.0

        return grid_drawn_electricity_in_kWh

    def normalize_variable_with_training_data(self, value, variable_name):
        """Normalizes the given value of a variable with the mean and the standard
           deviation this variable has in the training data.

        Keyword arguments:
        value -- the unnormalized value of the variable
        variable_name -- name of the variable whose value will be normalized as a string
        """
        variable_idx_in_state = self.s_a_pair_feature_order.index(variable_name)
        mean = self.stm_train_s_a_pairs_means[variable_idx_in_state]
        std = self.stm_train_s_a_pairs_stds[variable_idx_in_state]
        normalized_value = tf.reshape((value - mean) / std, [1,1])
        normalized_value = tf.dtypes.cast(normalized_value, tf.float32)

        return normalized_value

    def _denormalize_variable_with_training_data(self, value, variable_name):
        """Denormalizes the given normalized value of a variable with the mean and the standard
           deviation this variable has in the training data.

        Keyword arguments:
        value -- the unnormalized value of the state variable
        variable_name -- name of the state variable whose value will be normalized as a string
        """
        variable_idx_in_s_a_pair = self.s_a_pair_feature_order.index(variable_name)
        mean = self.stm_train_s_a_pairs_means[variable_idx_in_s_a_pair]
        std = self.stm_train_s_a_pairs_stds[variable_idx_in_s_a_pair]
        denormalized_value = tf.reshape(value * std + mean, [1,1])
        denormalized_value = tf.dtypes.cast(denormalized_value, tf.float32)

        return denormalized_value

    def substitute_old_value_of_state_variable(self, state, new_value, variable_name):
        """Substitutes the current value of a state variable in the state vector, which is
           a Tensorflow tensor with a new value.

        Keyword arguments:
        state -- state vector that contains the values of all state variables
        new_value -- value of the state variable that substitutes the old value
        variable_name -- name of the state variable whose value will be substituted as a string
        """
        variable_idx_in_state = self.state_feature_order.index(variable_name)
        state_slice_1 = tf.reshape(state[0,:variable_idx_in_state], (1, variable_idx_in_state))
        state_slice_2 = tf.reshape(state[0,variable_idx_in_state+1:], (1, self.num_state_variables - variable_idx_in_state - 1))
        state = tf.concat([state_slice_1, new_value, state_slice_2], axis=1)

        return state
