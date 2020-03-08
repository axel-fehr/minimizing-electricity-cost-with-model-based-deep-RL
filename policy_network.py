"""This file contains a class of a Bayesian neural network that is used as a policy in a reinforcement learning system."""
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pickle
from simulation import electricity_pricing as ep

import psutil

class PolicyNetwork:
    """Class that creates and trains a Bayesian neural network that acts as the policy."""

    def __init__(self, neurons_in_each_layer, energy_system):
        self.neurons_in_each_layer = neurons_in_each_layer
        self.energy_system = energy_system
        self.num_state_variables = neurons_in_each_layer[0]
        self.model = self._create_nn()
        num_GPUs = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=num_GPUs)
        self.state_features = self.energy_system.state_feature_order

    def _create_nn(self):
        """Creates and returns a model of a Bayesian neural network."""
        with tf.name_scope('policy_network'):
            with tf.variable_scope("policy_network"):
                model = tf.keras.Sequential(name='policy_network_model')
                model.add(tf.keras.layers.Dense(self.neurons_in_each_layer[0], activation=tf.nn.relu,
                                                input_shape=(1, self.neurons_in_each_layer[0])))
                for num_neurons in self.neurons_in_each_layer[1:-1]:
                    model.add(tf.keras.layers.Dense(num_neurons, activation=tf.nn.relu))
                model.add(tf.keras.layers.Dense(self.neurons_in_each_layer[-1], name='policy_output_layer'))

        return model

    def train(self, num_epochs, state_transition_model, save_weights_path='./stored_data/weights/policy_network/policy_network',
              load_weights=False, load_weights_path=None):
        """Trains the policy network (i.e. performs policy optimization).

        Keyword arguments:
        num_epochs -- number of epochs the network is trained
        load_weights -- Boolean value that determines whether weights from another
                        version of the network (e.g. a trained one) will be loaded before training
        save_weights_path -- file path, where the weights will be stored after training
        load_weights_path -- file path to the weights that will be loaded if 'load_weights' is set to True
        """
        with self.sess as sess:            
            stm_weights_path = ('./stored_data/weights/state_transition_model/state_transition_model_'
                                + self.energy_system.electricity_pricing_model + '.h5')
            state_transition_model.load_weights(stm_weights_path)

            starting_state = tf.placeholder(tf.float32, shape=(None, self.num_state_variables), name='starting_state')
            normalized_starting_state_vals = self._get_normalized_starting_state_values_and_set_battery_soc()
            print("\nNow creating computational graph")
            print("Available memory in GB:", psutil.virtual_memory().available / 1000000000.0)
            average_reward_per_rollout_op, average_electricity_cost_per_rollout_op = self.simulate_epoch(starting_state, state_transition_model)
            print("\nNow creating optimizer op")
            print("Available memory in GB:", psutil.virtual_memory().available / 1000000000.0)
            train_op = tf.train.AdamOptimizer().minimize(-average_reward_per_rollout_op, 
                                                         var_list=[self.model.trainable_variables],
                                                         name='policy_optimizer')
            print("\nNow merging sumaries")
            print("Available memory in GB:", psutil.virtual_memory().available / 1000000000.0)
            merged_tensorboard_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./stored_data/Tensorboard/policy_network/' + self.energy_system.electricity_pricing_model + '/')
            #writer.add_graph(sess.graph)

            print("\nNow initializing variables")
            print("Available memory in GB:", psutil.virtual_memory().available / 1000000000.0)
            self._initialize_local_and_global_variables()
            self.training_average_electricity_cost_in_euros = np.zeros(num_epochs)
            self.training_average_reward = np.zeros(num_epochs)

            if load_weights:
                self.load_weights(load_weights_path)

            feed_dict = {starting_state: normalized_starting_state_vals}
            print("\nNow running first epoch")
            print("Available memory in GB:", psutil.virtual_memory().available / 1000000000.0)
            for epoch in range(num_epochs):
                op_list = [train_op, average_reward_per_rollout_op, average_electricity_cost_per_rollout_op, merged_tensorboard_summary]
                _, epoch_average_reward, epoch_average_electricity_cost, epoch_summary = sess.run(op_list, feed_dict=feed_dict)
                self.training_average_electricity_cost_in_euros[epoch] = epoch_average_electricity_cost
                self.training_average_reward[epoch] = epoch_average_reward
                if epoch % 10 == 0 or epoch == num_epochs-1:
                    writer.add_summary(epoch_summary, epoch)
                    print("Finished " + str(epoch+1) + " epochs out of " + str(num_epochs)
                          + ". Electricity cost: %.3f euros" % epoch_average_electricity_cost)
                if epoch >= 99 and (epoch + 1) % 50 == 0:
                    self.model.save_weights(save_weights_path + '_'
                                            + self.energy_system.electricity_pricing_model + '_'
                                            + str(epoch + 1)
                                            + '.h5')

            if num_epochs > 0:
                self.save_weights_and_monitoring_data(save_weights_path)

        self.plot_progress()

    def _get_normalized_starting_state_values_and_set_battery_soc(self, starting_soc_in_percent=50.0):
        """Returns the normalized starting state with the specified battery SoC and sets 
           the battery state accordingly.

        Keyword arguments:
        starting_soc_in_percent -- the desired battery SoC that the battery must have at the beginning of the
                                   policy optimization in percent
        """
        normalized_starting_state_vals = self.energy_system.stm_train_subsequent_states_normalized[0].reshape((1, self.num_state_variables))
        normalized_starting_soc = self.normalize_variable_with_training_data(starting_soc_in_percent, 'SoC')
        normalized_starting_state_vals[0, self.energy_system.state_feature_order.index('SoC')] = normalized_starting_soc
        _ = self.energy_system.battery.update_soc_with_given_soc_and_constraints(starting_soc_in_percent)

        return normalized_starting_state_vals

    def simulate_epoch(self, starting_state, state_transition_model):
        """Simulates one epoch and returns the cumulative reward.

        Keyword arguments:
        starting_state -- state from which the epoch starts
        state_transition_model -- model that simulates state transitions
        """
        num_time_steps_in_one_week = int((7 * 24 * 60) / self.energy_system.TIME_STEP_LENGTH_IN_MINS)
        time_steps_per_epoch = num_time_steps_in_one_week
        num_rollouts = 3 # change this to 10 or 20 for a better estimate of the expected performance
        cumulative_reward = tf.Variable(tf.zeros([], dtype=tf.float32), name='epoch_reward')
        total_electricity_cost = tf.Variable(tf.zeros([], dtype=tf.float32), name='epoch_reward')

        for i in range(num_rollouts):
            with tf.name_scope('rollout_' + str(i)):
                normalized_current_state = starting_state
                for time_step in range(time_steps_per_epoch-1):
                    with tf.name_scope('time_step_' + str(time_step+1)):
                        if time_step % self.energy_system.LENGTH_OF_ACTION_INTERVAL_IN_TIME_STEPS == 0:
                            with tf.name_scope('choose_action'):
                                (normalized_unconstrained_action,
                                 normalized_constrained_action) = self._get_normalized_action_and_constrained_action(normalized_current_state)
                        with tf.name_scope('get_next_state'):
                            resulting_normalized_state = self.energy_system.compute_next_training_state_with_constraints(normalized_current_state,
                                                                                                                         normalized_constrained_action,
                                                                                                                         state_transition_model)
                        with tf.name_scope('reward_and_electricity_cost'):
                            reward, cost_of_net_drawn_electricity_in_euros = self._get_reward(normalized_current_state,
                                                                                              normalized_unconstrained_action,
                                                                                              normalized_constrained_action)
                            cumulative_reward = tf.add(cumulative_reward, reward)
                            total_electricity_cost = tf.add(total_electricity_cost, cost_of_net_drawn_electricity_in_euros)

                        normalized_current_state = resulting_normalized_state
                    print('rollout:', i, '\ttime_step:', time_step)

        average_electricity_cost_per_rollout = tf.divide(total_electricity_cost, float(num_rollouts), name='average_electricity_cost_per_rollout')
        average_cumulative_reward_per_rollout = tf.divide(cumulative_reward, float(num_rollouts), name='average_cumulative_reward_per_rollout')
        tf.summary.scalar('average_electricity_cost_per_epoch', average_electricity_cost_per_rollout)
        tf.summary.scalar('average_cumulative_reward_per_rollout', average_cumulative_reward_per_rollout)

        return average_cumulative_reward_per_rollout, average_electricity_cost_per_rollout

    def _get_normalized_action_and_constrained_action(self, current_state_normalized):
        """Returns the normalized charge rate action chosen based on the current state and its constrained equivalent,
           which is needed to meet hard constraints in the system.

        Keyword arguments:
        current_state_normalized -- normalized current state of the environment
        """
        normalized_charge_rate_action = self.model(current_state_normalized)
        denormalized_charge_rate_action_in_W = self.denormalize_network_output(normalized_charge_rate_action)
        tf.summary.scalar('denormalized_chosen_charge_rate_without_applied_constraints', tf.reshape(denormalized_charge_rate_action_in_W, []))
        corrected_denormalized_charge_rate_in_W = self.energy_system.battery.enforce_charge_rate_contraints(tf.reshape(denormalized_charge_rate_action_in_W, []),
                                                                                                            self.energy_system.TIME_STEP_LENGTH_IN_MINS)
        tf.summary.scalar('denormalized_charge_rate_action_with_applied_constraints_in_Watts', tf.reshape(corrected_denormalized_charge_rate_in_W, []))
        normalized_constrained_charge_rate_action = self.normalize_variable_with_training_data(corrected_denormalized_charge_rate_in_W, 
                                                                                               'charge_rate_in_W')

        return normalized_charge_rate_action, normalized_constrained_charge_rate_action

    def _get_reward(self, normalized_state, normalized_unconstrained_action, normalized_constrained_action):
        """Returns a reward based on the current electricity bill.

        Keyword arguments:
        normalized_state -- the normalized state of the environment
        normalized_unconstrained_action -- the normalized unconstrained action chosen by the agent in the given state
        normalized_constrained_action -- the normalized constrained charge rate action, which meets given constraints
        """
        denormalized_unconstrained_charge_rate_in_W = self.denormalize_network_output(normalized_unconstrained_action)
        denormalized_constrained_charge_rate_in_W = self.denormalize_network_output(normalized_constrained_action)
        denormalized_state = normalized_state * self.energy_system.stm_train_subsequent_states_stds + self.energy_system.stm_train_subsequent_states_means

        cost_of_net_drawn_electricity = self._get_cost_of_net_drawn_electricity_in_euros(denormalized_state, denormalized_constrained_charge_rate_in_W)
        charge_rate_punishment = self._get_punishment_for_excessive_charge_rate(denormalized_unconstrained_charge_rate_in_W)
        soc_punishment = self._get_punishment_for_impossible_resulting_soc(denormalized_state, denormalized_unconstrained_charge_rate_in_W) 
        reward = - cost_of_net_drawn_electricity - charge_rate_punishment - soc_punishment
        #tf.summary.scalar('cost_of_net_drawn_electricity in euros', cost_of_net_drawn_electricity)          
        #tf.summary.scalar('reward', reward)

        return reward, cost_of_net_drawn_electricity

    def _get_cost_of_net_drawn_electricity_in_euros(self, denormalized_state, denormalized_action):
        """Returns the cost of the net drawn electricity resulting from taking the 
           action (battery charge rate) in the given normalized_state.

        Keyword arguments:
        denormalized_state -- the denormalized state of the environment
        denormalized_action -- the denormalized action chosen by the agent in the given state
        """
        with tf.name_scope('current_electricity_bill'):
            denormalized_state = tf.reshape(denormalized_state, [self.num_state_variables,])
            current_power_drawn_from_grid_in_kW = self._get_current_power_drawn_from_grid_in_W(denormalized_state, denormalized_action) / 1000.0
            energy_drawn_from_grid_in_kWh = current_power_drawn_from_grid_in_kW * self.energy_system.TIME_STEP_LENGTH_IN_MINS / 60.0

            electricity_buy_price_in_euros_per_kWh = denormalized_state[self.state_features.index('buy_price_in_euros_per_kWh')]
            tf.summary.scalar('denormalized_electricity_buy_price_in_euros_per_kWh', electricity_buy_price_in_euros_per_kWh)
            electricity_sell_price_in_euros_per_kWh = denormalized_state[self.state_features.index('sell_price_in_euros_per_kWh')]
            tf.summary.scalar('denormalized_electricity_sell_price_in_euros_per_kWh', electricity_sell_price_in_euros_per_kWh)

            price_paid_in_euros = tf.cond(energy_drawn_from_grid_in_kWh > 0,
                                          true_fn=lambda: tf.math.multiply(energy_drawn_from_grid_in_kWh,
                                                                           tf.reshape(electricity_buy_price_in_euros_per_kWh,[])),
                                          false_fn=lambda: tf.math.multiply(energy_drawn_from_grid_in_kWh,
                                                                            tf.reshape(electricity_sell_price_in_euros_per_kWh,[])))
            #tf.summary.scalar('price paid for drawn electricity in euros', price_paid_in_euros)

        return price_paid_in_euros

    def _get_current_power_drawn_from_grid_in_W(self, denormalized_state, denormalized_action):
        """Returns the net electric power that is drawn from the grid in Watts. Return value is negative if electric
           power is being fed to the grid.

        Keyword arguments:
        denormalized_state -- the denormalized state of the environment
        denormalized_action -- the denormalized action chosen by the agent in the given state
        """
        with tf.name_scope('current_power_drawn_from_grid'):
            charge_rate_in_W = tf.reshape(denormalized_action, [])
            electric_load_in_W = denormalized_state[self.state_features.index('load_in_W')]
            electricity_generation_in_W = denormalized_state[self.state_features.index('generation_in_W')]
            current_power_drawn_from_grid = charge_rate_in_W + electric_load_in_W - electricity_generation_in_W

            tf.summary.scalar('charge rate in Watts with applied constraints', charge_rate_in_W)
            tf.summary.scalar('electric load in Watts', electric_load_in_W)
            tf.summary.scalar('electricity generation in Watts', electricity_generation_in_W)
            #tf.summary.scalar('power drawn from grid in Watts', current_power_drawn_from_grid)
            tf.summary.scalar('SoC in percent', denormalized_state[self.state_features.index('SoC')], 'SoC')

        return current_power_drawn_from_grid

    def _get_punishment_for_excessive_charge_rate(self, denormalized_charge_rate_action_in_W):
        """Returns the punishment that the agent receives if it chooses a charge rate that is
           outside the range of acceptable values (specified by the maximum charge and discharge rate).
           The punishment is 0 when the charge rate value is in an the acceptable range.

        Keyword arguments:
        denormalized_charge_rate_action_in_W -- the denormalized charge rate action chosen by the agent in the given state
        """
        denormalized_charge_rate_action_in_W = tf.reshape(denormalized_charge_rate_action_in_W, [])
        upper_limit = self.energy_system.battery.MAX_CHARGE_RATE_IN_W
        lower_limit = -self.energy_system.battery.MAX_DISCHARGE_RATE_IN_W
        with tf.name_scope('punishment_for_excessive_charge_rate'):
            punishment_for_excessive_charge_rate = tf.cond(denormalized_charge_rate_action_in_W > upper_limit, 
                                                           true_fn=lambda: tf.math.abs(tf.subtract(denormalized_charge_rate_action_in_W, 
                                                                                                   upper_limit)),
                                                           false_fn=lambda: tf.zeros(shape=[]))
            punishment_for_excessive_discharge_rate = tf.cond(denormalized_charge_rate_action_in_W < lower_limit, 
                                                              true_fn=lambda: tf.math.abs(tf.subtract(denormalized_charge_rate_action_in_W,
                                                                                                      lower_limit)),
                                                              false_fn=lambda: tf.zeros(shape=[]))
            punishment = punishment_for_excessive_charge_rate + punishment_for_excessive_discharge_rate

        return punishment

    def _get_punishment_for_impossible_resulting_soc(self, denormalized_state, denormalized_action):
        """Returns a non-zero value for the punishment that the agent receives if it chooses a charge rate that
           would push the state of charge (SoC) outside the range of possible values in the interval [0,100].

        Keyword arguments:
        denormalized_state -- the denormalized state of the environment
        denormalized_action -- the denormalized charge rate action in Watts chosen by the agent in the given state
        """
        with tf.name_scope('punishment_for_unrealistic_SoCs'):
            denormalized_state = tf.reshape(denormalized_state, [self.num_state_variables,])
            denormalized_action = tf.reshape(denormalized_action, [])
            soc_in_percent = denormalized_state[self.energy_system.state_feature_order.index('SoC')]
            resulting_soc_in_percent = self.energy_system.battery.get_soc_resulting_from_charging_in_percent(soc_in_percent,
                                                                                                             denormalized_action,
                                                                                                             self.energy_system.TIME_STEP_LENGTH_IN_MINS)
            soc_too_high_punishment = tf.cond(resulting_soc_in_percent > 100.0, 
                                              true_fn=lambda: tf.math.square(tf.math.subtract(resulting_soc_in_percent, 100.0)),
                                              false_fn=lambda: tf.zeros(shape=[]))
            soc_too_low_punishment = tf.cond(resulting_soc_in_percent < 0.0,
                                             true_fn=lambda: tf.math.square(resulting_soc_in_percent),
                                             false_fn=lambda: tf.zeros(shape=[]))
            punishment = soc_too_high_punishment + soc_too_low_punishment

        return punishment

    def _denormalize_state_feature(self, feature_value, feature_name):
        """Denormalizes a given value from a state-action pair with the means and standard deviations
           of the data the state transition model was trained with.

        Keyword arguments:
        feature_value -- value of a specific feature in a state-action pair
        feature_name -- name of the feature whose current value is provided (expected to be a string)
        """
        if feature_name in self.state_features:
            feature_idx_in_stm_output = self.state_features.index(feature_name)
            denormalized_value = ((feature_value * self.energy_system.stm_train_subsequent_states_stds[feature_idx_in_stm_output])
                                   + self.energy_system.stm_train_subsequent_states_means[feature_idx_in_stm_output])
            return denormalized_value
        else:
            raise ValueError('Unexpected feature name.')

    def load_weights(self, file_path):
        """Loads weights from a file specified by the file path.

        Keyword arguments:
        file_path -- file path to the file containing the weights
        """
        self.model.load_weights(file_path + '/policy_network.h5')
        print("\nrestored weights of the policy network.\n")

    def save_weights_and_monitoring_data(self, file_path):
        """Loads weights from a file specified by the file path.

        Keyword arguments:
        file_path -- file path where the weights will be stored
        """
        self.model.save_weights(file_path + '_' + self.energy_system.electricity_pricing_model + '.h5')
        print("\nsaved weights of the policy network to disk.\n")

        file_name_end = self.energy_system.electricity_pricing_model + '.pkl'
        with open('./stored_data/monitoring/policy_network/training_average_electricity_cost_in_euros_' + file_name_end, 'wb') as f:
            pickle.dump(self.training_average_electricity_cost_in_euros, f)
        with open('./stored_data/monitoring/policy_network/training_average_reward_' + file_name_end, 'wb') as f:
            pickle.dump(self.training_average_reward, f)
        print("saved policy monitoring data to disk.\n")

    def _initialize_local_and_global_variables(self):
        """Runs the initializers for global and local variables in a Tensorflow session."""
        variables_initialization_op = tf.group(tf.global_variables_initializer(),
                                               tf.local_variables_initializer())
        self.sess.run(variables_initialization_op)
    
    def normalize_variable_with_training_data(self, value, variable_name):
        """Normalizes the given value of a variable with the mean and the standard
           deviation this variable has in the training data.

        Keyword arguments:
        value -- the unnormalized value of the variable
        variable_name -- name of the variable whose value will be normalized as a string
        """
        variable_idx_in_state = self.energy_system.s_a_pair_feature_order.index(variable_name)
        mean = self.energy_system.stm_train_s_a_pairs_means[variable_idx_in_state]
        std = self.energy_system.stm_train_s_a_pairs_stds[variable_idx_in_state]
        normalized_value = (value - mean) / std

        return normalized_value

    def denormalize_network_output(self, normalized_output_value):
        """Denormalizes the given output value of the policy network.

        Keyword arguments:
        normalized_output_value -- raw output value of the policy network (is normalized)
        """
        charge_rate_idx_in_state_transition_model_input = self.energy_system.s_a_pair_feature_order.index('charge_rate_in_W')
        charge_rate_mean = self.energy_system.stm_train_s_a_pairs_means[charge_rate_idx_in_state_transition_model_input]
        charge_rate_std = self.energy_system.stm_train_s_a_pairs_stds[charge_rate_idx_in_state_transition_model_input]
        denormalized_value = normalized_output_value * charge_rate_std + charge_rate_mean

        return denormalized_value
    
    def plot_progress(self):
        """Plots the development of the average electricity cost over the course of training."""
        plt.plot(-self.training_average_reward, label='negative average reward')
        plt.plot(self.training_average_electricity_cost_in_euros, label='electricity cost in euros')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('cost in euros')
        plt.title('Average electricity cost in euros and reward')
        plt.show()
