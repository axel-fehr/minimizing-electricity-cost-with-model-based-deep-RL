"""This file is used to get plots for the thesis."""

import pickle
import numpy as np
from matplotlib import pyplot as plt
from state_transition_model import StateTransitionModel
from policy_network import PolicyNetwork
from simulation.energy_system import EnergySystem

pricing_models = ['constant', 'deterministic_market']
folder_path = "./stored_data/state_transitions_dataset/temporal_resolution_1_min"
pricing_model = pricing_models[0]
time_step_length_in_mins = 15
num_time_steps_in_one_week = int((7 * 24 * 60) / time_step_length_in_mins)
energy_system = EnergySystem(folder_path, electricity_pricing_model=pricing_model,
                             time_step_length_in_mins=time_step_length_in_mins, num_samples_in_test_set=num_time_steps_in_one_week)
'''
# STM part
data_folder_path = './stored_data/monitoring/state_transition_model/'
with open(data_folder_path + 'training_negative_ELBO_' + pricing_model + '.pkl', 'rb') as f:
    elbo = pickle.load(f)
with open(data_folder_path + 'training_mses_' + pricing_model + '.pkl', 'rb') as f:
    train_mses = pickle.load(f)
with open(data_folder_path + 'validation_mses_' + pricing_model + '.pkl', 'rb') as f:
    val_mses = pickle.load(f)

energy_system_features = ['generation', 'load', 'SoC']
weather_features = ['global radiation', 'diffuse radiation', 'time']
pricing_features = ['purchase price', 'selling price']
_, axes = plt.subplots(2, 4)

feature_list = ['generation', 'load', 'SoC', 'global radiation', 'diffuse radiation', 'time', 'purchase price', 'selling price']
for i, feature in enumerate(feature_list):
    subplot_x = int(np.floor(i / 4 ))
    subplot_y = i % 4
    x_axis = [x for x in range(1, train_mses.shape[0], 10)] + [train_mses.shape[0]]
    y_axis_train = [y for j, y in enumerate(train_mses[:,i]) if j % 10 == 0] + [train_mses[-1,i]]
    y_axis_val = [y for j, y in enumerate(val_mses[:,i]) if j % 10 == 0] + [val_mses[-1,i]]
    axes[subplot_x, subplot_y].plot(x_axis, y_axis_train, label=feature + ' training MSE')
    axes[subplot_x, subplot_y].plot(x_axis, y_axis_val, label=feature + ' validation MSE')
    axes[subplot_x, subplot_y].legend()
    axes[subplot_x, subplot_y].set_xlabel('Epoch')
    axes[subplot_x, subplot_y].set_ylabel('MSE')
    axes[subplot_x, subplot_y].set_ylim([0,0.7])

plt.show()
'''
'''
data_folder_path = './stored_data/monitoring/policy_network/'
with open(data_folder_path + 'training_average_electricity_cost_in_euros_constant.pkl', 'rb') as f:
    avg_el_cost_const_price = pickle.load(f)
with open(data_folder_path + 'training_average_electricity_cost_in_euros_deterministic_market.pkl', 'rb') as f:
    avg_el_cost_market_price = pickle.load(f)

_, axes = plt.subplots(1, 2)
x_axis = np.concatenate((np.arange(1,5000,10, dtype=int), 5000*np.ones(1, dtype=int)))
axes[0].plot(x_axis, avg_el_cost_const_price[x_axis-1])
axes[0].set_ylim([0.5, 1.48])
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Average electricity cost in euros')
axes[0].set_title('Average electricity cost with constant pricing')
axes[1].plot(x_axis, avg_el_cost_market_price[x_axis-1])
axes[1].set_ylim([0.5, 1.48])
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Average electricity cost in euros')
axes[1].set_title('Average electricity cost with variable pricing')
plt.show()
'''
'''
pricing_model = pricing_models[0]
energy_system = EnergySystem(folder_path, electricity_pricing_model=pricing_model,
                             time_step_length_in_mins=time_step_length_in_mins, num_samples_in_test_set=num_time_steps_in_one_week)
policy_network = PolicyNetwork([energy_system.num_state_variables, 16, 32, 1], energy_system)
energy_system.policy_type = 'rule_based'
formula_pol_cost_constant_price = energy_system.run_test_simulation()
energy_system.policy_type = 'neural_network'
energy_system.policy_network = policy_network
nn_pol_cost_constant_price = energy_system.run_test_simulation()
energy_system.policy_type = 'random'
random_pol_costs_constant_price = np.zeros(20)
for i in range(20):
    random_pol_costs_constant_price[i] = energy_system.run_test_simulation()
avg_random_pol_costs_constant_price = np.mean(random_pol_costs_constant_price)
print("Std of random constant:", np.std(random_pol_costs_constant_price))

pricing_model = pricing_models[1]
energy_system = EnergySystem(folder_path, electricity_pricing_model=pricing_model,
                             time_step_length_in_mins=time_step_length_in_mins, num_samples_in_test_set=num_time_steps_in_one_week)
policy_network = PolicyNetwork([energy_system.num_state_variables, 16, 32, 1], energy_system)
energy_system.policy_type = 'rule_based'
formula_pol_cost_market_price = energy_system.run_test_simulation()
energy_system.policy_type = 'neural_network'
energy_system.policy_network = policy_network
nn_pol_cost_market_price = energy_system.run_test_simulation()
energy_system.policy_type = 'random'
random_pol_costs_market_price = np.zeros(20)
for i in range(20):
    random_pol_costs_market_price[i] = energy_system.run_test_simulation()
avg_random_pol_cost_market_price = np.mean(random_pol_costs_market_price)
print("Std of random market:", np.std(random_pol_costs_market_price))

_, axes = plt.subplots(1, 2)
bar_labels = ['random','formula','neural network']
bar_width = 0.6
axes[0].bar(bar_labels, [avg_random_pol_costs_constant_price, formula_pol_cost_constant_price, nn_pol_cost_constant_price], width=bar_width)
axes[0].set_ylabel('Electricity cost in euros')
axes[0].set_title('Electricity cost with constant pricing')
axes[1].bar(bar_labels, [avg_random_pol_cost_market_price, formula_pol_cost_market_price, nn_pol_cost_market_price], width=bar_width)
axes[1].set_ylabel('Electricity cost in euros')
axes[1].set_title('Electricity cost with variable pricing')
plt.show()
'''
'''
def relu(x):
    if x < 0:
        return 0
    return x

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

step_size = 0.01
x_axis = np.arange(-5, 5 + step_size, step_size)
sigmoid_vals = np.zeros(len(x_axis))
tanh_vals = np.zeros(len(x_axis))
relu_vals = np.zeros(len(x_axis))
for i, x in enumerate(x_axis):
    sigmoid_vals[i] = sigmoid(x)
    tanh_vals[i] = np.tanh(x)
    relu_vals[i] = relu(x)

plt.rcParams.update({'font.size': 11})
label_and_title_size = 'x-large'
_, axes = plt.subplots(1, 3)
axes[0].plot(x_axis, sigmoid_vals)
axes[0].set_xlim([-5, 5])
axes[0].set_ylim([-1.5, 1.5])
axes[0].grid(True)
axes[0].set_xlabel('z', size=label_and_title_size)
axes[0].set_ylabel('g(z)', size=label_and_title_size)
axes[0].set_title('Sigmoid function', size=label_and_title_size)
axes[1].plot(x_axis, tanh_vals)
axes[1].set_xlim([-5, 5])
axes[1].set_ylim([-1.5, 1.5])
axes[1].grid(True)
axes[1].set_xlabel('z', size=label_and_title_size)
axes[1].set_ylabel('g(z)', size=label_and_title_size)
axes[1].set_title('Hyperbolic tangent', size=label_and_title_size)
axes[2].plot(x_axis, relu_vals)
axes[2].set_xlim([-5, 5])
axes[2].set_ylim([-1, 5])
axes[2].grid(True)
axes[2].set_xlabel('z', size=label_and_title_size)
axes[2].set_ylabel('g(z)', size=label_and_title_size)
axes[2].set_title('Rectified linear unit', size=label_and_title_size)
plt.show()
'''