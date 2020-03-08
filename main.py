"""This is the main file. It loads a trained state transition model or trains one (see comment below) and trains a policy
   network with the trained state transition model."""

from state_transition_model import StateTransitionModel
from policy_network import PolicyNetwork
from simulation.energy_system import EnergySystem
import psutil

pricing_models = ['constant', 'deterministic_market']
dataset_folder_path = "./stored_data/state_transitions_dataset/temporal_resolution_1_min"
pricing_model = pricing_models[0]
time_step_length_in_mins = 15
num_time_steps_in_one_week = int((7 * 24 * 60) / time_step_length_in_mins)
energy_system = EnergySystem(dataset_folder_path,
                             electricity_pricing_model=pricing_model,
                             time_step_length_in_mins=time_step_length_in_mins,
                             num_samples_in_test_set=num_time_steps_in_one_week)
'''
# Trains the state transition model (has to be done before training the policy network)
normalized_state_action_pairs = energy_system.stm_train_s_a_pairs_normalized 
normalized_subsequent_states = energy_system.stm_train_subsequent_states_normalized

state_transition_model = StateTransitionModel([normalized_state_action_pairs.shape[1], 16, 32, 64, normalized_subsequent_states.shape[1]])
state_transition_model.train(normalized_state_action_pairs, normalized_subsequent_states, num_epochs=1000,
                             pricing_model=pricing_model, plot_errors=True, load_weights=False)
'''
# Trains the policy network with the trained state transition model
state_transition_model = StateTransitionModel([energy_system.num_state_variables+1, 16, 32, 64, energy_system.num_state_variables])

policy_network = PolicyNetwork([energy_system.num_state_variables, 16, 32, 1], energy_system)
print("\nStarting training...")
policy_network.train(num_epochs=1000, state_transition_model=state_transition_model,
                     load_weights=False, load_weights_path='./stored_data/weights/policy_network')
'''
# Compares the policy network trained under a chosen pricing model with other policies
policy_network = PolicyNetwork([energy_system.num_state_variables, 16, 32, 1], energy_system)

print("\nWith pricing model " + pricing_model + ":")
energy_system.policy_type = 'rule_based'
energy_system.run_test_simulation()
energy_system.policy_type = 'random'
energy_system.run_test_simulation()
energy_system.policy_type = 'neural_network'
energy_system.policy_network = policy_network
energy_system.run_test_simulation()
'''
print("\nDone!")
