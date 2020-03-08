"""This file creates the dataset, different parts of which are used for the training of the state transition model,
   the policy optimization and the test simulation."""

import pickle
from preprocessing import preprocessing as pp
from webcrawling import webcrawler_main as wc

energy_system_data_file_path = "stored_data/spreadsheet_data/energy_system/Messdaten-47850-01.01.2016-31.12.2016.xlsx"
weather_data_file_path = "stored_data/spreadsheet_data/weather/2016-11-12-2018-12-2019-01-stadt.xlsx"
s_a_pairs, subsequent_states, time_stamps, s_a_features = pp.load_state_transitions_dataset(energy_system_data_file_path,
                                                                                            weather_data_file_path,
                                                                                            1)
(train_s_a_pairs_1_min_res,
 train_states_1_min_res,
 train_time_stamp_1_min_res) = pp.get_simulation_data_of_longest_state_sequence(s_a_pairs,
                                                                                time_stamps, 
                                                                                s_a_features)

dataset_folder_path = "./stored_data/state_transitions_dataset/temporal_resolution_1_min"
with open(dataset_folder_path + '/longest_sequence_of_state_action_pairs_with_time_stamps.pkl', 'wb') as f:
    pickle.dump([train_s_a_pairs_1_min_res, train_time_stamp_1_min_res], f)
with open(dataset_folder_path + '/state_action_pair_feature_order.pkl', 'wb') as f:
    pickle.dump(s_a_features, f)

# crawls pricing data from the web and saves them to disk
train_data_first_day = train_time_stamp_1_min_res[0][:10]
train_data_last_day = train_time_stamp_1_min_res[-1][:10]
electricity_prices_in_euros_per_kWh = wc.get_weighted_average_electricity_prices_in_euros_per_kWh(start_date="08.12.2016",
                                                                                                  end_date="31.12.2016")
with open(dataset_folder_path + '/weighted_average_electricity_prices_in_euros_per_kWh_08-12-2016_31-12-2016.pkl', 'wb') as f:
    pickle.dump(electricity_prices_in_euros_per_kWh, f)

print("\nDone!")
