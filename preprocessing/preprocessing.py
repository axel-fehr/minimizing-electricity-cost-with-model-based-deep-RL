"""This file contains functions for loading and preprocessing data."""

import pandas as pd
import numpy as np
import pickle
from preprocessing import preprocessing_tests as pt
from preprocessing import merging_data as md

def load_state_transitions_dataset(energy_system_file_path, weather_data_file_path, temporal_resolution_in_mins):
    """Returns a dataset of state transitions along with their corresponding time stamps and the used features
       in the order they are were added.

    Keyword arguments:
    energy_system_file_path -- file path to the Excel file containing data from the energy system
    weather_data_file_path -- file path to the Excel file containing weather data
    temporal_resolution_in_mins -- the desired temporal resolution of the dataset (or the time 
                                   between each data point) in minutes
    """
    energy_system_data = get_energy_system_data(energy_system_file_path)
    weather_data = get_weather_data(weather_data_file_path)
    merged_data_with_time_stamps, state_action_pair_feature_order = md.merge_data_with_same_time_stamp(energy_system_data,
                                                                                                       weather_data)
    merged_data_with_time_stamps = replace_missing_data_with_interpolations(merged_data_with_time_stamps)
    s_a_pairs, subsequent_states, state_transition_time_stamps = get_state_transitions_as_separate_arrays(merged_data_with_time_stamps,
                                                                                                          temporal_resolution_in_mins,
                                                                                                          state_action_pair_feature_order)

    return s_a_pairs, subsequent_states, state_transition_time_stamps, state_action_pair_feature_order


def get_simulation_data_of_longest_state_sequence(state_action_pairs, time_stamps, state_action_pair_feature_order):
    """Returns the states and the state-action pairs of the longest, uninterrupted state sequence along with their time stamps.
       The temporal resolution of the data will be 1 minute.

    Keyword arguments:
    state_action_pairs -- 2D array with state-action pairs from a recorded dataset (rows: s-a pairs, columns: features)
    time_stamps -- list of time stamps of the of the state-action pairs. Time stamp format: "dd.mm.yyyy hh:mm"
    state_action_pair_feature_order -- list of the features contained in each state-action pair apart from the time 
                                       stamp strings in the order they were added
    """
    action_idx_in_s_a_pair = state_action_pair_feature_order.index('charge_rate_in_W')
    states = np.concatenate((state_action_pairs[:, :action_idx_in_s_a_pair], state_action_pairs[:, action_idx_in_s_a_pair+1:]), axis=1)

    state_idxs_in_longest_sequence = find_idxs_of_n_longest_state_sequences(states, time_stamps, n=1, temporal_resolution_in_mins=1)
    training_states_idxs = state_idxs_in_longest_sequence[0]
    training_states = states[training_states_idxs]
    training_s_a_pairs = state_action_pairs[training_states_idxs]
    training_states_time_stamps = time_stamps[training_states_idxs[0]:training_states_idxs[-1]+1]

    return training_s_a_pairs, training_states, training_states_time_stamps


def subsample_data(data, time_stamps_1_min_res, temporal_resolution_in_mins):
    """Subsamples the given data so that the given temporal resolution is achieved.

    Keyword arguments:
    data -- 2-D array of data, where each row of data has a corresponding time stamp (temporal 
            resolution is expected to be 1 minute)
    time_stamps_1_min_res -- list of ordered time stamps in the format of "dd.mm.yyyy hh:mm";
                             each time stamp is only one minute apart from the next
    temporal_resolution_in_mins -- temporal resolution that the subsampled data should have in minutes
    """
    if temporal_resolution_in_mins == 1:
        return data, time_stamps_1_min_res

    subsampled_data = data[0].reshape((1,data.shape[1]))
    subsampled_time_stamps = [time_stamps_1_min_res[0]]

    for i in range(temporal_resolution_in_mins, data.shape[0], temporal_resolution_in_mins):
        subsampled_data = np.concatenate((subsampled_data, data[i].reshape(1,data.shape[1])), axis=0)
        subsampled_time_stamps.append(time_stamps_1_min_res[i])

    return subsampled_data, subsampled_time_stamps


def get_simulation_data(state_action_pairs, time_stamps, state_action_pair_feature_order,
                        temporal_resolution_in_mins, num_days_of_simulation):
    """Returns a sequence of states used for the simulation of the energy system along with
       the time stamps of the states in the sequence.

    Keyword arguments:
    state_action_pairs -- 2D array with state-action pairs from a recorded dataset (rows: s-a pairs, columns: features)
    time_stamps -- list of time stamps of the of the state-action pairs. Time stamp format: "dd.mm.yyyy hh:mm"
    state_action_pair_feature_order -- list of the features contained in each state-action pair apart from the time 
                                       stamp strings in the order they were added
    temporal_resolution_in_mins -- temporal resolution the simulation data should have (or how many minutes are between
                                   each subsequent state in the dataset)
    num_days_of_simulation -- the number of days that have to be simulated
    """
    action_idx_in_s_a_pair = state_action_pair_feature_order.index('charge_rate_in_W')
    states = np.concatenate((state_action_pairs[:, :action_idx_in_s_a_pair], state_action_pairs[:, action_idx_in_s_a_pair+1:]), axis=1)

    state_idxs_in_longest_sequence = find_idxs_of_n_longest_state_sequences(states, time_stamps, 1, temporal_resolution_in_mins) 
    num_states_in_longest_sequence = len(state_idxs_in_longest_sequence[0])
    simulation_period_in_mins = num_days_of_simulation * 24 * 60
    necessary_sequence_length = int(np.ceil(simulation_period_in_mins / temporal_resolution_in_mins))

    if necessary_sequence_length > num_states_in_longest_sequence:
        raise ValueError("No uninterrupted state sequence that is long enough. "
                         + "Necessary sequence length: " + str(necessary_sequence_length)
                         + ". Sequence length of largest available sequence: " + str(num_states_in_longest_sequence))

    simulation_states_idxs = state_idxs_in_longest_sequence[0][:necessary_sequence_length]
    simulation_states = states[simulation_states_idxs]
    simulation_states_time_stamps = time_stamps[simulation_states_idxs[0]:simulation_states_idxs[-1]+1]

    return simulation_states, simulation_states_time_stamps


def get_stm_training_data(state_action_pairs, resulting_states, s_a_pairs_time_stamps, simulation_time_stamps):
    """Returns two 2D-arrays with training data for the state transition model (STM). The first array contains the 
       state-action pairs (rows: s-a pairs, columns: features) and the other one contains the states resulting 
       from the state-action pairs (rows: states, columns: features).

    Keyword arguments:
    state_action_pairs -- 2D array with state-action pairs from a recorded dataset (rows: s-a pairs, columns: features)
    resulting_states -- 2D array containing the states resulting from the state-action pairs (rows: states, columns: features)
    s_a_pairs_time_stamps -- list containing the time stamps of each state-action pair. Time stamp format: "dd.mm.yyyy hh:mm"
    simulation_time_stamps -- time stamps of the state-action pairs that are used for the simulation
    """
    training_s_a_pairs_idxs = [i for i in range(len(s_a_pairs_time_stamps)) if not s_a_pairs_time_stamps[i] in simulation_time_stamps]
    training_s_a_pairs = state_action_pairs[training_s_a_pairs_idxs]
    training_resulting_states = resulting_states[training_s_a_pairs_idxs]

    return training_s_a_pairs, training_resulting_states


def get_z_scores_with_means_and_stds(data):
    """Performs z-score normalization (also called standard score) on the input data
       along the rows and returns the normalized data with the means and standard deviations
       of every column.

    Keyword arguments:
    data -- data to be normalized
    """
    column_means = np.mean(data, axis=0)
    column_stds = np.std(data, axis=0)
    data = data - column_means.reshape((1, data.shape[1])).repeat(data.shape[0], axis=0)
    normalized_data = data / column_stds.reshape((1, data.shape[1])).repeat(data.shape[0], axis=0)

    return normalized_data, column_means, column_stds


def normalize_with_given_means_and_stds(data, means, stds):
    """Performs a z-score normalization on the given data with the given means and
       standard deviations.
    
    Keyword arguments:
    data -- 2D matrix (rows: samples, columns: features)
    means -- vector containing the mean value of each feature
    stds -- vector containint the standard deviations of each feature
    """
    data = data - means.reshape((1, data.shape[1])).repeat(data.shape[0], axis=0)
    normalized_data = data / stds.reshape((1, data.shape[1])).repeat(data.shape[0], axis=0)

    return normalized_data


def get_energy_system_data(file_path):
    """Reads columns in an excel file containing data from
       the energy system and returns them in a dictionary.

    Keyword arguments:
    file_path -- file path to the excel file
    """
    table = pd.read_excel(file_path, skiprows=2)

    discharge = table[" Entladung(W)"].values
    charge = table[" Ladung(W)"].values
    generation = table[" Erzeugung(W)"].values
    load = table[" Verbrauch(W)"].values
    state_of_charge = table[" Ladestand/SoC(%)"].values
    time_stamps = table[" Datum/Zeit"].values # strings in the format of "dd.mm.yyyy hh:mm:ss"

    pt.test_same_array_length([discharge, charge, generation, load, state_of_charge, time_stamps])
    charge_rate = md.merge_charge_discharge_array(charge, discharge)
    time_stamps = remove_seconds_from_time_stamps(time_stamps)

    energy_system_data = {'charge_rate_in_W': charge_rate,
                          'generation_in_W': generation,
                          'load_in_W': load,
                          'SoC': state_of_charge,
                          'time_stamps': time_stamps}

    return energy_system_data


def remove_seconds_from_time_stamps(time_stamps):
    """
    Removes the last three elements in time_stamps of the format "dd.mm.yyyy hh:mm:ss" 
    that encode the seconds. The output format is "dd.mm.yyyy hh:mm".

    string_time_stamps -- array with strings encoding time in 
                          the format "dd.mm.yyyy hh:mm:ss"
    """
    map_object = map(lambda x: x[:-3], time_stamps)
    array_of_mapped_values = np.asarray(list(map_object))

    return array_of_mapped_values


def get_weather_data(file_path):
    """Reads columns in an excel file containing
       weather data and returns them in a dictionary.

    Keyword arguments:
    file_path -- file path to the excel file
    """
    table = pd.read_excel(file_path)

    year = table["Year"].values
    month = table["Month"].values
    day = table["Day"].values
    hour = table["Hour"].values
    minute = table["Minute"].values
    global_radiation_in_W_per_square_meter = table["GlobalRadiation"].values
    diffuse_radiation_in_W_per_square_meter = table["diffuseRadiation"].values

    # convert time stamps to strings in the format of "dd.mm.yyyy hh:mm"
    time_stamps = convert_time_stamp_numbers_to_string_format(day, month, year, hour, minute)

    pt.test_time_stamp_string_format(time_stamps)
    pt.test_same_array_length([year, month, day, hour, minute,
                               global_radiation_in_W_per_square_meter,
                               diffuse_radiation_in_W_per_square_meter])

    weather_data = {'global_radiation_in_W_per_square_meter': global_radiation_in_W_per_square_meter,
                    'diffuse_radiation_in_W_per_square_meter': diffuse_radiation_in_W_per_square_meter,
                    'time_stamps': time_stamps}

    return weather_data


def replace_missing_data_with_interpolations(data_with_time_stamps):
    """Returns the data with their time stamps, where missing data points between data points
       that are at most 10 minute apart are interpolated. A missing data point is not replaced,
       if one or more of the data points needed for the interpolation is missing.

    Keyword arguments:
    data_with_time_stamps -- a list of lists, where each list in this list is a data point of merged data (from the energy
                             system and the weather) that have the same time stamp
                             Structure of returned list of lists:
                             --> dim 1: data points consisting of data from both datasets
                             --> dim 2: numerical values of a data point, with a time stamp as a string as the last element
    """
    num_variables = len(data_with_time_stamps[0]) - 1
    time_stamp_strings = [data_point[num_variables] for data_point in data_with_time_stamps]
    data_without_time_stamp_strings = np.asarray([data_point[:-1] for data_point in data_with_time_stamps], dtype=np.float64)
    interpolated_data_without_time_stamps = np.copy(data_without_time_stamp_strings)
    interpolated_data_time_stamps = time_stamp_strings.copy()
    num_interpolations_so_far = 0

    for i in range(len(data_with_time_stamps)-1):
        next_min_time_stamp = get_corresponding_time_stamp_string(time_stamp_strings[i], 1)
        if not time_stamp_strings[i+1] == next_min_time_stamp:
            time_stamps_of_next_10_min = [get_corresponding_time_stamp_string(time_stamp_strings[i], j) for j in range(2,11)]
            if time_stamp_strings[i+1] in time_stamps_of_next_10_min:
                num_missing_points = time_stamps_of_next_10_min.index(time_stamp_strings[i+1]) + 1
                value_difference = data_without_time_stamp_strings[i+1] - data_without_time_stamp_strings[i]
                for k in range(1, num_missing_points + 1):
                    interpolated_point = data_without_time_stamp_strings[i] + k * (value_difference / (num_missing_points + 1))
                    insert_idx = i + 1 + num_interpolations_so_far
                    interpolated_data_without_time_stamps = np.insert(interpolated_data_without_time_stamps, insert_idx, interpolated_point, axis=0)
                    interpolated_data_time_stamps.insert(insert_idx, get_corresponding_time_stamp_string(time_stamp_strings[i], k))
                    num_interpolations_so_far += 1

    interpolated_data_with_time_stamps =  [x.tolist() + [interpolated_data_time_stamps[i]] 
                                           for i, x in enumerate(interpolated_data_without_time_stamps)]

    return interpolated_data_with_time_stamps


def get_state_transitions_as_separate_arrays(merged_data_with_time_stamps, temporal_resolution_in_mins, state_action_pair_feature_order):
    """Returns two 2D-numpy arrays (one containing state-action pairs and the other containing
       the states resulting from them after the specified time) and a list of time stamps of each state-action pair in the dataset.
       The rows in each of the returned arrays correspond to state-action pairs / states and the columns correspond to the values of
       the variables in each state-action pair / state.

    Keyword arguments:
    merged_data_with_time_stamps -- a list of lists, where each list in this list is a data point of merged data (from the energy
                                    system and the weather) that have the same time stamp
                                    Structure of returned list of lists:
                                    --> dim 1: data points consisting of data from both datasets
                                    --> dim 2: numerical values of a data point, with a time stamp as a string as the last element
    temporal_resolution_in_mins -- the desired temporal resolution of the dataset that will be returned (i.e. the number of minutes
                                   between each state-action pair and its corresponding state in the other returned array)
    state_action_pair_feature_order -- list of the features contained in each state-action pair apart from the time stamp strings in the
                                       order they were added    
    """
    num_state_action_variables = len(merged_data_with_time_stamps[0]) - 1
    time_stamp_strings = [s_a_pair[num_state_action_variables] for s_a_pair in merged_data_with_time_stamps]
    s_a_pairs_without_time_stamp_strings = [s_a_pair[:-1] for s_a_pair in merged_data_with_time_stamps]
    action_idx_in_s_a_pair = state_action_pair_feature_order.index('charge_rate_in_W')
    first_correspondence_already_found = False
    last_search_for_corresponding_state_failed = False
    state_transition_time_stamps = []
    state_action_pairs = []
    resulting_states = []

    i = 0
    last_corresponding_state_action_pair = merged_data_with_time_stamps[0]

    while True:
        if last_search_for_corresponding_state_failed:
            s_a_pair_time_stamp_string = time_stamp_of_corresponding_s_a_pair
        else:
            s_a_pair_time_stamp_string = last_corresponding_state_action_pair[-1]
        time_stamp_of_corresponding_s_a_pair = get_corresponding_time_stamp_string(s_a_pair_time_stamp_string,
                                                                                   temporal_resolution_in_mins)
        if is_later_time_stamp(time_stamp_of_corresponding_s_a_pair, time_stamp_strings[-1]):
            break

        try:
            corresponding_state_idx = time_stamp_strings.index(time_stamp_of_corresponding_s_a_pair)
        except ValueError: # ValueError is raised when no matching time stamp was found
            i += 1
            if not first_correspondence_already_found:
                last_corresponding_state_action_pair = merged_data_with_time_stamps[i]
            else:
                last_search_for_corresponding_state_failed = True
            continue

        corresponding_s_a_pair = s_a_pairs_without_time_stamp_strings[corresponding_state_idx]
        corresponding_state = [x for i, x in enumerate(corresponding_s_a_pair) if i != action_idx_in_s_a_pair]

        if not last_search_for_corresponding_state_failed:
            state_action_pairs.append(last_corresponding_state_action_pair[:-1])
            resulting_states.append(corresponding_state)
            state_transition_time_stamps.append(s_a_pair_time_stamp_string)

        last_corresponding_state_action_pair = merged_data_with_time_stamps[corresponding_state_idx]
        last_search_for_corresponding_state_failed = False
        first_correspondence_already_found = True

    if not first_correspondence_already_found:
        raise ValueError("No state-action pair with a corresponding state according to the specified temporal resolution was found")

    state_action_pairs = np.asarray(state_action_pairs)
    resulting_states = np.asarray(resulting_states)

    return state_action_pairs, resulting_states, state_transition_time_stamps


def find_idxs_of_n_longest_state_sequences(states, time_stamps, n, temporal_resolution_in_mins):
    """Returns the indices of the states in the n longest uninterrupted state sequences in 
       a list of lists (one state sequence per list in the list of lists).

    Keyword arguments:
    states -- 2D array with states from a recorded dataset (rows: states, columns: features)
    time_stamps -- list of time stamps of the of the states. Time stamp format: "dd.mm.yyyy hh:mm"
    n -- how many of the longest state sequences should be returned (n=1 --> the longest is returned, 
         n=2 --> the two longest are returned and so on)
    temporal_resolution_in_mins -- how many minutes have to be between each subsequent state in a state sequence
    """
    subsequent_states_idxs = get_idxs_of_state_sequences(time_stamps, temporal_resolution_in_mins)
    num_subsequent_states = [len(x) for x in subsequent_states_idxs]
    state_idxs_in_longest_sequence = []

    for _ in range(n):
        num_states_in_longest_sequence = max(num_subsequent_states)
        longest_sequence_idx = num_subsequent_states.index(num_states_in_longest_sequence)
        state_idxs_in_longest_sequence.append(subsequent_states_idxs[longest_sequence_idx])
        del num_subsequent_states[longest_sequence_idx]
        del subsequent_states_idxs[longest_sequence_idx]

    return state_idxs_in_longest_sequence


def get_idxs_of_state_sequences(time_stamps, temporal_resolution_in_mins):
    """Returns a list of lists, where each list contains the indices of an uninterrupted state sequence. 

    Keyword arguments:
    time_stamps -- list of time stamps of the of the states. Time stamp format: "dd.mm.yyyy hh:mm"
    temporal_resolution_in_mins -- how many minutes have to be between each subsequent state in a state sequence
    """
    subsequent_states_idxs = []
    num_uninterrupted_sequences_of_states = -1
    next_state_found = False

    for i in range(len(time_stamps)-1):
        next_state_time_stamp = get_corresponding_time_stamp_string(time_stamps[i], temporal_resolution_in_mins)
        if time_stamps[i+1] == next_state_time_stamp:
            if next_state_found == False:
                subsequent_states_idxs.append([])
                num_uninterrupted_sequences_of_states += 1
            subsequent_states_idxs[num_uninterrupted_sequences_of_states].append(i)
            if i == len(time_stamps) - 2:
                subsequent_states_idxs[num_uninterrupted_sequences_of_states].append(i+1)
            next_state_found = True
        else:
            if next_state_found:
                subsequent_states_idxs[num_uninterrupted_sequences_of_states].append(i)
            next_state_found = False

    return subsequent_states_idxs


def get_corresponding_time_stamp_string(time_stamp_string, period_between_corr_time_stamps_in_mins):
    """Returns a time stamp string corresponding to the time that is a specified number of minutes after the
       input time stamp. The format of the returned time stamp string is "dd.mm.yyyy hh:mm".

    Keyword arguments:
    time_stamp_string -- time stamp string in the format of "dd.mm.yyyy hh:mm"
    period_between_corr_time_stamps_in_mins -- the period between corresponding time stamps in minutes
    """
    if period_between_corr_time_stamps_in_mins > 24 * 60:
        raise ValueError("Period between time stamps should be 24h at most")

    year = int(time_stamp_string[6:10])
    month = int(time_stamp_string[3:5])
    day = int(time_stamp_string[:2])
    hour = int(time_stamp_string[11:13])
    minute = int(time_stamp_string[-2:])

    hours_to_add = int(np.floor(period_between_corr_time_stamps_in_mins / 60.0))
    mins_to_add = period_between_corr_time_stamps_in_mins % 60
    minute += mins_to_add
    hour += hours_to_add

    if minute > 59:
        minute = minute % 60
        hour += 1
    if hour > 23:
        hour = hour % 24
        if md.is_from_last_day_of_the_month(time_stamp_string):
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
        else:
            day += 1

    corresponding_time_stamp_string = convert_time_stamp_numbers_to_string_format(day, month, year, hour, minute)

    return corresponding_time_stamp_string


def convert_time_stamp_numbers_to_string_format(day, month, year, hour, minute):
    """Converts the given numbers for the date and time to the format "dd.mm.yyyy hh:mm".
       The function accepts integers and numpy arrays as inputs.

    Keyword arguments:
    day -- value(s) encoding the day of the month
    month -- value(s) encoding the month of the year
    year -- value(s) encoding the year
    hour -- value(s) encoding the hour of the day (ranging from 0 to 23)
    minute -- value(s) encoding the minute of the hour
    """
    if (isinstance(day, int) and isinstance(month, int) and isinstance(year, int) 
        and isinstance(hour, int) and isinstance(minute, int)):
        time_string = ""
        time_string += convert_number_to_two_digit_string(day) + "."
        time_string += convert_number_to_two_digit_string(month) + "."
        time_string += str(year) + " "
        time_string += convert_number_to_two_digit_string(hour) + ":"
        time_string += convert_number_to_two_digit_string(minute)
        return time_string

    elif (isinstance(day, np.ndarray) and isinstance(month, np.ndarray) and isinstance(year, np.ndarray) 
          and isinstance(hour, np.ndarray) and isinstance(minute, np.ndarray)):
        time_stamp_strings = np.empty(day.shape[0], dtype=object)
        for i in range(len(day)):
            time_string = ""
            time_string += convert_number_to_two_digit_string(day[i]) + "."
            time_string += convert_number_to_two_digit_string(month[i]) + "."
            time_string += str(year[i]) + " "
            time_string += convert_number_to_two_digit_string(hour[i]) + ":"
            time_string += convert_number_to_two_digit_string(minute[i])
            time_stamp_strings[i] = time_string
        return time_stamp_strings

    else:
        raise ValueError("Unexpected input type")


def convert_number_to_two_digit_string(number):
    """Turns the input number (integer consisting of one or two digits) into a 
       string that is always of length 2. Two-digit numbers are simply converted to strings,
       one-digit numbers are converted to strings where the first character is '0'.

    Keyword arguments:
    number -- one or two-digit number
    """
    number_as_string = str(number)

    if len(number_as_string) == 1:
        return '0' + number_as_string
    elif len(number_as_string) == 2:
        return number_as_string
    else:
        raise ValueError("String does not have length 1 or 2")


def is_later_time_stamp(time_stamp_2, time_stamp_1):
    """Returns a Boolean value indicating whether time_stamp_2 encodes a time and date 
       that is after that of time_stamp_1. Expected format of the time stamps is "dd.mm.yyyy hh:mm".

    Keyword arguments:
    time_stamp_1 -- first time stamp
    time_stamp_2 -- second time stamp, which is after the first one if the returned 
                    Boolean value is 'True'
    """
    year_ts1 = int(time_stamp_1[6:10])
    month_ts1 = int(time_stamp_1[3:5])
    day_ts1 = int(time_stamp_1[:2])
    hour_ts1 = int(time_stamp_1[11:13])
    minute_ts1 = int(time_stamp_1[-2:])
    year_ts2 = int(time_stamp_2[6:10])
    month_ts2 = int(time_stamp_2[3:5])
    day_ts2 = int(time_stamp_2[:2])
    hour_ts2 = int(time_stamp_2[11:13])
    minute_ts2 = int(time_stamp_2[-2:])

    years_match = (year_ts1 == year_ts2)
    months_match = (month_ts1 == month_ts2)
    days_match = (day_ts1 == day_ts2)
    hours_match = (hour_ts1 == hour_ts2)
    mins_match = (minute_ts1 == minute_ts2)

    if not years_match:
        if year_ts2 > year_ts1: return True
        else: return False
    elif not months_match:
        if month_ts2 > month_ts1: return True
        else: return False
    elif not days_match:
        if day_ts2 > day_ts1: return True
        else: return False
    elif not hours_match:
        if hour_ts2 > hour_ts1: return True
        else: return False
    elif not mins_match:
        if minute_ts2 > minute_ts1: return True
        else: return False
    else:
        return False


def convert_time_string_to_minute_of_day(string_time_stamps):
    """Converts an array of strings saying the time of day in the format 
       of "dd.mm.yyyy hh:mm" into a number saying what minute of the day 
       it is (ranging from 0 to 1440) with 0 being midnight and 1440 being 11:59 p.m.

    Keyword arguments:
    string_time_stamps -- array with strings encoding time in 
                          the format "dd.mm.yyyy hh:mm"
    """
    time_stamps_as_minute_of_day = np.empty(string_time_stamps.shape[0], dtype=int)
    hour = encode_hour_as_int(string_time_stamps)
    minute = encode_minute_as_int(string_time_stamps)

    for i in range(len(string_time_stamps)):
        time_stamps_as_minute_of_day[i] = hour[i] * 60 + minute[i]

    return time_stamps_as_minute_of_day


def encode_hour_as_int(string_time_stamps):
    """Returns the hour of a time encoded as a string as an int.

    Keyword arguments:
    string_time_stamps -- array with strings encoding time
    """
    hours_as_int = np.empty(string_time_stamps.shape[0], dtype=int)
    time_stamps_without_date = remove_date_strings(string_time_stamps)

    for i in range(len(time_stamps_without_date)):
        hours_as_int[i] = int(time_stamps_without_date[i][:2])

    pt.test_hour_values(hours_as_int)

    return hours_as_int


def encode_minute_as_int(string_time_stamps):
    """Returns the minute of a time encoded as a string as an int.

    Keyword arguments:
    string_time_stamps -- array with strings encoding time
    """
    minutes_as_int = np.empty(string_time_stamps.shape[0], dtype=int)
    time_stamps_without_date = remove_date_strings(string_time_stamps)

    for i in range(len(string_time_stamps)):
        minutes_as_int[i] = int(time_stamps_without_date[i][3:5])

    pt.test_minute_values(minutes_as_int)

    return minutes_as_int


def remove_date_strings(time_stamps_with_date):
    """Removes the date in the string of every element in the input array.

    Keyword arguments:
    string_time_stamps -- array with strings encoding time
    """
    cutoff_index = 11
    time_stamps_without_date = np.empty(time_stamps_with_date.shape, dtype=object)

    for i in range(len(time_stamps_with_date)):
        time_stamps_without_date[i] = time_stamps_with_date[i][cutoff_index:]

    return time_stamps_without_date
