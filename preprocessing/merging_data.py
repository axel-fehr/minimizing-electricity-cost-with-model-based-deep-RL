"""This file contains functions to merge data of different kinds during preprocessing."""

import numpy as np
from preprocessing import preprocessing_tests as pt

def merge_charge_discharge_array(charge, discharge):
    """Puts the charge and discharge values in one array.

    Keyword arguments:
    charge -- array with the charge rate per time step
    discharge -- array with the discharge rate per time step
    """
    pt.test_for_abnormal_charge_discharge_values(charge, discharge)

    merged_array = np.zeros(charge.shape)

    for i in range(len(charge)):
        if charge[i] > 0:
            merged_array[i] = charge[i]
        elif discharge[i] > 0:
            merged_array[i] = -discharge[i]

    return merged_array


def merge_data_with_same_time_stamp(energy_system_data, weather_data):
    """Merges the data from the energy system and the weather data and returns the result.

    Keyword arguments:
    energy_system_data -- dictionary with energy system data
    weather_data -- dictionary with weather data
    """
    pt.test_for_strictly_ascending_time_stamps(energy_system_data['time_stamps'])
    pt.test_for_strictly_ascending_time_stamps(weather_data['time_stamps'])
    overlap_idxs = get_start_and_end_indices_of_overlaps(energy_system_data, weather_data)
    overlap_idxs = remove_idxs_of_single_element_overlaps(overlap_idxs)
    merged_data_with_time_stamps, state_action_pair_feature_order = merge_energy_system_and_weather_data(energy_system_data,
                                                                                                         weather_data,
                                                                                                         overlap_idxs)

    return merged_data_with_time_stamps, state_action_pair_feature_order


def get_start_and_end_indices_of_overlaps(energy_system_data, weather_data):
    """Returns the indices that mark the beginning and end of the parts that
       habe values with the same time stamp. Two sets of indices are returned.
       One for the energy system data and one for the weather data.

    Keyword arguments:
    energy_system_data -- dictionary with energy system data
    weather_data -- dictionary with weather data
    """
    overlap_beginning_idxs = find_overlap_beginnings(energy_system_data, weather_data)
    overlap_end_idxs = find_overlap_ends(energy_system_data, weather_data, overlap_beginning_idxs)
    overlap_idxs = merge_idxs_into_single_array(overlap_beginning_idxs, overlap_end_idxs)

    return overlap_idxs


def remove_idxs_of_single_element_overlaps(overlap_idxs):
    """Removes the indices of overlaps that contain only one single element / time stamp.

    Keyword arguments:
    overlap_idxs -- 3-dimensional array containing start and end indices of the overlaps
                    --> #rows: number of overlaps
                    --> #columns: number of datasets (1st: energy system dataset, 2nd: weather dataset)
                    --> #3rd dim: beginning and end of the overlaps (1st: beginning, 2nd: end)
    """
    deletion_idxs = []

    for i, idx_tuples in enumerate(overlap_idxs):
        single_element_overlap_in_first_dataset = (idx_tuples[0,0] == idx_tuples[0,1])
        single_element_overlap_in_second_dataset = (idx_tuples[1,0] == idx_tuples[1,1])

        if single_element_overlap_in_first_dataset and single_element_overlap_in_second_dataset:
            deletion_idxs.append(i)
    
    no_overlaps_with_multiple_elements = (len(deletion_idxs) == overlap_idxs.shape[0])
    if no_overlaps_with_multiple_elements:
        raise ValueError("No overlaps with multiple elements and therefore no training data")

    return np.delete(overlap_idxs, deletion_idxs, axis=0)


def merge_energy_system_and_weather_data(energy_system_data, weather_data, overlap_idxs):
    """Returns a list with the names of the features that are added to the merged data in the order they are added,
       and also returns a list of lists, where each list in this list is a data point of merged data (from the energy
       system and the weather) that have the same time stamp.

       Structure of returned list of lists:
        --> dim 1: data points consisting of data from both datasets
        --> dim 2: numerical values in each data point time stamp as a string (as the last element)

    Keyword arguments:
    energy_system_data -- dictionary with energy system data
    weather_data -- dictionary with weather data
    overlap_idxs -- indices indicating the beginning and end of parts with the same time stamps (i.e. overlaps).
                    Structure:
                    --> #rows: number of overlaps
                    --> #columns: number of datasets (1st: energy system dataset, 2nd: weather dataset)
                    --> #3rd dim: beginning and end of the overlaps (1st: beginning, 2nd: end)
    """
    pt.test_for_unequal_overlap_lengths(overlap_idxs)
    energy_system_data_keys = [k for k in list(energy_system_data.keys()) if k != 'time_stamps']
    weather_data_keys = [k for k in list(weather_data.keys()) if k != 'time_stamps']
    data_keys = energy_system_data_keys + weather_data_keys + ['time_stamps']
    merged_data_with_time_stamps = []

    for i in range(overlap_idxs.shape[0]):
        length_of_overlap = overlap_idxs[i,0,1] - overlap_idxs[i,0,0] + 1
        en_sys_data_overlap_start = overlap_idxs[i,0,0]
        wthr_data_overlap_start = overlap_idxs[i,1,0]

        for j in range(length_of_overlap):
            merged_data_with_time_stamps.append([])
            for k in energy_system_data_keys:
                merged_data_with_time_stamps[-1].append(energy_system_data[k][en_sys_data_overlap_start+j])
            for k in weather_data_keys:
                merged_data_with_time_stamps[-1].append(weather_data[k][wthr_data_overlap_start+j])

            wthr_time_stamp = weather_data['time_stamps'][wthr_data_overlap_start+j]
            es_sys_time_stamp = energy_system_data['time_stamps'][en_sys_data_overlap_start+j]
            if wthr_time_stamp != es_sys_time_stamp:
                raise ValueError('Found elements in overlap that do not have the same time stamp')  

            time_stamp_string_of_overlap = weather_data['time_stamps'][wthr_data_overlap_start+j]
            time_stamp_as_integer = time_stamp_string_to_integer(time_stamp_string_of_overlap)
            merged_data_with_time_stamps[-1].append(time_stamp_as_integer)
            merged_data_with_time_stamps[-1].append(time_stamp_string_of_overlap)

    pt.test_for_strictly_ascending_time_stamps([x[-1] for x in merged_data_with_time_stamps])

    return merged_data_with_time_stamps, data_keys


def time_stamp_string_to_integer(time_stamp_string):
    """Converts the time stamps encoded as strings in the format of "dd.mm.yyyy hh:mm"
       to integers that encode the corresponding minute of the day. These integers range
       from 1 to 1440, where 1 corresponds to midnight and 1440 to 11:59 p.m.

    Keyword arguments:
    time_stamp_string -- a time stamp string in the format of "dd.mm.yyyy hh:mm"
    """
    pt.test_time_stamp_string_format([time_stamp_string])

    hour_of_day_as_int = int(time_stamp_string[11:13])
    minute_of_hour_as_int = int(time_stamp_string[-2:]) + 1
    minutes_per_hour = 60
    minute_of_day_as_int = hour_of_day_as_int * minutes_per_hour + minute_of_hour_as_int

    return minute_of_day_as_int


def find_overlap_beginnings(energy_system_data, weather_data):
    """Returns the indices of the first elements in all the parts the two data sets (energy system data 
       and weather data) that have the same time stamps as a numpy array. The number of rows 
       is equal to the numer of overlapping parts that were found. The first column contains the indices 
       corresponding to the first input dictionary and the second column contains the indices corresponding to
       the second input dictionary.

    Keyword arguments:
    energy_system_data -- dictionary with energy system data
    weather_data -- dictionary with weather data
    """
    already_found_first_overlap = False
    is_beginning_of_new_overlap = False
    overlap_beginnings = np.array([[]])
    previous_es_time_stamp = ""
    weather_dat = weather_data["time_stamps"].tolist()

    # TODO: could be sped up by jumping to the next day / month / year etc. if the day / month / year etc. doesn't match
    for es_time_stamp_idx, current_es_time_stamp in enumerate(energy_system_data["time_stamps"]):
        try:
            idx_of_matching_wd_time_stamp = weather_dat.index(current_es_time_stamp)
        except ValueError: # ValueError is raised when no matching time stamp was found
            is_beginning_of_new_overlap = True
            previous_es_time_stamp = current_es_time_stamp
            continue
        if not already_found_first_overlap:
            already_found_first_overlap = True
            overlap_beginnings = np.array([[es_time_stamp_idx, idx_of_matching_wd_time_stamp]])
            is_beginning_of_new_overlap = False
        else:
            if is_beginning_of_new_overlap or not are_subsequent_time_stamps(previous_es_time_stamp, current_es_time_stamp):
                next_tuple_of_overlap_beginnings = np.array([[es_time_stamp_idx, idx_of_matching_wd_time_stamp]])
                overlap_beginnings = np.concatenate((overlap_beginnings, next_tuple_of_overlap_beginnings), axis=0)
                is_beginning_of_new_overlap = False
            else:
                previous_es_time_stamp = current_es_time_stamp
                continue
        previous_es_time_stamp = current_es_time_stamp

    if already_found_first_overlap:
        return overlap_beginnings
    else:
        raise ValueError("Could not find elements in the two datasets with the same time stamps")


def find_overlap_ends(energy_system_data, weather_data, overlap_beginning_idxs):
    """Returns the indices of the last elements in all the parts of the two data sets (energy system data 
       and weather data) that have the same time stamps as a numpy array. The number of rows 
       is equal to the numer of overlapping parts. The first column contains the indices 
       corresponding to the first input dictionary and the second column contains the indices corresponding to
       the second input dictionary.

    Keyword arguments:
    energy_system_data -- dictionary with energy system data
    weather_data -- dictionary with weather data
    overlap_beginning_idxs -- numpy array with indices indicating the beginnings of each 
                              part where the time stamps in the two dictionaries are the same.
                              Each row contains a tuple of indices, where the first corresponds 
                              to the first dictionary (energy system data) and the second to the 
                              second dictionary (weather data).
    """
    overlap_ends = np.array([[]])
    overlap_beginning_idxs_local_copy = np.copy(overlap_beginning_idxs)

    for _, overlap_beginning in enumerate(overlap_beginning_idxs_local_copy):
        overlap_end = overlap_beginning
        while True:
            same_time_stamp = (energy_system_data['time_stamps'][overlap_end[0]]
                               == weather_data['time_stamps'][overlap_end[1]])
            reached_index_boundary = ((overlap_end[0] == len(energy_system_data["time_stamps"]) - 1)
                                       or overlap_end[1] == len(weather_data["time_stamps"]) - 1)
            if not same_time_stamp:
                overlap_ends = create_array_or_concatenate(overlap_ends, overlap_end - 1)
                break
            elif reached_index_boundary:
                overlap_ends = create_array_or_concatenate(overlap_ends, overlap_end)
                break
            else:
                overlap_end += 1

    return overlap_ends


def merge_idxs_into_single_array(overlap_beginning_idxs, overlap_end_idxs):
    """Merges the indices indicating the beginning and end of parts with the
       same time stamps (i.e. overlaps) and returns the result.

       Structure of returned array:
       --> #rows: number of overlaps
       --> #columns: number of datasets (1st: energy system dataset, 2nd: weather dataset)
       --> #3rd dim: beginning and end of the overlaps (1st: beginning, 2nd: end)

    Keyword arguments:
    overlap_beginning_idxs -- two-dimensional array with indices indicating the
                              overlap beginnings (#rows: #overlaps, #cols: #datasets)
    overlap_end_idxs -- two-dimensional array with indices indicating the
                        overlap ends (#rows: #overlaps, #cols: #datasets)
    """
    overlap_beginning_idxs = overlap_beginning_idxs.reshape((overlap_beginning_idxs.shape[0],
                                                             overlap_beginning_idxs.shape[1],
                                                             1))
    overlap_end_idxs = overlap_end_idxs.reshape((overlap_end_idxs.shape[0],
                                                 overlap_end_idxs.shape[1],
                                                 1))

    return np.concatenate((overlap_beginning_idxs, overlap_end_idxs), axis=2)


def are_subsequent_time_stamps(previous_time_stamp, current_time_stamp):
    """Returns a Boolean value indicating whether the input time stamps are only 1 minute apart,
       meaning that are no missing measurements between them. 
       Expected format of the time stamps: "dd.mm.yyyy hh:mm".

    Keyword arguments:
    previous_time_stamp -- the previous time stamp (must be the earlier one)
    current_time_stamp -- the current time stamp
    """
    current_time_stamp_date = current_time_stamp[:10]
    previous_time_stamp_date = previous_time_stamp[:10]
    dates_match = (current_time_stamp_date == previous_time_stamp_date)
    previous_time_stamp_is_at_11_59_pm = (previous_time_stamp[-5:] == "23:59")

    if dates_match:
        return are_subsequent_time_stamps_on_the_same_day(previous_time_stamp, current_time_stamp)
    elif previous_time_stamp_is_at_11_59_pm: 
        return are_subsequent_time_stamps_on_different_days(previous_time_stamp, current_time_stamp)
    else:
        return False


def create_array_or_concatenate(array, row_to_concatenate):
    """Concatenates "row_to_concatenate" to "array" if "array" already has
       initialized values and returns the result. Otherwise, "row_to_concatenate"
       is returned as an array with one row.

    Keyword arguments:
    array -- array the row has to be concatenated with (potentially empty)
    row_to_concatenate -- row to be concatenated with the array
    """
    if array.shape[1] == 0:
        return np.array([row_to_concatenate])
    else:
        return np.concatenate((array, np.array([row_to_concatenate])), axis=0)


def are_subsequent_time_stamps_on_the_same_day(previous_time_stamp, current_time_stamp):
    """Returns a Boolean value indicating whether the two given time stamps, which
       must be from the same day are only 1 minute apart.

    Keyword arguments:
    previous_time_stamp -- the previous time stamp (must be the earlier one)
    current_time_stamp -- the current time stamp
    """
    current_time_stamp_hour = int(current_time_stamp[11:13])
    current_time_stamp_min = int(current_time_stamp[-2:])
    previous_time_stamp_hour = int(previous_time_stamp[11:13])
    previous_time_stamp_min = int(previous_time_stamp[-2:])

    if previous_time_stamp_min == 59:
        current_hour_is_next_hour = (current_time_stamp_hour == previous_time_stamp_hour + 1)
        if current_hour_is_next_hour and current_time_stamp_min == 0:
            return True
        return False
    else:
        current_min_is_next_min = (current_time_stamp_min == previous_time_stamp_min + 1)
        hours_match = (current_time_stamp_hour == previous_time_stamp_hour)
        if hours_match and current_min_is_next_min:
            return True
        return False


def are_subsequent_time_stamps_on_different_days(previous_time_stamp, current_time_stamp):
    """Returns a Boolean value indicating whether the the previous time stamp, whose minute 
       must be the last of the day (time is 23:59), is only 1 minute apart from the current time stamp.

    Keyword arguments:
    previous_time_stamp -- the previous time stamp (must be the earlier one)
    current_time_stamp -- the current time stamp
    """
    current_time_stamp_year = int(current_time_stamp[6:10])
    current_time_stamp_month = int(current_time_stamp[3:5])
    current_time_stamp_day = int(current_time_stamp[:2])
    previous_time_stamp_year = int(previous_time_stamp[6:10])
    previous_time_stamp_month = int(previous_time_stamp[3:5])
    previous_time_stamp_day = int(previous_time_stamp[:2])

    years_match = (current_time_stamp_year == previous_time_stamp_year)
    months_match = (current_time_stamp_month == previous_time_stamp_month)
    current_time_is_midnight_and_next_min = (previous_time_stamp[-5:] == "23:59"
                                             and current_time_stamp[-5:] == "00:00")

    if is_from_last_day_of_the_month(previous_time_stamp):
        if years_match:
            current_month_is_next_month = (current_time_stamp_month == previous_time_stamp_month + 1)
            if current_month_is_next_month and current_time_stamp_day == 1 and current_time_is_midnight_and_next_min:
                return True
        else:
            current_year_is_next_year = (current_time_stamp_year == previous_time_stamp_year + 1)
            current_month_is_next_month = (previous_time_stamp_month == 12 and current_time_stamp_month == 1)
            if current_year_is_next_year and current_month_is_next_month and current_time_stamp_day == 1 and current_time_is_midnight_and_next_min:
                return True
    else:
        current_day_is_next_day = (current_time_stamp_day == previous_time_stamp_day + 1)
        if years_match and months_match and current_day_is_next_day and current_time_is_midnight_and_next_min:
            return True

    return False


def is_from_last_day_of_the_month(time_stamp):
    """Returns a Boolean value indicating whether the day in the date included in the given
       time stamp is the last day of its month.

    Keyword arguments:
    time_stamp -- time stamp expected to be a string of the format "dd.mm.yyyy hh:mm"
    """
    num_days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_as_int = int(time_stamp[:2])
    month_as_int = int(time_stamp[3:5])
    year_as_int = int(time_stamp[6:10])
    year_is_leap_year = (year_as_int % 4 == 0) # does not work for all leap years but for all leap years the data could contain
    month_is_february = (month_as_int == 2)

    if year_is_leap_year and month_is_february:
        if day_as_int == 29:
            return True
        return False
    elif day_as_int == num_days_in_month[month_as_int-1]:
        return True
    else:
        return False
