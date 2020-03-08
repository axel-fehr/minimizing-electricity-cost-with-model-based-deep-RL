"""This file contains functions that test the preprocessed data."""

import numpy as np
from preprocessing import preprocessing as pp

def test_same_array_length(list_of_arrays):
    """Throws an error if the arrays in the input list are not of the same length.

    Keyword arguments:
    list_of_arrays -- input list of arrays
    """
    if not all_same_length(list_of_arrays):
        raise ValueError("Array lengths not the same!")


def test_time_stamp_string_format(time_stamp_strings):
    """Tests whether the time stamp strings in the input
       array all have the format "dd.mm.yy hh:mm".

    Keyword arguments:
    time_stamp_strings -- array containing time stamp strings
    """
    test_time_stamp_string_lengths(time_stamp_strings)
    test_colon_positions(time_stamp_strings)
    test_period_positions(time_stamp_strings)
    test_for_abnormal_time_stamp_values(time_stamp_strings)


def test_for_abnormal_charge_discharge_values(charge, discharge):
    """Performs tests on the charge and discharge values.

    Keyword arguments:
    charge -- array with the charge rate per time step
    discharge -- array with the discharge rate per time step
    """
    test_for_negative_values([charge, discharge])
    test_for_simultaneous_charge_and_discharge(charge, discharge)


def test_for_unequal_overlap_lengths(overlap_idxs):
    """Tests if there are any overlaps where the corresponding indices indicating the start and the end of it in
       each dataset do not have the same distance from each other.

    Keyword arguments:
    overlap_idxs -- 3-dimensional array containing start and end indices of the overlaps
                    --> #rows: number of overlaps
                    --> #columns: number of datasets (1st: energy system dataset, 2nd: weather dataset)
                    --> #3rd dim: beginning and end of the overlaps (1st: beginning, 2nd: end)
    """
    for overlap in overlap_idxs:
        energy_system_data_overlap_length = overlap[0,1] - overlap[0,0] + 1
        weather_data_overlap_length = overlap[0,1] - overlap[0,0] + 1
        if energy_system_data_overlap_length != weather_data_overlap_length:
            raise ValueError("Size of the current overlap in both datasets is not the same")


def all_same_length(list_of_arrays):
    """Checks if the arrays in the input list are of the same length.

    Keyword arguments:
    list_of_arrays -- input list of arrays
    """
    arr_lengths = []

    for array in list_of_arrays:
        arr_lengths.append(len(array))

    # if the length is the same for all arrays
    if arr_lengths.count(arr_lengths[0]) == len(arr_lengths):
        return True

    return False


def test_time_stamp_string_lengths(time_stamp_strings):
    """Tests whether the strings in the input array have the
       length of the expected format, which is "dd.mm.yy hh:mm:ss".

    Keyword arguments:
    time_stamp_strings -- array containing time stamp strings
    """
    length_of_expected_format = 16

    for time_stamp_string in time_stamp_strings:
        if len(time_stamp_string) != length_of_expected_format:
            raise ValueError("Found string with unexpected length")


def test_for_strictly_ascending_time_stamps(time_stamp_strings):
    """Tests if the time stamp strings in the input array are in strictly ascending order.
       Expected format of the time stamps: "dd.mm.yy hh:mm:ss".
    
    Keyoword arguments:
    time_stamp_strings -- array or list containing time stamp strings
    """
    for i in range(len(time_stamp_strings)-1):
        if not pp.is_later_time_stamp(time_stamp_strings[i+1], time_stamp_strings[i]):
            raise ValueError("Time stamps are not in strictly ascending order.")


def test_colon_positions(string_array):
    """Tests whether the strings in the input array have the colons
       at the same position as time stamp strings of the format "dd.mm.yyyy hh:mm".

    Keyword arguments:
    string_array -- input array containing strings
    """
    for string in string_array:
        colon_positions = [pos for pos, char in enumerate(string) if char == ':']

        if colon_positions != [13]:
            raise ValueError("Colon characters not at the expected positions")


def test_period_positions(string_array):
    """Tests whether the strings in the input array have the periods
       at the same position as time stamp strings of the format "dd.mm.yyyy hh:mm".

    Keyword arguments:
    string_array -- input array containing strings
    """
    for string in string_array:
        period_positions = [pos for pos, char in enumerate(string) if char == '.']

        if period_positions != [2, 5]:
            raise ValueError("Period characters not at the expected positions")


def test_for_abnormal_time_stamp_values(time_stamp_strings):
    """Tests whether the time stamp strings in the input array contain
       values for days, months, years, hours, minutes and seconds within
       the expected ranges. The expected format of each string is "dd.mm.yyyy hh:mm".

    Keyword arguments:
    time_stamp_strings -- input array containing strings
    """
    test_day_values(get_values_from_time_stamps_as_int(time_stamp_strings, 'days'))
    test_month_values(get_values_from_time_stamps_as_int(time_stamp_strings, 'months'))
    test_year_values(get_values_from_time_stamps_as_int(time_stamp_strings, 'years'))
    test_hour_values(get_values_from_time_stamps_as_int(time_stamp_strings, 'hours'))
    test_minute_values(get_values_from_time_stamps_as_int(time_stamp_strings, 'minutes'))


def test_for_negative_values(list_of_arrays):
    """Throws an error if arrays in the input list contains negative values.

    Keyword arguments:
    list_of_arrays -- input list of arrays
    """
    if contains_negative_values(list_of_arrays):
        raise ValueError("Unexpected negative values encountered")


def test_for_simultaneous_charge_and_discharge(charge, discharge):
    """Throws an error if arrays in the input list contain negative values.

    Keyword arguments:
    charge -- array with the charge rate per time step
    discharge -- array with the discharge rate per time step
    """
    num_time_steps_with_charge_and_discharge = sum((charge > 0) & (discharge > 0))

    if num_time_steps_with_charge_and_discharge > 0:
        raise ValueError("Simultaneous charge and discharge (one positive value for each)")


def get_values_from_time_stamps_as_int(time_stamp_strings, what_to_extract):
    """Returns the values for the days in the time stamp strings of the
       format "dd.mm.yyyy hh:mm" as an array of integers.

    Keyword arguments:
    time_stamp_strings -- array with time stamp strings of the format "dd.mm.yyyy hh:mm"
    what_to_extract -- string indicating what values to extract. Accepted values
                       are 'days', 'months', 'years', 'hours', 'minutes'.
    """
    if what_to_extract == 'days':
        substring_start_idx = 0
        substring_end_idx = 2
    elif what_to_extract == 'months':
        substring_start_idx = 3
        substring_end_idx = 5
    elif what_to_extract == 'years':
        substring_start_idx = 6
        substring_end_idx = 10
    elif what_to_extract == 'hours':
        substring_start_idx = 11
        substring_end_idx = 13
    elif what_to_extract == 'minutes':
        substring_start_idx = 14
        substring_end_idx = 16
    else:
        raise ValueError("Encountered unexpected input string encountered")

    values_as_int_list = [int(time_stamp[substring_start_idx:substring_end_idx])
                          for time_stamp in time_stamp_strings]

    return np.asarray(values_as_int_list)


def test_day_values(day_values):
    """Tests whether the integer values for the day in a calendar date
       are within expected boundaries (between 1 and 31).

    Keyword arguments:
    day_values -- array with days from calendar dates encoded as integers
    """
    values_larger_than_31 = day_values[day_values > 31]
    values_less_than_1 = day_values[day_values < 1]

    if sum(values_larger_than_31) + sum(values_less_than_1) > 0:
        raise ValueError("Unexpected values for the day in the calendar date")


def test_month_values(month_values):
    """Tests whether the integer values for the month in a calendar date
       are within expected boundaries (between 1 and 12).

    Keyword arguments:
    month_values -- array with months from calendar dates encoded as integers
    """
    values_larger_than_12 = month_values[month_values > 12]
    values_less_than_1 = month_values[month_values < 1]

    if sum(values_larger_than_12) + sum(values_less_than_1) > 0:
        raise ValueError("Unexpected values for the month in the calendar date")


def test_year_values(year_values):
    """Tests whether the integer values for the year in a calendar date
       are within expected boundaries (between 2013 and 2019).

    Keyword arguments:
    year_values -- array with years from calendar dates encoded as integers
    """
    values_larger_than_2019 = year_values[year_values > 2019]
    values_less_than_2013 = year_values[year_values < 2013]

    if sum(values_larger_than_2019) + sum(values_less_than_2013) > 0:
        raise ValueError("Unexpected values for the year in the calendar date")


def test_hour_values(hour_values):
    """Tests whether the integer values for the hour in a time of day
       are within expected boundaries (between 0 and 23).

    Keyword arguments:
    hour_values -- array with hours in a time of day encoded as integers
    """
    values_larger_than_23 = hour_values > 23
    values_less_than_0 = hour_values < 0

    if sum(values_larger_than_23) + sum(values_less_than_0) > 0:
        raise ValueError("Unexpected values for the hour")


def test_minute_values(minute_values):
    """Tests whether the integer values for the minute in a time of day
       are within expected boundaries (between 0 and 59).

    Keyword arguments:
    minute_values -- array with minutes in a time of day encoded as integers
    """
    values_larger_than_59 = minute_values[minute_values > 59]
    values_less_than_0 = minute_values[minute_values < 0]

    if sum(values_larger_than_59) + sum(values_less_than_0) > 0:
        raise ValueError("Unexpected values for the minutes")


def contains_negative_values(list_of_arrays):
    """Checks if arrays in the input list contain negative values.

    Keyword arguments:
    list_of_arrays -- input list of arrays
    """
    for array in list_of_arrays:
        if sum(array < 0) != 0:
            return True

    return False
