"""This file contains different models to set the buy and sell price for electricity"""
import numpy as np
import pickle

def get_constant_buying_price_in_euros_per_kWh(buying_price_in_euros=0.2):
    """Returns the same buying price for each kWh of electricity for all time steps.

    Keyword arguments:
    buy_price_in_euros -- the buying price for one kWh in euros
    """
    return buying_price_in_euros


def get_constant_selling_price_in_euros_per_kWh(selling_price_in_euros=0.1):
    """Returns the same selling price for each kWh of electricity for all time steps.

    Keyword arguments:
    sell_price_in_euros -- the selling price for one kWh in euros
    """
    return selling_price_in_euros


def get_normally_distributed_buying_price_in_euros_per_kWh(price_mean=0.2, price_std=0.1):
    """Returns a normally distributed buying price for each kWh of electricity.

    Keyword arguments:
    price_mean -- mean of the buying price per kWh in euros
    price_std -- standard deviation of the buying price
    """
    return np.random.normal(price_mean, price_std)


def get_normally_distributed_selling_price_in_euros_per_kWh(price_mean=0.1, price_std=0.05):
    """Returns a normally distributed selling price for each kWh of electricity.

    Keyword arguments:
    price_mean -- mean of the selling price per kWh in euros
    price_std -- standard deviation of the selling price
    """
    return np.random.normal(price_mean, price_std)


def add_electricity_buy_and_sell_prices_to_s_a_pairs(state_action_pairs, time_stamps, state_action_features,
                                                     pricing_model, price_update_interval_in_mins):
    """Appends a column containing the electricity prices corresponding to the given price model and the given time stamps 
       in €/kWh to the given matrix containing state-action pairs.

    Keyword arguments:
    state_action_pairs -- 2D matrix containing state-action pairs, (rows: state-action pair, columns: features)
    time_stamps -- list of time stamps as strings, where each time stamp corresponds to the state-action pair
                   in the row in state_action_pairs with the same index (expected format: "dd.mm.yyyy hh:mm)
    state_action_features -- list of strings that are the names of the features in the state-action pair 
    pricing_model -- string that indicates which pricing model the prices should come from.
                     Options: 'constant', 'normally_distributed', 'deterministic_market', 'noisy_market'
    price_update_interval_in_mins -- period after which the electricity prices will be updated in 
                                     minutes (unless the pricing model is constant)
    """
    buy_price_in_euros_per_kWh = np.empty([len(time_stamps),1])
    sell_price_in_euros_per_kWh = np.empty([len(time_stamps),1])

    for i, time_stamp in enumerate(time_stamps):
        if i % price_update_interval_in_mins == 0:
            if pricing_model == 'constant':
                current_buy_price = get_constant_buying_price_in_euros_per_kWh()
                current_sell_price = get_constant_selling_price_in_euros_per_kWh()
            elif pricing_model == 'normally_distributed':
                current_buy_price = get_normally_distributed_buying_price_in_euros_per_kWh()
                current_sell_price = get_normally_distributed_selling_price_in_euros_per_kWh()
            elif pricing_model == 'deterministic_market':
                current_buy_price = get_deterministic_market_price_in_euros_per_kWh(time_stamp, time_stamps[0])
                current_sell_price = current_buy_price
            elif pricing_model == 'noisy_market':
                current_buy_price = get_noisy_market_price_in_euros_per_kWh(time_stamp, time_stamps[0])
                current_sell_price = current_buy_price
            else:
                raise ValueError("Unknown pricing model: " + pricing_model)

        buy_price_in_euros_per_kWh[i] = current_buy_price
        sell_price_in_euros_per_kWh[i] = current_sell_price

    s_a_pairs_with_prices = np.concatenate((state_action_pairs, buy_price_in_euros_per_kWh, sell_price_in_euros_per_kWh), axis=1)
    state_action_features += ['buy_price_in_euros_per_kWh', 'sell_price_in_euros_per_kWh']

    return s_a_pairs_with_prices, state_action_features


def get_noisy_market_price_in_euros_per_kWh(time_stamp, first_time_stamp_in_dataset, train_mode=None):
    """Returns the weighted average price in euros per kWh on the energy exchange with added i.i.d. Gaussian noise 
       at the time of the given time stamp.

    Keyword arguments:
    time_stamp -- time stamp as a string in the format of "dd.mm.yyyy hh:mm"
    first_time_stamp_in_dataset -- the first time stamp of the training / test dataset in the format of "dd.mm.yyyy hh:mm"
    train_mode -- Boolean variable indicating whether the requested price is for the training or the test simulation
    """
    price_in_euros_per_kWh = get_deterministic_market_price_in_euros_per_kWh(time_stamp, first_time_stamp_in_dataset, train_mode)
    price_in_euros_with_noise = price_in_euros_per_kWh + np.random.normal(loc=0.0, scale=0.1)

    return price_in_euros_with_noise


def get_deterministic_market_price_in_euros_per_kWh(time_stamp, first_time_stamp_in_dataset, train_mode=None):
    """Returns the weighted average price in euros per kWh on the energy exchange at the time of the given time stamp.

    Keyword arguments:
    time_stamp -- time stamp as a string in the format of "dd.mm.yyyy hh:mm"
    first_time_stamp_in_dataset -- the first time stamp of the training / test dataset in the format of "dd.mm.yyyy hh:mm"
    train_mode -- Boolean variable indicating whether the requested price is for the training or the test simulation
    """
    folder_path = './stored_data/state_transitions_dataset/temporal_resolution_1_min'
    if train_mode == None:
        with open(folder_path + '/weighted_average_electricity_prices_in_euros_per_kWh_08-12-2016_31-12-2016.pkl', 'rb') as f:
            prices_in_euros_per_kWh = pickle.load(f)
    elif train_mode:
        with open(folder_path + '/training_simulation_weighted_average_price_in_euros_per_MWh_15-12-2016_31-12-2016.pkl', 'rb') as f:
            prices_in_euros_per_MWh = pickle.load(f)
            prices_in_euros_per_kWh = prices_in_euros_per_MWh / 1000.0 
    else:
        with open(folder_path + '/test_simulation_weighted_average_price_in_euros_per_MWh_08-12-2016_15-12-2016.pkl', 'rb') as f:
            prices_in_euros_per_MWh = pickle.load(f)
            prices_in_euros_per_kWh = prices_in_euros_per_MWh / 1000.0 

    hour_of_day = int(time_stamp[11:13])
    day_idx = get_difference_in_days_disregarding_time(time_stamp, first_time_stamp_in_dataset)
    hour_idx = hour_of_day - 1
    quarter_hour_idx = get_quarter_hour_index(time_stamp)

    price_in_euros_per_kWh = prices_in_euros_per_kWh[day_idx, hour_idx, quarter_hour_idx]

    return price_in_euros_per_kWh


def get_difference_in_days_disregarding_time(time_stamp_1, time_stamp_2):
    """Computes and returns the temporal difference between the dates of two given time stamps in days,
       disregarding the time of day. The time stamps must be from days in the same month!

    Keyword arguments:
    time_stamp_1 -- time stamp as a string in the format of "dd.mm.yyyy hh:mm"
    time_stamp_2 -- time stamp as a string in the format of "dd.mm.yyyy hh:mm"
    """
    year_1 = int(time_stamp_1[6:10])
    month_1 = int(time_stamp_1[3:5])
    day_1 = int(time_stamp_1[:2])
    year_2 = int(time_stamp_2[6:10])
    month_2 = int(time_stamp_2[3:5])
    day_2 = int(time_stamp_2[:2])

    if year_1 != year_2 or month_1 != month_2:
        raise ValueError("Given time stamps are not from the same month.")
    
    difference_in_days = abs(day_1 - day_2)

    return difference_in_days


def get_quarter_hour_index(time_stamp):
    """Returns an inded indicating in what quarter of the hour the time in the given time stamp is.
       The index therefore ranges from 0 to 3.
    
    Keyword arguments:
    time_stamp -- time stamp as a string in the format of "dd.mm.yyyy hh:mm"
    """
    minute = int(time_stamp[-2:])
    quarter_idx = int(np.floor(minute / 15))

    if quarter_idx not in [0,1,2,3]:
        raise ValueError("Unexpected resulting value for the index of the quarter of the hour.")

    return quarter_idx
