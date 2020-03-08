import requests
from bs4 import BeautifulSoup
from lxml import html
import numpy as np
from preprocessing import preprocessing as pp

def get_weighted_average_electricity_prices_in_euros_per_kWh(start_date, end_date):
    """Returns the weighted average prices from the given start date to the given
       end date (included) as a matrix containing the prices in €/MWh.

       Dimensions of returned matrix:
       rows - day
       columns - hour
       3rd dim - quarter of the hour

    Keyword arguments:
    start_date -- date from which the weighted averages are of interest (included) - format: "dd.mm.yyyy"
    end_date -- date until which the weighted averages are of interest (included) - format: "dd.mm.yyyy"
    """
    weighted_avg_prices_in_euros_per_kWh = []
    date = start_date + " 12:00" # time is attached as a string so it matches the format that is expected a used function
    end_date_reached = False

    while True:
        weighted_avg_prices_in_euros_per_kWh.append([])
        url_format_date = convert_to_url_date_string(date)
        url = "https://www.epexspot.com/de/marktdaten/intradaycontinuous/intraday-table/" + url_format_date + "/DE"
        page_html = requests.get(url)
        soup = BeautifulSoup(page_html.text, 'lxml')

        for hour_idx in range(24):
            weighted_avg_prices_in_euros_per_kWh[-1].append([])
            for quarter_idx in range(4):
                id_string = "intra_15_" + str(hour_idx) + "_" + str(quarter_idx) + "_data"
                quarter_trading_data = soup.find(id=id_string)
                weighted_average_in_euros_per_MWh = find_weighted_average_price_in_euros_per_MWh(quarter_trading_data.prettify())
                weighted_average_in_euros_per_kWh = weighted_average_in_euros_per_MWh / 1000.0
                weighted_avg_prices_in_euros_per_kWh[-1][hour_idx].append(weighted_average_in_euros_per_kWh)

        next_day_date = pp.get_corresponding_time_stamp_string(date, 24 * 60)
        date = next_day_date

        if end_date_reached:
            break
        if next_day_date[:10] == end_date:
            end_date_reached = True

    weighted_avg_prices_in_euros_per_kWh = np.asarray(weighted_avg_prices_in_euros_per_kWh)

    return weighted_avg_prices_in_euros_per_kWh


def convert_to_url_date_string(time_stamp):
    """Converts the date of the given time stamp string into a format that
       is needed for the URL from which the electricity prices will be crawled.

    Keyword arguments:
    time_stamp -- time stamp as a string in the format of "dd.mm.yyyy hh:mm"
    """
    year = time_stamp[6:10]
    month = time_stamp[3:5]
    day = time_stamp[:2]
    url_fromat_date = year + "-" + month + "-" + day

    return url_fromat_date


def find_weighted_average_price_in_euros_per_MWh(quarter_trading_data):
    """Returns the weighted average price contained in the trading data of a given quarter in €/MWh.

    Keyword arguments:
    quarter_trading_data -- string that contains the html of trading data in a given quarter of an hour
    """
    numbers_in_string = find_numbers_with_comma_in_string(quarter_trading_data)
    weighted_average_price_in_euros_per_MWh = numbers_in_string[11]

    return weighted_average_price_in_euros_per_MWh


def find_numbers_with_comma_in_string(input_string):
    """Returns all the numbers in the given string with a comma as floats in a list.

    Keyword arguments:
    input_string -- a given string
    """
    found_numbers = []
    digits = [str(i) for i in range(10)]
    i = 0

    while True:
        if input_string[i] in digits: 
            number_as_string = get_number_as_string(input_string[i:])
            if '.' in number_as_string:
                number_as_float = float(number_as_string)
                found_numbers.append(number_as_float)
            i += len(number_as_string)
        else:
            i += 1

        if i >= len(input_string):
            break

    return found_numbers


def get_number_as_string(input_string):
    """Returns a list of characters that are the digits and the comma that make up the 
       number at the beginning of the given string.

    Keyword arguments:
    input_string -- a given string whose first character is a digit
    """
    digits = [str(i) for i in range(10)]
    number_as_string = '' + input_string[0]

    if input_string[0] not in digits:
        raise ValueError("First character is not a digit")

    for i in range(1, len(input_string)):
        comma_encountered = False
        if input_string[i] in digits or (input_string[i] == ',' and not comma_encountered):
            if input_string[i] == ',':
                comma_encountered = True
                number_as_string += '.'
            else:
                number_as_string += input_string[i]
        else:
            break

    return number_as_string
