import datetime
import math
import numpy as np


DAY_IN_HOURS = 24


def generate_cyclical_time_value(time_value, period_length):
    """
    Converts the given time_value to a cyclical value. Improves features with a cyclical behavior like hour-of-day or
    day of week.

    :param time_value: the value to transform
    :param period_length: the length of the period (e.g. 23 for a hourly value)
    :return: the cyclical feature
    """
    return [
        math.sin(2 * math.pi * time_value / period_length),
        math.cos(2 * math.pi * time_value / period_length)
    ]


def convert_datetime_to_hour_of_the_week(dt: datetime.datetime) -> int:
    """
    Converts the given datetime to the hour of the week.

    :param dt: the datetime to convert
    :return: the hour of the week in the interval [0, 167]
    """
    return datetime.datetime.weekday(dt) * DAY_IN_HOURS + dt.hour


def one_hot_encode(time_value, period_length):
    encoding = np.zeros(period_length, dtype=int)
    encoding[time_value] = 1
    return encoding
