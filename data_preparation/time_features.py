import math


def generate_cyclical_time_value(time_value, period_length):
    """
    Converts the given time_value to a cyclical value. Improves features with a cyclical behavior like hour-of-day or
    day of week.

    :param time_value: the value to transform
    :param period_length: the length of the period (e.g. 24 for a hourly value)
    :return: the cyclical feature
    """
    return [
        math.sin(2 * math.pi * time_value / period_length),
        math.cos(2 * math.pi * time_value / period_length)
    ]
