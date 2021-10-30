import datetime
from os import PathLike

import pandas as pd
from typing import List


class TimeInterval:

    def __init__(self, min_date: datetime.date, max_date: datetime.date):
        """
        Creates a time-interval: [min_date, max_date]

        :param min_date: the lower time bound (inclusive)
        :param max_date: the upper time bound (also inclusive)
        """
        self.min_date = min_date
        self.max_date = max_date

    def is_interval_overlapping(self, time_interval: 'TimeInterval') -> bool:
        """
        Checks whether the given time-interval overlaps with this time-interval. An overlap happens, if at least one
        date of one interval lays in the other interval. The interval bounds are considered as inclusive borders.

        :param time_interval: the interval to compare with
        :return: True if the intervals do overlap and else False
        """
        return not (self.min_date > time_interval.max_date or self.max_date < time_interval.min_date)


class TimeSeriesDataframeLoader:
    """
    Is responsible for loading the time-series data contained in the given csv file. Provides the ability to create the
    train, validation and test set.
    """

    def __init__(self, path_to_csv: PathLike, time_variable: str, target_variable: str):
        """
        Initiates the class by loading the raw csv into a dataframe

        :param path_to_csv:   declares the path to the csv-file
        :param time_variable: indicates the column in the csv-file, which contains the time information for the series
        """
        self.time_variable = time_variable
        self.csv_dataframe = self.load_dataframe_from_csv(path_to_csv, [time_variable],
                                                          [time_variable, target_variable])

    def get_train_validation_test_datasets(self, train_set_time_interval: TimeInterval,
                                           validation_set_time_interval: TimeInterval,
                                           test_set_time_interval: TimeInterval) \
            -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Splits the whole dataset into three parts: the training-, validation- and test-dataset. The whole dataset is
        split by the given time-interval filters. Those filters shall not overlap

        :param train_set_time_interval:      filter for the train dataset
        :param validation_set_time_interval: filter for the validation dataset
        :param test_set_time_interval:       filter for the test dataset
        :return: the three sub-datasets
        """
        if train_set_time_interval.is_interval_overlapping(validation_set_time_interval) \
                or validation_set_time_interval.is_interval_overlapping(test_set_time_interval):
            raise Exception('The train, validation and test dataset should not overlap.')

        return self.extract_dataframe_by_year_filter(train_set_time_interval), \
               self.extract_dataframe_by_year_filter(validation_set_time_interval), \
               self.extract_dataframe_by_year_filter(test_set_time_interval)

    def extract_dataframe_by_year_filter(self, time_interval: TimeInterval) -> pd.DataFrame:
        """
        Extracts from the dataframe a subset dataframe. The subset is created by only including the rows which are
        within the given time_interval.

        :param time_interval: filter to include only rows with the date inside the interval (is inclusive on both sides)
        :return: the filtered dataframe; if the dataframe does not contain the years, then the result is empty
        """
        return self.csv_dataframe[(self.csv_dataframe[self.time_variable].dt.date >= time_interval.min_date)
                                  & (self.csv_dataframe[self.time_variable].dt.date <= time_interval.max_date)]

    @staticmethod
    def load_dataframe_from_csv(path_to_csv: PathLike, columns_to_parse_as_dates: List[str],
                                columns_to_include: List[str]) -> pd.DataFrame:
        """
        Loads the CSV-file from the given path as a dataframe

        :param path_to_csv:               the path to the CSV-file
        :param columns_to_parse_as_dates: the columns to parse as dates
        :param columns_to_include:        filter to include only this columns
        :return: returns the dataframe if the file is present and can be read
        """
        return pd.read_csv(path_to_csv, parse_dates=columns_to_parse_as_dates, usecols=columns_to_include)
