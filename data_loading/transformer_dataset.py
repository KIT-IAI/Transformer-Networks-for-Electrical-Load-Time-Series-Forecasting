import datetime
from abc import ABC
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from workalendar.europe import Germany

from data_preparation.time_features import generate_cyclical_time_value, convert_datetime_to_hour_of_the_week


class TransformerDataset(Dataset, ABC):
    """
    The Dataset which is used
    """

    def __init__(self, df: pd.DataFrame, time_variable: str, target_variable: str, time_series_window_in_hours: int,
                 forecasting_horizon_in_hours: int, is_single_time_point_prediction: bool,
                 include_time_information: bool, time_series_scaler: StandardScaler, is_training_set: bool):
        """
        Creates the data-preparer.

        :param df:                              contains the raw data to prepare
        :param time_series_window_in_hours:     how many time-series values are used as input
        :param forecasting_horizon_in_hours:    how far the prediction reaches
        :param is_single_time_point_prediction: indicates whether a single time-point is used as target
        :param include_time_information:        indicates whether time-information is included in the input
        :param time_series_scaler               scales the time-series data
        :param is_training_set                  indicates, whether the given dataframe contains the training data
        """
        self._df = df
        self._time_variable = time_variable
        self._target_variable = target_variable
        self._time_series_window_in_hours = time_series_window_in_hours
        self._forecasting_horizon_in_hours = forecasting_horizon_in_hours
        self._is_single_time_point_prediction = is_single_time_point_prediction
        self._include_time_information = include_time_information
        self._time_series_scaler = time_series_scaler
        self._is_training_set = is_training_set

        self.elements: List
        self._prepare_time_series_data()

    def __len__(self) -> int:
        return len(self.rows) - self._time_series_window_in_hours - self._forecasting_horizon_in_hours

    def __getitem__(self, index: int):
        # return self.rows[index:index + self._time_series_window_in_hours], \
        #        self.rows[index + self._time_series_window_in_hours - self._forecasting_horizon_in_hours + 1
        #                  : index + self._time_series_window_in_hours + 1], \
        #        self.rows[index + self._time_series_window_in_hours: index + self._time_series_window_in_hours
        #                                                             + self._forecasting_horizon_in_hours],
        return self.rows[index:index + self._time_series_window_in_hours], \
               self.rows[index + self._time_series_window_in_hours: index + self._time_series_window_in_hours
                                                                    + self._forecasting_horizon_in_hours],

    def _prepare_time_series_data(self) -> None:
        """
        Prepares the time-series data.
        """
        # extract the unprocessed time series
        load_data = np.array(self._df[self._target_variable][::4])  # use hourly resolution
        time_stamps = np.array(self._df[self._time_variable][::4])

        # scale the values
        if self._is_training_set:
            scaled_load_data = self._time_series_scaler \
                .fit_transform(load_data.reshape(-1, 1)) \
                .flatten()
        else:
            scaled_load_data = self._time_series_scaler \
                .transform(load_data.reshape(-1, 1)) \
                .flatten()

        self.rows = []
        for index in range(0, len(time_stamps)):
            load_data_value = scaled_load_data[index]
            time_stamp = time_stamps[index]
            hour_of_the_day_context = generate_cyclical_time_value(time_stamp.hour, 23)
            hour_of_the_week_context = generate_cyclical_time_value(convert_datetime_to_hour_of_the_week(time_stamp), 6)

            calendar = Germany()
            is_workday_context = calendar.is_working_day(time_stamp)
            is_holiday_context = calendar.is_holiday(time_stamp)
            is_previous_day_workday_context = calendar.is_working_day(time_stamp - datetime.timedelta(days=1))
            is_next_day_workday_context = calendar.is_working_day(time_stamp + datetime.timedelta(days=1))

            self.rows.append([
                load_data_value,
                hour_of_the_week_context[0], hour_of_the_week_context[1],
                hour_of_the_day_context[0], hour_of_the_day_context[1],
                is_workday_context, is_holiday_context, is_previous_day_workday_context, is_next_day_workday_context
            ])
        self.rows = torch.tensor(np.array(self.rows, dtype=np.float32))
        # # create the input and target
        # time_series = np.array([np.array([d], dtype=np.float32) for d in scaled_load_data])
        # time_series[0] = 0
        #
        # # create the labels
        # t2 = np.array([[time_stamp.hour / 23 - 0.5, time_stamp.weekday() / 6 - 0.5,
        #                 time_stamp.month / 12 - 0.5, time_stamp.day / 31 - 0.5] for time_stamp in time_stamps])
        #
        # e = []
        # for index in range(self._time_series_window_in_hours, len(time_series) - self._forecasting_horizon_in_hours):
        #     seq_x = time_series[index - self._time_series_window_in_hours:index]
        #     seq_x_mark = t2[index - self._time_series_window_in_hours:index]
        #     seq_y = time_series[index:index + self._forecasting_horizon_in_hours]
        #     seq_y_mark = t2[index:index + self._forecasting_horizon_in_hours]
        #     e.append(np.array([seq_x, seq_y, seq_x_mark, seq_y_mark]))
        # self.elements = np.array(e)
