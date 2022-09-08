from abc import ABC

import numpy as np
import pandas as pd
import datetime

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from workalendar.europe import BadenWurttemberg

from data_loading.time_features import generate_cyclical_time_value, convert_datetime_to_hour_of_the_week, \
    one_hot_encode


DAY_IN_HOURS = 24
WEEK_IN_DAYS = 7
WEEK_IN_HOURS = WEEK_IN_DAYS * DAY_IN_HOURS


class StandardDataset(Dataset, ABC):
    """
    The StandardDataset provides the functionality to prepare the data and load data point by index. It is intended to
    be used for ML models like linear regression or neural nets.
    The prepared dataset consists of
        - the time series target (dimension: [data points, target sequence length]),
        - time series input (dimension: [data points, input sequence length]),
        - and corresponding time-labels, which indicate the datetime of the forecast. (dimension: [data points]
    """

    def __init__(self, df: pd.DataFrame, time_variable: str, target_variable: str, time_series_window_in_hours: int,
                 forecasting_horizon_in_hours: int, is_single_time_point_prediction: bool,
                 include_time_information: bool, time_series_scaler: StandardScaler, is_training_set: bool,
                 one_hot_time_variables: bool):
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
        self._one_hot_time_variables = one_hot_time_variables

        self.prepared_time_series_input: torch.Tensor
        self.prepared_time_series_target: torch.Tensor
        self.time_labels: np.ndarray
        self._prepare_time_series_data()

    def __len__(self) -> int:
        return len(self.prepared_time_series_input)

    def __getitem__(self, index: int):
        """
        :param index: indicates the position of the tuple
        :return: the input-target tuple of the given index
        """
        return self.prepared_time_series_input[index], self.prepared_time_series_target[index]

    def get_number_of_input_features(self):
        """
        :return: how many features are used in the dataset
        """
        return self.prepared_time_series_input.shape[1]

    def get_number_of_target_variables(self):
        """
        :return: how many target variables are used
        """
        return self.prepared_time_series_target.shape[1]

    def _prepare_time_series_data(self) -> None:
        """
        Prepares the time-series data.
        """
        # extract the unprocessed time series
        load_data = np.mean(np.array(self._df[self._target_variable]).reshape((-1, 4)), axis=1)  # average of 4 values
        time_stamps = np.array(self._df[self._time_variable][::4])

        # scale the values
        if self._time_series_scaler:
            if self._is_training_set:
                scaled_load_data = self._time_series_scaler \
                    .fit_transform(load_data.reshape(-1, 1)) \
                    .flatten()
            else:
                scaled_load_data = self._time_series_scaler \
                    .transform(load_data.reshape(-1, 1)) \
                    .flatten()
        else:
            scaled_load_data = load_data

        # create the input and target
        time_series = np.array(scaled_load_data, dtype=np.float32)
        target_rows = []
        input_rows = []
        for index in range(self._time_series_window_in_hours, len(time_series) - self._forecasting_horizon_in_hours):
            # prepare the target
            if self._is_single_time_point_prediction:
                target_rows.append([time_series[index + self._forecasting_horizon_in_hours]])
            else:
                target_rows.append(time_series[index:index + self._forecasting_horizon_in_hours])

            # prepare the input
            day_context = []
            week_context = []
            year_context = []

            is_workday_context = []
            is_holiday_context = []
            is_previous_day_workday_context = []
            is_next_day_workday_context = []
            if self._include_time_information:
                prediction_datetime = time_stamps[index]
                if self._one_hot_time_variables:
                    day_context = one_hot_encode(prediction_datetime.hour, DAY_IN_HOURS)
                    #week_context = one_hot_encode(prediction_datetime.weekday(), WEEK_IN_DAYS)
                    year_context = one_hot_encode(prediction_datetime.month - 1, 12)
                else:
                    day_context = generate_cyclical_time_value(prediction_datetime.hour, DAY_IN_HOURS)
                    #week_context = generate_cyclical_time_value(
                    #    prediction_datetime.weekday(), WEEK_IN_DAYS)
                    year_context = generate_cyclical_time_value(prediction_datetime.weekofyear, 53)

                calendar = BadenWurttemberg()
                if self._is_single_time_point_prediction:
                    is_workday_context = [calendar.is_working_day(prediction_datetime)]
                    is_holiday_context = [calendar.is_holiday(prediction_datetime)]
                    is_previous_day_workday_context = [
                        calendar.is_working_day(prediction_datetime - datetime.timedelta(days=1))]
                    is_next_day_workday_context = [
                        calendar.is_working_day(prediction_datetime + datetime.timedelta(days=1))]
                else:
                    predictions_datetime = time_stamps[index:index + self._forecasting_horizon_in_hours]
                    is_workday_context = [calendar.is_working_day(date_time) for date_time in predictions_datetime]
                    is_holiday_context = [calendar.is_holiday(date_time) for date_time in predictions_datetime]
                    is_previous_day_workday_context = [calendar.is_working_day(date_time - datetime.timedelta(days=1))
                                                       for date_time in predictions_datetime]
                    is_next_day_workday_context = [calendar.is_working_day(date_time + datetime.timedelta(days=1))
                                                   for date_time in predictions_datetime]

            previous_time_series_value: np.array = time_series[index - self._time_series_window_in_hours:index]
            input_row = np.concatenate((previous_time_series_value,
                                        day_context,
                                        week_context,
                                        year_context,
                                        is_workday_context,
                                        is_holiday_context,
                                        is_previous_day_workday_context,
                                        is_next_day_workday_context))
            input_rows.append(input_row)
        self.prepared_time_series_input = torch.tensor(np.array(input_rows), dtype=torch.float32)
        self.prepared_time_series_target = torch.tensor(np.array(target_rows), dtype=torch.float32)
        self.time_labels = np.array(time_stamps[self._time_series_window_in_hours
                                                :len(time_series) - self._forecasting_horizon_in_hours])
