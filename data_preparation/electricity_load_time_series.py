import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from workalendar.europe import Germany

from data_preparation.time_features import generate_cyclical_time_value, convert_datetime_to_hour_of_the_week

DAY_IN_HOURS = 24
WEEK_IN_DAYS = 7


class PreparedDataset:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs

    def get_number_of_input_features(self):
        """
        :return: how many features are used in the dataset
        """
        return self.inputs.shape[1]

    def get_number_of_target_variables(self):
        """
        :return: how many target variables are used
        """
        return self.outputs.shape[1]


class ElectricityLoadTimeSeriesDataPreparer:
    """
    Is responsible for preparing the given electricity load time-series.
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
        self.df = df
        self.time_variable = time_variable
        self.target_variable = target_variable
        self.time_series_window_in_hours = time_series_window_in_hours
        self.forecasting_horizon_in_hours = forecasting_horizon_in_hours
        self.is_single_time_point_prediction = is_single_time_point_prediction
        self.include_time_information = include_time_information
        self.time_series_scaler = time_series_scaler
        self.is_training_set = is_training_set

        self.prepared_time_series_input: np.ndarray = np.array([])
        self.prepared_time_series_target: np.ndarray = np.array([])
        self.prepare_time_series_data()

    def prepare_time_series_data(self) -> None:
        """
        Prepares the time-series data.
        """
        # extract the unprocessed time series
        load_data = np.array(self.df[self.target_variable][::4])  # use hourly resolution
        time_stamps = np.array(self.df[self.time_variable][::4])

        # scale the values
        if self.is_training_set:
            scaled_load_data = self.time_series_scaler \
                .fit_transform(load_data.reshape(-1, 1)) \
                .flatten()
        else:
            scaled_load_data = self.time_series_scaler \
                .transform(load_data.reshape(-1, 1)) \
                .flatten()

        # create the input and target
        time_series = np.array(scaled_load_data, dtype=np.float32)
        time_series[0] = 0
        target_rows = []
        input_rows = []
        for index in range(self.time_series_window_in_hours, len(time_series) - self.forecasting_horizon_in_hours):
            # prepare the target
            if self.is_single_time_point_prediction:
                target_rows.append([time_series[index + self.forecasting_horizon_in_hours]])
            else:
                target_rows.append(time_series[index:index + self.forecasting_horizon_in_hours])

            # prepare the input
            hour_of_the_day_context = []
            hour_of_the_week_context = []
            is_workday_context = []
            is_holiday_context = []
            is_previous_day_workday_context = []
            is_next_day_workday_context = []
            if self.include_time_information:
                prediction_datetime = time_stamps[index + self.forecasting_horizon_in_hours]
                hour_of_the_day_context = generate_cyclical_time_value(prediction_datetime.hour, DAY_IN_HOURS)
                hour_of_the_week_context = generate_cyclical_time_value(
                    convert_datetime_to_hour_of_the_week(prediction_datetime), WEEK_IN_DAYS)

                calendar = Germany()
                if self.is_single_time_point_prediction:
                    is_workday_context = [calendar.is_working_day(prediction_datetime)]
                    is_holiday_context = [calendar.is_holiday(prediction_datetime)]
                    is_previous_day_workday_context = [
                        calendar.is_working_day(prediction_datetime - datetime.timedelta(days=1))]
                    is_next_day_workday_context = [
                        calendar.is_working_day(prediction_datetime + datetime.timedelta(days=1))]
                else:
                    predictions_datetime = time_stamps[index:index + self.forecasting_horizon_in_hours]
                    is_workday_context = [calendar.is_working_day(date_time) for date_time in predictions_datetime]
                    is_holiday_context = [calendar.is_holiday(date_time) for date_time in predictions_datetime]
                    is_previous_day_workday_context = [calendar.is_working_day(date_time - datetime.timedelta(days=1))
                                                       for date_time in predictions_datetime]
                    is_next_day_workday_context = [calendar.is_working_day(date_time + datetime.timedelta(days=1))
                                                   for date_time in predictions_datetime]

            previous_time_series_value: np.array = time_series[index - self.time_series_window_in_hours:index]
            input_row = np.concatenate((previous_time_series_value, hour_of_the_day_context, hour_of_the_week_context,
                                        is_workday_context, is_holiday_context, is_previous_day_workday_context,
                                        is_next_day_workday_context))
            input_rows.append(input_row)
        self.prepared_time_series_input = np.array(input_rows)
        self.prepared_time_series_target = np.array(target_rows)

    def get_prepared_data(self) -> PreparedDataset:
        """
        :return: the prepared dataset
        """
        return PreparedDataset(self.prepared_time_series_input, self.prepared_time_series_target)
