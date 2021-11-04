import datetime
import enum
import os
from pathlib import Path

import torch
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from data_loading.time_series import TimeSeriesDataframeLoader, TimeInterval
from data_preparation.electricity_load_time_series import ElectricityLoadTimeSeriesDataPreparer, PreparedDataset
from evaluation.evaluator import Evaluator
from models.simple_nn import SimpleNeuralNet
from models.wrappers.pytorch_model_wrapper import PytorchModelWrapper
from models.wrappers.sklearn_model_wrapper import SklearnModelWrapper
from training.trainer import Trainer

UTC_TIMESTAMP = 'utc_timestamp'
TARGET_VARIABLE = 'DE_transnetbw_load_actual_entsoe_transparency'
PATH_TO_CSV = os.path.join('data', 'opsd', 'time_series_15min_singleindex.csv')


class ModelType(enum.Enum):
    LinearRegression = 'LinearRegression'
    SimpleNeuralNet = 'SimpleNeuralNet'


class Pipeline:

    def __init__(self, model_type: ModelType, forecasting_horizon: int, predict_single_value: bool, window_length: int,
                 include_time_context: bool):
        """
        Creates a pipeline.

        :param model_type:           determines which model is executed
        :param forecasting_horizon:  how far the prediction reaches
        :param predict_single_value: indicates whether the target is a single value
        :param window_length:        how many historical values of the time series are used as input for a forecast
        :param include_time_context: indicates whether the time information is used as an additional input
        """
        self.model_type = model_type
        self.forecasting_horizon = forecasting_horizon
        self.predict_single_value = predict_single_value
        self.window_length = window_length
        self.include_time_context = include_time_context

    def prepare_datasets(self, train_df: DataFrame, validation_df: DataFrame,
                         test_df: DataFrame, scaler) -> (PreparedDataset, PreparedDataset, PreparedDataset):
        """
        Prepares the the training, validation and test datasets.

        :param train_df:      a dataframe with the training data
        :param validation_df: a dataframe with the validation data
        :param test_df:       a dataframe with the test data
        :param scaler:        scales the time-series data
        :return: the prepared datasets (train, validation, test)
        """
        prepared_train_set = ElectricityLoadTimeSeriesDataPreparer(train_df, UTC_TIMESTAMP, TARGET_VARIABLE,
                                                                   self.window_length, self.forecasting_horizon,
                                                                   self.predict_single_value, self.include_time_context,
                                                                   scaler, True).get_prepared_data()
        prepared_validation_set = ElectricityLoadTimeSeriesDataPreparer(validation_df, UTC_TIMESTAMP, TARGET_VARIABLE,
                                                                        self.window_length, self.forecasting_horizon,
                                                                        self.predict_single_value,
                                                                        self.include_time_context, scaler, False
                                                                        ).get_prepared_data()
        prepared_test_set = ElectricityLoadTimeSeriesDataPreparer(test_df, UTC_TIMESTAMP, TARGET_VARIABLE,
                                                                  self.window_length, self.forecasting_horizon,
                                                                  self.predict_single_value, self.include_time_context,
                                                                  scaler, False).get_prepared_data()
        return prepared_train_set, prepared_validation_set, prepared_test_set

    def start(self) -> None:
        """
        Starts the pipeline calculations.
        """
        dfl = TimeSeriesDataframeLoader(Path(PATH_TO_CSV), UTC_TIMESTAMP, TARGET_VARIABLE)
        train, validation, test = dfl.get_train_validation_test_datasets(
            TimeInterval(datetime.date(2015, 1, 1), datetime.date(2017, 12, 31)),
            TimeInterval(datetime.date(2018, 1, 1), datetime.date(2018, 12, 31)),
            TimeInterval(datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)))

        scaler = StandardScaler()
        prepared_train_set, prepared_validation_set, prepared_test_set = self.prepare_datasets(train, validation, test,
                                                                                               scaler)
        train_tds = TensorDataset(torch.tensor(prepared_train_set.inputs, dtype=torch.float32),
                                  torch.tensor(prepared_train_set.outputs, dtype=torch.float32))
        validation_tds = TensorDataset(torch.tensor(prepared_validation_set.inputs, dtype=torch.float32),
                                       torch.tensor(prepared_validation_set.outputs, dtype=torch.float32))
        # test_tds = TensorDataset(prepared_test_set.get_prepared_data())

        train_dl = DataLoader(train_tds, batch_size=16, sampler=SequentialSampler(train_tds))
        validation_dl = DataLoader(validation_tds, batch_size=1028, sampler=SequentialSampler(validation_tds))

        if self.model_type == ModelType.LinearRegression:
            linear_regression = LinearRegression()
            linear_regression.fit(prepared_train_set.inputs, prepared_train_set.outputs)

            model_wrapper = SklearnModelWrapper(linear_regression)

        elif self.model_type == ModelType.SimpleNeuralNet:
            model = SimpleNeuralNet(prepared_train_set.get_number_of_input_features(),
                                    prepared_train_set.get_number_of_target_variables())
            criterion = MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
            trainer = Trainer(train_dl, validation_dl, model, criterion, optimizer, 250)

            trainer.train()

            model_wrapper = PytorchModelWrapper(model)

        else:
            print(self.model_type)
            print(type(self.model_type))
            raise Exception('Received a model, which is not included.')

        evaluator = Evaluator(model_wrapper, prepared_validation_set, scaler)
        evaluator.evaluate()
        pass

    def save_to_file(self):
        pass
