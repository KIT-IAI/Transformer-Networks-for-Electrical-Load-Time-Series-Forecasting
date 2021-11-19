import datetime
import os
from pathlib import Path

import torch
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.nn import L1Loss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, SequentialSampler

from data_loading.standard_dataset import StandardDataset
from data_loading.time_series import TimeSeriesDataframeLoader, TimeInterval
from data_loading.transformer_dataset import TransformerDataset
from data_preparation.electricity_load_time_series import ElectricityLoadTimeSeriesDataPreparer, PreparedDataset
from evaluation.evaluator import Evaluator
from experiments.experiment import Experiment
from models.model_type import ModelType
from models.simple_nn import SimpleNeuralNet
from models.tranformers.transformer import TimeSeriesTransformer
from models.wrappers.pytorch_model_wrapper import PytorchModelWrapper
from models.wrappers.sklearn_model_wrapper import SklearnModelWrapper
from training.neural_net_trainer import NeuralNetTrainer
from training.training_config import TrainingConfig
from training.tranformer_trainer import TransformerTrainer

UTC_TIMESTAMP = 'utc_timestamp'
TARGET_VARIABLE = 'DE_transnetbw_load_actual_entsoe_transparency'
PATH_TO_CSV = os.path.join('data', 'opsd', 'time_series_15min_singleindex.csv')


class Pipeline:

    def __init__(self, model_type: ModelType, forecasting_horizon: int, predict_single_value: bool, window_length: int,
                 include_time_context: bool, training_config: TrainingConfig, args):
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
        self.training_config = training_config

        self.args = args

        self.experiment = None

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
            TimeInterval(datetime.date(2015, 1, 2), datetime.date(2017, 12, 31)),
            TimeInterval(datetime.date(2018, 1, 1), datetime.date(2018, 12, 31)),
            TimeInterval(datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)))

        scaler = StandardScaler()

        validation_outputs: torch.Tensor
        validation_targets: torch.Tensor
        if self.model_type == ModelType.LinearRegression or self.model_type == ModelType.SimpleNeuralNet:
            train_dataset = StandardDataset(train, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                                            self.forecasting_horizon, self.predict_single_value,
                                            self.include_time_context, scaler, True)
            validation_dataset = StandardDataset(train, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                                                 self.forecasting_horizon, self.predict_single_value,
                                                 self.include_time_context, scaler, False)
            if self.model_type == ModelType.LinearRegression:
                linear_regression = LinearRegression()
                linear_regression.fit(train_dataset.prepared_time_series_input,
                                      train_dataset.prepared_time_series_target)

                model_wrapper = SklearnModelWrapper(linear_regression, self.model_type)
                training_report = None
            else:
                train_dl = DataLoader(train_dataset, batch_size=80, sampler=SequentialSampler(train_dataset))
                validation_dl = DataLoader(validation_dataset, batch_size=80,
                                           sampler=SequentialSampler(validation_dataset))
                model = SimpleNeuralNet(train_dataset.get_number_of_input_features(),
                                        train_dataset.get_number_of_target_variables())
                criterion = L1Loss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.training_config.learning_rate)
                scheduler = StepLR(optimizer, self.args.learning_rate_scheduler_step,
                                   self.args.learning_rate_scheduler_gamma)
                trainer = NeuralNetTrainer(train_dl, validation_dl, model, criterion, optimizer,
                                           self.training_config.max_epochs, scheduler,
                                           self.training_config.use_early_stopping,
                                           self.training_config.early_stopping_patience)
                training_report = trainer.train()

                model_wrapper = PytorchModelWrapper(model, self.model_type)
            validation_outputs = model_wrapper.predict(torch.Tensor(validation_dataset.prepared_time_series_input))
            validation_targets = torch.Tensor(validation_dataset.prepared_time_series_target)
        elif self.model_type == ModelType.TimeSeriesTransformer:
            train_dataset = TransformerDataset(train, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                                               self.forecasting_horizon, self.predict_single_value,
                                               self.include_time_context, scaler, True)
            validation_dataset = TransformerDataset(validation, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                                                    self.forecasting_horizon, self.predict_single_value,
                                                    self.include_time_context, scaler, False)
            train_dl = DataLoader(train_dataset, batch_size=1000)
            validation_dl = DataLoader(validation_dataset, batch_size=1000)
            model = TimeSeriesTransformer(d_model=self.args.transformer_d_model,
                                          input_features_count=self.args.transformer_input_features_count,
                                          num_encoder_layers=self.args.transformer_num_encoder_layers,
                                          num_decoder_layers=self.args.transformer_num_decoder_layers,
                                          dim_feedforward=self.args.transformer_dim_feedforward,
                                          dropout=self.args.transformer_dropout,
                                          attention_heads=self.args.transformer_attention_heads)
            criterion = L1Loss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.training_config.learning_rate, betas=(0.9, 0.98))
            scheduler = StepLR(optimizer, self.args.learning_rate_scheduler_step,
                               self.args.learning_rate_scheduler_gamma)
            trainer = TransformerTrainer(train_dl, validation_dl, model, criterion, optimizer,
                                         self.training_config.max_epochs, scheduler,
                                         self.training_config.use_early_stopping,
                                         self.training_config.early_stopping_patience)
            training_report = trainer.train()

            model_wrapper = PytorchModelWrapper(model, self.model_type)
            pass
        else:
            print(self.model_type)
            print(type(self.model_type))
            raise Exception('Received a model, which is not included.')

        evaluator = Evaluator(validation_outputs, validation_targets, scaler, self.forecasting_horizon)
        evaluation = evaluator.evaluate()

        self.experiment = Experiment(model_wrapper, evaluation, self.training_config, training_report)
        print(str(self.experiment))

    def save_to_file(self):
        self.experiment.save_to_json_file()
