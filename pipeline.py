import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pickle
import time

from data_loading.standard_dataset import StandardDataset
from data_loading.time_series import TimeSeriesDataframeLoader, TimeInterval
from data_loading.transformer_dataset import TransformerDataset
from evaluation.evaluator import Evaluator
from experiments.experiment import Experiment
from models.model_type import ModelType
from models.simple_nn import SimpleNeuralNet
from models.tranformers.informer import Informer
from models.tranformers.transformer import TimeSeriesTransformer
from models.tranformers.transformer_convolution_attention import TimeSeriesTransformerWithConvolutionalAttention
from models.wrappers.base_model_wrapper import BaseModelWrapper
from models.wrappers.pytorch_neural_net_model_wrapper import PytorchNeuralNetModelWrapper
from models.wrappers.pytorch_transformer_model_wrapper import PytorchTransformerModelWrapper
from models.wrappers.sklearn_model_wrapper import SklearnModelWrapper

# dataset related constants
UTC_TIMESTAMP = 'utc_timestamp'
TARGET_VARIABLE = 'DE_transnetbw_load_actual_entsoe_transparency'
PATH_TO_CSV = os.path.join('data', 'opsd-time_series-2020-10-06', 'time_series_15min_singleindex.csv')


class Pipeline:
    """
    The Pipeline is the starting point for the training, validation and testing of the models.
    """

    def __init__(self, model_type: ModelType, args: argparse.Namespace):
        """
        Creates a pipeline.

        :param model_type: determines which model is executed
        :param args: all configuration arguments
        """
        self.model_type = model_type
        self.args = args
        self.forecasting_horizon = args.forecasting_horizon
        self.predict_single_value = args.predict_single_value
        self.window_length = args.time_series_window
        self.include_time_context = args.include_time_context
        self.one_hot_time_variables = args.one_hot_time_variables

        self.args = args

        self.experiment = None

    def start(self, run_experiment: bool = True) -> None:
        """
        Starts the pipeline calculations.
        First, the dataset is loaded, split into train, validation and test set, and preprocessed. Afterward, the models
        are trained depending on their underlying architecture. Last, the model is evaluated.
        """

        model = None

        # Load the data and it split into subsets
        dfl = TimeSeriesDataframeLoader(Path(PATH_TO_CSV), UTC_TIMESTAMP, TARGET_VARIABLE)
        train, validation, test = dfl.get_train_validation_test_datasets(
            TimeInterval(datetime.date(2015, 1, 2), datetime.date(2017, 12, 31)),
            TimeInterval(datetime.date(2018, 1, 1), datetime.date(2018, 12, 31)),
            TimeInterval(datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)))

        scaler = StandardScaler()
        model_wrapper: BaseModelWrapper
        train_dataset: Dataset
        validation_dataset: Dataset
        test_dataset: Dataset

        # differentiate between the different architecture types (Transformers need other data format)
        if not self.model_type.is_transformer_model():
            # convert the raw data into the preprocessed datasets
            train_dataset: StandardDataset = StandardDataset(train, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                                                             self.forecasting_horizon, self.predict_single_value,
                                                             self.include_time_context, scaler, True,
                                                             self.one_hot_time_variables)
            validation_dataset: StandardDataset = StandardDataset(validation, UTC_TIMESTAMP, TARGET_VARIABLE,
                                                                  self.window_length, self.forecasting_horizon,
                                                                  self.predict_single_value, self.include_time_context,
                                                                  scaler, False, self.one_hot_time_variables)
            test_dataset: StandardDataset = StandardDataset(test, UTC_TIMESTAMP, TARGET_VARIABLE,
                                                            self.window_length, self.forecasting_horizon,
                                                            self.predict_single_value, self.include_time_context,
                                                            scaler, False, self.one_hot_time_variables)

            # differentiate between Linear Regression and Neural Net, because they are trained with different libraries
            if self.model_type == ModelType.LinearRegression:
                linear_regression = LinearRegression()
                model_wrapper = SklearnModelWrapper(linear_regression, self.model_type, self.args)
            else:
                model = SimpleNeuralNet(number_of_input_features=train_dataset.get_number_of_input_features(),
                                        number_of_target_variables=train_dataset.get_number_of_target_variables(),
                                        number_of_layers=self.args.nn_layers,
                                        number_of_units=self.args.nn_units)
                model_wrapper = PytorchNeuralNetModelWrapper(model, self.model_type, self.args)

        else:  # Transformer models
            # convert the raw data into the preprocessed datasets
            train_dataset = TransformerDataset(
                train, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                self.forecasting_horizon, self.args.transformer_labels_count,
                self.predict_single_value, self.include_time_context, scaler, True)
            validation_dataset = TransformerDataset(
                validation, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                self.forecasting_horizon, self.args.transformer_labels_count,
                self.predict_single_value, self.include_time_context, scaler, False)
            test_dataset = TransformerDataset(
                test, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                self.forecasting_horizon, self.args.transformer_labels_count,
                self.predict_single_value, self.include_time_context, scaler, False)

            # differentiate between the three available transformer models
            if self.model_type == ModelType.TimeSeriesTransformer:
                model = TimeSeriesTransformer(
                    d_model=self.args.transformer_d_model,
                    input_features_count=self.args.transformer_input_features_count,
                    num_encoder_layers=self.args.transformer_num_encoder_layers,
                    num_decoder_layers=self.args.transformer_num_decoder_layers,
                    dim_feedforward=self.args.transformer_dim_feedforward,
                    dropout=self.args.transformer_dropout,
                    attention_heads=self.args.transformer_attention_heads)
            elif self.model_type == ModelType.TimeSeriesTransformerWithConvolutionalAttention:
                model = TimeSeriesTransformerWithConvolutionalAttention(
                    d_model=self.args.transformer_d_model,
                    input_features_count=self.args.transformer_input_features_count,
                    num_encoder_layers=self.args.transformer_num_encoder_layers,
                    num_decoder_layers=self.args.transformer_num_decoder_layers,
                    dim_feedforward=self.args.transformer_dim_feedforward,
                    kernel_size=self.args.conv_transformer_kernel_size,
                    dropout=self.args.transformer_dropout,
                    attention_heads=self.args.transformer_attention_heads)
            else:
                model = Informer(input_features_count=self.args.transformer_input_features_count,
                                 d_model=self.args.transformer_d_model,
                                 d_ff=self.args.transformer_dim_feedforward,
                                 e_layers=self.args.transformer_num_encoder_layers,
                                 d_layers=self.args.transformer_num_decoder_layers,
                                 n_heads=self.args.transformer_attention_heads,
                                 dropout=self.args.transformer_dropout,
                                 attn='full')

            model_wrapper = PytorchTransformerModelWrapper(model, self.model_type, self.args)

        if run_experiment:
            # train the model
            training_start_time = time.time()
            training_report = model_wrapper.train(train_dataset, validation_dataset)
            training_time = time.time() - training_start_time

            # evaluate the model on the test data
            test_start_time = time.time()
            test_outputs, test_targets = model_wrapper.predict(test_dataset)
            test_time = time.time() - test_start_time
            time_labels: np.ndarray = test_dataset.time_labels
            evaluator = Evaluator(test_outputs, test_targets, time_labels, scaler, self.forecasting_horizon)
            evaluation = evaluator.evaluate()

            self.experiment = Experiment(model_wrapper, evaluation, self.args, training_report, training_time,
                                         test_time)
            print(str(self.experiment))
            if model is not None:
                self.save_model(model, scaler)
        else:
            self.model = model

    def save_to_file(self):
        """
        Saves the completed experiment (pipeline has been run once) to a file.
        """
        self.experiment.save_to_json_file()

    def save_model(self, model, scaler):
        if self.args.model_name is None:
            return
        path = "data/models/" + self.args.model_name
        torch.save(model, path)
        with open(path + ".scaler", "wb") as f:
            pickle.dump(scaler, f)
        print(f"model saved at {path}")
