import datetime
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from data_loading.standard_dataset import StandardDataset
from data_loading.time_series import TimeSeriesDataframeLoader, TimeInterval
from data_loading.transformer_dataset import TransformerDataset
from evaluation.evaluator import Evaluator
from experiments.experiment import Experiment
from models.model_type import ModelType
from models.simple_nn import SimpleNeuralNet
from models.tranformers.informer import Informer, InformerStack
from models.tranformers.transformer import TimeSeriesTransformer
from models.tranformers.transformer_convolution_attention import TimeSeriesTransformerWithConvolutionalAttention
from models.wrappers.base_model_wrapper import BaseModelWrapper
from models.wrappers.pytorch_neural_net_model_wrapper import PytorchNeuralNetModelWrapper
from models.wrappers.pytorch_transformer_model_wrapper import PytorchTransformerModelWrapper
from models.wrappers.sklearn_model_wrapper import SklearnModelWrapper

UTC_TIMESTAMP = 'utc_timestamp'
TARGET_VARIABLE = 'DE_transnetbw_load_actual_entsoe_transparency'
PATH_TO_CSV = os.path.join('data', 'opsd', 'time_series_15min_singleindex.csv')


class Pipeline:

    def __init__(self, model_type: ModelType, args):
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

        self.args = args

        self.experiment = None

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

        model_wrapper: BaseModelWrapper
        train_dataset: Dataset
        validation_dataset: Dataset
        if self.model_type == ModelType.LinearRegression or self.model_type == ModelType.SimpleNeuralNet:
            train_dataset: StandardDataset = StandardDataset(train, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                                                             self.forecasting_horizon, self.predict_single_value,
                                                             self.include_time_context, scaler, True)
            validation_dataset: StandardDataset = StandardDataset(validation, UTC_TIMESTAMP, TARGET_VARIABLE,
                                                                  self.window_length, self.forecasting_horizon,
                                                                  self.predict_single_value, self.include_time_context,
                                                                  scaler, False)
            if self.model_type == ModelType.LinearRegression:
                linear_regression = LinearRegression()
                model_wrapper = SklearnModelWrapper(linear_regression, self.model_type, self.args)

            else:
                model = SimpleNeuralNet(train_dataset.get_number_of_input_features(),
                                        train_dataset.get_number_of_target_variables())
                model_wrapper = PytorchNeuralNetModelWrapper(model, self.model_type, self.args)
        elif self.model_type == ModelType.TimeSeriesTransformer \
                or self.model_type == ModelType.TimeSeriesTransformerWithConvolutionalAttention \
                or self.model_type == ModelType.Informer:
            train_dataset = TransformerDataset(
                train, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                self.forecasting_horizon, self.args.transformer_labels_count,
                self.predict_single_value, self.include_time_context, scaler, True)
            validation_dataset = TransformerDataset(
                validation, UTC_TIMESTAMP, TARGET_VARIABLE, self.window_length,
                self.forecasting_horizon, self.args.transformer_labels_count,
                self.predict_single_value, self.include_time_context, scaler, False)

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
                    dropout=self.args.transformer_dropout,
                    attention_heads=self.args.transformer_attention_heads)
            else:
                model = Informer(enc_in=1, dec_in=1, c_out=1, seq_len=self.args.time_series_window,
                                 d_model=self.args.transformer_d_model, d_ff=self.args.transformer_dim_feedforward,
                                 e_layers=self.args.transformer_num_encoder_layers,
                                 d_layers=self.args.transformer_num_decoder_layers,
                                 n_heads=self.args.transformer_attention_heads,
                                 dropout=self.args.transformer_dropout,
                                 label_len=self.args.transformer_labels_count, out_len=self.args.forecasting_horizon)

            model_wrapper = PytorchTransformerModelWrapper(model, self.model_type, self.args)

        else:
            print(self.model_type)
            print(type(self.model_type))
            raise Exception('Received a model, which is not included.')

        training_report = model_wrapper.train(train_dataset, validation_dataset)
        validation_outputs, validation_targets = model_wrapper.predict(validation_dataset)

        time_labels: np.ndarray = validation_dataset.time_labels
        evaluator = Evaluator(validation_outputs, validation_targets, time_labels, scaler, self.forecasting_horizon)
        evaluation = evaluator.evaluate()

        self.experiment = Experiment(model_wrapper, evaluation, self.args, training_report)
        print(str(self.experiment))

    def save_to_file(self):
        self.experiment.save_to_json_file()
