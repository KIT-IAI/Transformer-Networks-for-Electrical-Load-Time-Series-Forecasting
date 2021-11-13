import argparse

from pipeline import Pipeline, ModelType
from training.training_config import TrainingConfig


def parse_arguments():
    """
    Parses the arguments, which are needed for the pipeline initialization.

    :return: an object containing the arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=False, default='SimpleNeuralNet',
                        choices=['LinearRegression', 'SimpleNeuralNet'], help="Determines which model is executed.")
    parser.add_argument('--forecasting_horizon', type=int, required=False, default=24,
                        help="How far the prediction reaches.")
    parser.add_argument('--predict_single_value', type=bool, required=False, default=False,
                        help="Indicates whether a single value is the target.")
    parser.add_argument('--time_series_window', type=int, required=False, default=168,
                        help="How many historical values of the time series are used as input for a forecast.")
    parser.add_argument('--include_time_context', type=bool, required=False, default=True,
                        help="Indicates whether the time information is used as additional input.")
    parser.add_argument('--learning_rate', type=float, required=False, default=0.0002,
                        help="The learning rate for PyTorch model-training.")
    parser.add_argument('--max_epochs', type=int, required=False, default=2000,
                        help="The maximum of number of epochs which are executed.")
    parser.add_argument('--use_early_stopping', type=bool, required=False, default=True,
                        help="Indicates whether the training should stop if the loss does not decreases.")
    parser.add_argument('--early_stopping_patience', type=bool, required=False, default=10,
                        help="The allowed number of epochs with no loss decrease.")

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    training_config = TrainingConfig(arguments.learning_rate, arguments.max_epochs, arguments.use_early_stopping,
                                     arguments.early_stopping_patience)
    pipeline = Pipeline(ModelType[arguments.model], arguments.forecasting_horizon, arguments.predict_single_value,
                        arguments.time_series_window, arguments.include_time_context, training_config)
    pipeline.start()
    pipeline.save_to_file()


if __name__ == '__main__':
    main()
