import argparse

import torch

from pipeline import Pipeline, ModelType


def parse_arguments() -> argparse.Namespace:
    """
    Parses the arguments, which are needed for the pipeline initialization.

    :return: an object containing the arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=False, default='SimpleNeuralNet',
                        choices=['LinearRegression', 'SimpleNeuralNet', 'TimeSeriesTransformer',
                                 'TimeSeriesTransformerWithConvolutionalAttention', 'Informer'],
                        help="Determines which model is executed.")
    parser.add_argument("--model_name", type=str, required=False, default=None)

    # problem specification
    parser.add_argument('--forecasting_horizon', type=int, required=False, default=96,
                        help="How far the prediction reaches.")
    parser.add_argument('--predict_single_value', type=bool, required=False, default=False,
                        help="Indicates whether a single value is the target.")
    parser.add_argument('--time_series_window', type=int, required=False, default=168,
                        help="How many historical values of the time series are used as input for a forecast.")

    ## NN and regression dataset settings
    parser.add_argument('--one_hot_time_variables', type=bool, required=False, default=False,
                        help="Whether to one-hot-encode the hour of the day and day of the week. "
                             "If false, cyclical encoding is used.")

    # general learning settings
    parser.add_argument('--include_time_context', type=bool, required=False, default=True,
                        help="Indicates whether the time information is used as additional input.")
    parser.add_argument('--learning_rate', type=float, required=False, default=0.0005,
                        help="The learning rate for PyTorch model-training.")
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--max_epochs', type=int, required=False, default=200,
                        help="The maximum of number of epochs which are executed.")

    # early stopping
    parser.add_argument('--use_early_stopping', type=bool, required=False, default=True,
                        help="Indicates whether the training should stop if the loss does not decreases.")
    parser.add_argument('--early_stopping_patience', type=bool, required=False, default=5,
                        help="The allowed number of epochs with no loss decrease.")

    # learning rate scheduling
    parser.add_argument('--learning_rate_scheduler_step', type=int, required=False, default=2)
    parser.add_argument('--learning_rate_scheduler_gamma', type=float, required=False, default=0.1)

    # simple NN setting
    parser.add_argument("--nn_layers", type=int, required=False, default=2)
    parser.add_argument("--nn_units", type=int, required=False, default=2048)

    # transformer setting
    parser.add_argument('--transformer_d_model', type=int, required=False, default=160)
    parser.add_argument('--transformer_input_features_count', type=int, required=False, default=10)
    parser.add_argument('--transformer_num_encoder_layers', type=int, required=False, default=3)
    parser.add_argument('--transformer_num_decoder_layers', type=int, required=False, default=3)
    parser.add_argument('--transformer_dim_feedforward', type=int, required=False, default=160)
    parser.add_argument('--transformer_dropout', type=float, required=False, default=0.01)
    parser.add_argument('--transformer_attention_heads', type=int, required=False, default=8)
    parser.add_argument('--transformer_labels_count', type=int, required=False, default=24)
    parser.add_argument('--transformer_use_teacher_forcing', type=bool, required=False, default=False)
    parser.add_argument('--transformer_use_auto_regression', type=bool, required=False, default=False)
    parser.add_argument('--conv_transformer_kernel_size', type=int, required=False, default=3)

    # other
    parser.add_argument('--seed', type=int, required=False, default=0)
    return parser.parse_args()


def set_seed(seed) -> None:
    """
    Sets the random seed for pytorch.
    """
    torch.manual_seed(seed)


def main() -> None:
    """
    Starts the program by parsing the arguments and initializing the pipeline.
    The result of the pipeline is saved.
    """
    arguments = parse_arguments()
    set_seed(arguments.seed)
    pipeline = Pipeline(ModelType[arguments.model], arguments)
    pipeline.start()
    pipeline.save_to_file()


if __name__ == '__main__':
    main()
