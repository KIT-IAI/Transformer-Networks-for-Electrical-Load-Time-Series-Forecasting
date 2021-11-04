import torch


def calculate_mape_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean-absolute-percentage-loss. This loss criterion can be usd for the better comparison of result in
    the case of time-series forecasting.

    :param output: the tensor with predictions
    :param target: the tensor which is expected
    :return: the mape metric
    """
    return torch.mean(torch.abs((target - output) / target))


def calculate_mae_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean-absolute-error (L1 norm).

    :param output: the tensor with predictions
    :param target: the tensor which is expected
    :return: the mae metric
    """
    return torch.mean(torch.abs((target - output)))


def calculate_naive_forecast_loss(time_series: torch.Tensor, seasonal_cycle: int) -> float:
    """
    Calculates the naive forecast loss. The is the loss, which uses the average of the absolute deviations, from
    predictions made by simply using the value before n time-steps.

    :param time_series: the values on which the forecast is made
    :param seasonal_cycle: the number of steps after a seasonal pattern starts again
    :return: the naive forecasting loss
    """
    absolute_deviation_sum = 0
    for index in range(seasonal_cycle, len(time_series)):
        absolute_deviation_sum += torch.abs(time_series[index] - time_series[index - seasonal_cycle])
    return absolute_deviation_sum / (len(time_series) - seasonal_cycle)


def calculate_mase_loss(output: torch.Tensor, target: torch.Tensor, seasonal_cycle: int) -> float:
    """
    Calculates the mean-absolute-scaled-error. This loss compares the MAE to the naive forecasting loss.
    If the loss (the division of both) is less than 1, then the model outperforms the naive model. Else the model should
    be rejected, because the model has a lower prediction accuracy.
    The naive forecasting error is calculated on the target values.

    :param output: the tensor with predictions
    :param target: the tensor which is expected
    :param seasonal_cycle: the number of steps after a seasonal pattern starts again
    :return: the mase
    """
    mae_loss = calculate_mae_loss(output, target)
    naive_forecast_loss = calculate_naive_forecast_loss(target, seasonal_cycle)
    print('MAE:', mae_loss)
    print('Naive Forecast loss', naive_forecast_loss)
    return mae_loss / naive_forecast_loss
