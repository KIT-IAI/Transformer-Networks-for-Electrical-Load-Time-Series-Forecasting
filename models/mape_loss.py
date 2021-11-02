import torch
from torch import nn


def calculate_mape_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean-absolute-percentage-loss. This loss criterion can be usd for the better comparison of result in
    the case of time-series forecasting.

    :param output: the tensor with predictions
    :param target: the tensor which is expected
    :return: the mape metric
    """
    return torch.mean(torch.abs((target - output) / target))


class MAPELoss(nn.Module):
    """
    Module for the mean-absolute-percentage-loss. Should not be used on scaled data, i.e. on standardized data.
    """

    def __init__(self):
        super(MAPELoss, self).__init__()
        self.loss_criterion = calculate_mape_loss

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_criterion(output, target)
