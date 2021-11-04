import torch
from torch import nn

from utils.losses import calculate_mape_loss


class MAPELoss(nn.Module):
    """
    Module for the mean-absolute-percentage-loss. Should not be used on scaled data, i.e. on standardized data.
    """

    def __init__(self):
        super(MAPELoss, self).__init__()
        self.loss_criterion = calculate_mape_loss

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_criterion(output, target)
