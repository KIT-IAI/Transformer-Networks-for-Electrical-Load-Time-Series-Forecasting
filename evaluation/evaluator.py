from typing import Dict

import torch
from sklearn.preprocessing import StandardScaler

from utils.losses import calculate_mase_loss, calculate_mape_loss


class Evaluation:

    def __init__(self, total_mape_loss: float, total_mase_loss: float,
                 mape_losses_by_prediction_variable: Dict[str, float],
                 mase_losses_by_prediction_variable: Dict[str, float]):
        self.total_mape_loss = total_mape_loss
        self.total_mase_loss = total_mase_loss
        self.mape_losses_by_prediction_variable = mape_losses_by_prediction_variable
        self.mase_losses_by_prediction_variable = mase_losses_by_prediction_variable

    def __str__(self):
        return str(self.__dict__)


class Evaluator:
    """
    Provides the possibility to evaluate trained models on test data (or validation data after training)
    """

    def __init__(self, model_outputs: torch.Tensor, targets: torch.Tensor, scaler: StandardScaler,
                 number_of_target_variables: int):
        self.model_outputs = model_outputs
        self.targets = targets
        self.scaler = scaler
        self.number_of_target_variables = number_of_target_variables

    def evaluate(self) -> Evaluation:
        """
        :return: the evaluation
        """
        unscaled_output = self.scaler.inverse_transform(self.model_outputs)
        expected_unscaled_output = self.scaler.inverse_transform(self.targets)

        # calculate the mean absolute percentage loss on the rescaled predictions
        mape_loss = calculate_mape_loss(torch.Tensor(unscaled_output), torch.Tensor(expected_unscaled_output)).item()

        # calculate the mean absolute scaled error
        mase_loss = calculate_mase_loss(
            torch.Tensor(unscaled_output), torch.Tensor(expected_unscaled_output), 168).item()

        mape_losses_by_prediction_variable: Dict[str, float] = dict()
        mase_losses_by_prediction_variable: Dict[str, float] = dict()
        if self.number_of_target_variables > 1:
            for index in range(0, self.number_of_target_variables):
                t1 = torch.Tensor(expected_unscaled_output[:, index])
                t2 = torch.Tensor(unscaled_output[:, index])
                mape_losses_by_prediction_variable[str(index)] = calculate_mape_loss(t2, t1).item()
                mase_losses_by_prediction_variable[str(index)] = calculate_mase_loss(t2, t1, 168).item()

        return Evaluation(mape_loss, mase_loss, mape_losses_by_prediction_variable,
                          mase_losses_by_prediction_variable)
