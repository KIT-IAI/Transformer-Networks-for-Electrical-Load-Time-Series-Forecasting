import json
from typing import Dict

import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler

from data_preparation.electricity_load_time_series import PreparedDataset
from models.wrappers.base_model_wrapper import BaseModelWrapper
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

    def __init__(self, trained_model: BaseModelWrapper, prepared_dataset: PreparedDataset, scaler: StandardScaler):
        self.trained_model = trained_model
        self.prepared_dataset = prepared_dataset
        self.scaler = scaler

    def evaluate(self) -> Evaluation:
        """
        :return: the evaluation
        """
        inputs = torch.Tensor(self.prepared_dataset.inputs)
        targets = torch.Tensor(self.prepared_dataset.outputs)

        output = self.trained_model.predict(inputs).detach().numpy()

        unscaled_output = self.scaler.inverse_transform(output)
        expected_unscaled_output = self.scaler.inverse_transform(targets)

        # calculate the mean absolute percentage loss on the rescaled predictions
        mape_loss = calculate_mape_loss(torch.Tensor(unscaled_output), torch.Tensor(expected_unscaled_output)).item()

        # calculate the mean absolute scaled error
        mase_loss = calculate_mase_loss(
            torch.Tensor(unscaled_output), torch.Tensor(expected_unscaled_output), 168).item()

        mape_losses_by_prediction_variable: Dict[str, float] = dict()
        mase_losses_by_prediction_variable: Dict[str, float] = dict()
        if self.prepared_dataset.get_number_of_target_variables() > 1:
            for index in range(0, self.prepared_dataset.get_number_of_target_variables()):
                t1 = torch.Tensor(expected_unscaled_output[:, index])
                t2 = torch.Tensor(unscaled_output[:, index])
                mape_losses_by_prediction_variable[str(index)] = calculate_mape_loss(t2, t1).item()
                mase_losses_by_prediction_variable[str(index)] = calculate_mase_loss(t2, t1, 168).item()

        return Evaluation(mape_loss, mase_loss, mape_losses_by_prediction_variable,
                          mase_losses_by_prediction_variable)
