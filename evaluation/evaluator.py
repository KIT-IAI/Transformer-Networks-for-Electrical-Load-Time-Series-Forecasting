import datetime
from typing import Dict, List

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from utils.losses import calculate_mase_loss, calculate_mape_loss


class ExpectedPredictedComparisonElement:

    def __init__(self, predicted: np.ndarray, expected: np.ndarray, date: datetime.date):
        self.predicted = predicted
        self.expected = expected
        self.date = date

    def serialize(self):
        return {
            'predicted': self.predicted.tolist(),
            'expected': self.expected.tolist(),
            'date': str(self.date)
        }


class Evaluation:

    def __init__(self, total_mape_loss: float, total_mase_loss: float,
                 mape_losses_by_prediction_variable: Dict[str, float],
                 mase_losses_by_prediction_variable: Dict[str, float],
                 expected_predicted_comparison: List[ExpectedPredictedComparisonElement]):
        self.total_mape_loss = total_mape_loss
        self.total_mase_loss = total_mase_loss
        self.mape_losses_by_prediction_variable = mape_losses_by_prediction_variable
        self.mase_losses_by_prediction_variable = mase_losses_by_prediction_variable
        self.expected_predicted_comparison = expected_predicted_comparison

    def __str__(self):
        return str(self.__dict__)

    def serialize(self):
        return {
            'total_mape_loss': self.total_mape_loss,
            'total_mase_loss': self.total_mase_loss,
            'mape_losses_by_prediction_variable': self.mape_losses_by_prediction_variable,
            'mase_losses_by_prediction_variable': self.mase_losses_by_prediction_variable,
            'expected_predicted_comparison': [c.serialize() for c in self.expected_predicted_comparison]
        }


class Evaluator:
    """
    Provides the possibility to evaluate trained models on test data (or validation data after training)
    """

    def __init__(self, model_outputs: torch.Tensor, targets: torch.Tensor, time_labels: np.ndarray,
                 scaler: StandardScaler, number_of_target_variables: int):
        self.model_outputs = model_outputs
        self.targets = targets
        self.time_labels = time_labels
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

        expected_predicted_comparison = self._combine_to_expected_predicted_comparison(expected_unscaled_output,
                                                                                       unscaled_output)

        return Evaluation(mape_loss, mase_loss, mape_losses_by_prediction_variable,
                          mase_losses_by_prediction_variable, expected_predicted_comparison)

    def _combine_to_expected_predicted_comparison(self, expected: np.ndarray, predicted: np.ndarray):
        """
        :param expected: the expected values (a 2D Array [validation_set_length, # of predicted time steps])
        :param predicted: the predicted values (a 2D Array [validation_set_length, # of predicted time steps])
        :return: a combination of the expectation and prediction over each time step
        """
        assert len(expected) == len(predicted) == len(self.time_labels)
        expected_predicted_comparison: List[ExpectedPredictedComparisonElement] = []
        for i in range(0, len(expected)):
            expected_predicted_comparison\
                .append(ExpectedPredictedComparisonElement(predicted[i], expected[i], self.time_labels[i]))
        return expected_predicted_comparison
