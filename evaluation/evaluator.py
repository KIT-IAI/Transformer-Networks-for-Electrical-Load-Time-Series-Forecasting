import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler

from data_preparation.electricity_load_time_series import PreparedDataset
from models.wrappers.base_model_wrapper import BaseModelWrapper
from utils.losses import calculate_mase_loss, calculate_mape_loss


class Evaluator:
    """
    Provides the possibility to evaluate trained models on test data (or validation data after training)
    """

    def __init__(self, trained_model: BaseModelWrapper, prepared_dataset: PreparedDataset, scaler: StandardScaler):
        self.trained_model = trained_model
        self.prepared_dataset = prepared_dataset
        self.scaler = scaler

    def evaluate(self):
        inputs = torch.Tensor(self.prepared_dataset.inputs)
        targets = torch.Tensor(self.prepared_dataset.outputs)

        output = self.trained_model.predict(inputs).detach().numpy()

        unscaled_output = self.scaler.inverse_transform(output)
        expected_unscaled_output = self.scaler.inverse_transform(targets)

        # calculate the mean absolute percentage loss on the rescaled predictions
        mape_loss = calculate_mape_loss(torch.Tensor(unscaled_output),
                                        torch.Tensor(expected_unscaled_output))
        print('MAPE loss:', mape_loss)

        # calculate the mean absolute scaled error
        mase_loss = calculate_mase_loss(torch.Tensor(unscaled_output),
                                        torch.Tensor(expected_unscaled_output),
                                        168)
        print('MASE loss:', mase_loss)

        # visualize the predictions
        plt.plot(unscaled_output)
        plt.plot(expected_unscaled_output)
        plt.show()

        if self.prepared_dataset.get_number_of_target_variables() > 1:
            # analyse the error depending on the variable
            mape_losses = []
            for index in range(0, self.prepared_dataset.get_number_of_target_variables()):
                t1 = torch.Tensor(expected_unscaled_output[:, index])
                t2 = torch.Tensor(unscaled_output[:, index])
                mape_loss = calculate_mape_loss(t2, t1)
                mape_losses.append(mape_loss)
            plt.plot(mape_losses)
            plt.show()
            print(mape_losses)
