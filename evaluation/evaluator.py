import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

import models.mape_loss as mape
from data_preparation.electricity_load_time_series import PreparedDataset


class Evaluator:
    """
    Provides the possibility to evaluate trained models on test data (or validation data after training)
    """

    def __init__(self, trained_model: nn.Module, prepared_dataset: PreparedDataset, scaler: StandardScaler):
        self.trained_model = trained_model
        self.prepared_dataset = prepared_dataset
        self.scaler = scaler

    def evaluate(self):
        device = 'cpu'
        print('Used device: ', device)
        self.trained_model = self.trained_model.to(device)

        with torch.no_grad():
            inputs = torch.Tensor(self.prepared_dataset.inputs)
            targets = torch.Tensor(self.prepared_dataset.outputs)

            output = self.trained_model(inputs)

            unscaled_output = self.scaler.inverse_transform(output)
            expected_unscaled_output = self.scaler.inverse_transform(targets)

            # calculate the mean absolute percentage loss on the rescaled predictions
            mape_loss = mape.calculate_mape_loss(torch.Tensor(unscaled_output),
                                                 torch.Tensor(expected_unscaled_output))
            print(mape_loss)

            # visualize the predictions
            plt.plot(unscaled_output)
            plt.plot(expected_unscaled_output)
            plt.show()

            if self.prepared_dataset.get_number_of_target_variables() > 1:
                # analyse the error depending on the variable
                mape_losses = []
                for index in range(0,self.prepared_dataset.get_number_of_target_variables()):
                    t1 = torch.Tensor(expected_unscaled_output[:, index])
                    t2 = torch.Tensor(unscaled_output[:, index])
                    mape_loss = mape.calculate_mape_loss(t1, t2)
                    mape_losses.append(mape_loss)
                plt.plot(mape_losses)
                plt.show()
                print(mape_losses)
