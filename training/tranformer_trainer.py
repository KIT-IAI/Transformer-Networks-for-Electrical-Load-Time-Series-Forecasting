from abc import ABC

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from training.trainer import Trainer


class TransformerTrainer(Trainer, ABC):
    """
    Is a Trainer, which can be used to train transformers.
    """

    def __init__(self, train_data_loader: DataLoader, validation_data_loader: DataLoader, model: nn.Module,
                 loss_criterion, optimizer, epochs_count: int, learning_rate_scheduler: StepLR, args):
        """
        Creates a TransformerTrainer.

        :param train_data_loader:       loads the training data
        :param validation_data_loader:  loads the validation data
        :param model:                   the model to train
        :param loss_criterion:          can be any criterion to measure the loss of predictions
        :param optimizer:               the algorithm to optimize the weights
        :param epochs_count:            how many epochs are executed
        """
        super().__init__(train_data_loader, validation_data_loader, model, loss_criterion, optimizer, epochs_count,
                         learning_rate_scheduler, args)

    def train_phase(self, device) -> float:
        self.model.train()
        total_training_loss = 0.0
        for encoder_input, decoder_input in self.train_data_loader:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            self.optimizer.zero_grad()
            predicted, expected = self.execute_model_on_batch(encoder_input, decoder_input, device)
            training_loss = self.loss_criterion(predicted, expected)
            training_loss.backward()
            self.optimizer.step()
            total_training_loss += training_loss.item()

        return total_training_loss / len(self.train_data_loader)

    def validation_phase(self, device) -> float:
        self.model.eval()
        total_validation_loss = 0.0
        with torch.no_grad():
            for batch_index, (encoder_input, decoder_input) in enumerate(self.validation_data_loader):
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)

                predicted, expected = self.execute_model_on_batch(encoder_input, decoder_input, device)
                validation_loss = self.loss_criterion(predicted, expected)

                if batch_index % 30 == 0:
                    print(batch_index, validation_loss)
                    plt.plot(predicted.to('cpu').detach().numpy()[:, 1], label='pred')
                    plt.plot(expected.to('cpu').detach().numpy()[:, 1], label='expected')
                    plt.legend(loc='upper left')
                    plt.show()
                total_validation_loss += validation_loss.item()

        return total_validation_loss / len(self.validation_data_loader)

    def execute_model_on_batch(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor,
                               device: str) -> [torch.Tensor, torch.Tensor]:
        batch_size = encoder_input.shape[0]
        expected = decoder_input[:, self.args.transformer_labels_count:, 0]
        u = decoder_input[:, :, 1:]
        o1 = decoder_input[:, :self.args.transformer_labels_count, 0:1]
        o2 = torch.zeros([batch_size, self.args.forecasting_horizon, 1]).to(device)
        adjusted_decoder_input = torch.cat([torch.cat([o1, o2], dim=1), u], dim=2).to(device)
        predicted = torch.reshape(self.model(encoder_input, adjusted_decoder_input),
                                  torch.Size([batch_size,
                                              self.args.transformer_labels_count + self.args.forecasting_horizon]))
        target_predicted = predicted[:, self.args.transformer_labels_count:]
        return target_predicted, expected


def create_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask \
        .float() \
        .masked_fill(mask == 0, float('-inf')) \
        .masked_fill(mask == 1, float(0.0))
    return mask
