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
                 loss_criterion, optimizer, epochs_count: int, learning_rate_scheduler: StepLR,
                 use_early_stopping: bool, early_stopping_patience: int = 0):
        """
        Creates a TransformerTrainer.

        :param train_data_loader:       loads the training data
        :param validation_data_loader:  loads the validation data
        :param model:                   the model to train
        :param loss_criterion:          can be any criterion to measure the loss of predictions
        :param optimizer:               the algorithm to optimize the weights
        :param epochs_count:            how many epochs are executed
        :param use_early_stopping:      whether the training process should stop if the validation does not decreases
        :param early_stopping_patience: how many epochs the validation loss has to increase to execute early stopping
        """
        super().__init__(train_data_loader, validation_data_loader, model, loss_criterion, optimizer, epochs_count,
                         learning_rate_scheduler, use_early_stopping, early_stopping_patience)

    def train_phase(self, device) -> float:
        self.model.train()
        total_training_loss = 0.0
        for encoder_input, decoder_input in self.train_data_loader:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            self.optimizer.zero_grad()
            expected = decoder_input[:, :, 0]
            src_mask = create_mask(168).to(device)
            target_mask = create_mask(24).to(device)
            decoder_input = torch.cat([encoder_input[:, 167:168, :], decoder_input[:, :23, :]], dim=1)
            predicted = torch.reshape(self.model(encoder_input, decoder_input, src_mask, target_mask), expected.shape)
            training_loss = self.loss_criterion(predicted, expected)

            training_loss.backward()
            self.optimizer.step()
            total_training_loss += training_loss.item()
            # plt.plot(predicted.to('cpu').detach().numpy()[0])
            # plt.plot(expected.to('cpu').detach().numpy()[0])
            # plt.show()

        return total_training_loss / len(self.train_data_loader)

    def validation_phase(self, device) -> float:
        self.model.eval()
        total_validation_loss = 0.0
        with torch.no_grad():
            for batch_index, (encoder_input, decoder_input) in enumerate(self.validation_data_loader):
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                expected = decoder_input[:, :, 0].to(device)

                start_decoder_input = encoder_input[:, 167:168, :].to(device)
                for i in range(0, 24):
                    predicted = self.model(encoder_input, start_decoder_input).to(device)
                    known_decoder_input = decoder_input[:, i:i + 1, 1:].to(device)
                    new_predicted = predicted[:, i:i+1, 0:1].to(device)
                    predicted = torch.cat([new_predicted, known_decoder_input], dim=2).to(device)
                    start_decoder_input = torch.cat([start_decoder_input[:, :, :], predicted], dim=1).to(device)
                total_prediction = start_decoder_input[:, 1:, 0].to(device)
                training_loss = self.loss_criterion(total_prediction, expected)

                print(batch_index, training_loss)
                plt.plot(total_prediction.to('cpu').detach().numpy()[900:, 5], label='pred')
                plt.plot(expected.to('cpu').detach().numpy()[900:, 5], label='expected')
                plt.legend(loc='upper left')
                plt.show()
                total_validation_loss += training_loss.item()

        return total_validation_loss / len(self.validation_data_loader)


def create_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask \
        .float() \
        .masked_fill(mask == 0, float('-inf')) \
        .masked_fill(mask == 1, float(0.0))
    return mask
