from abc import ABC

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from training.trainer import Trainer


class NeuralNetTrainer(Trainer, ABC):
    """
    Is a Trainer, which can be used to train neural-nets.
    """

    def __init__(self, train_data_loader: DataLoader, validation_data_loader: DataLoader, model: nn.Module,
                 loss_criterion, optimizer, epochs_count: int, learning_rate_scheduler: StepLR, args):
        """
        Creates a NeuralNetTrainer.

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
        for inputs, targets in self.train_data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            self.optimizer.zero_grad()

            output = self.model(inputs)
            training_loss = self.loss_criterion(output, targets)
            training_loss.backward()
            self.optimizer.step()

            total_training_loss += training_loss.item()

        return total_training_loss / len(self.train_data_loader)

    def validation_phase(self, device) -> float:
        self.model.eval()
        total_validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.validation_data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                output = self.model(inputs)
                output = output.to(device)

                validation_loss = self.loss_criterion(output, targets)
                total_validation_loss += validation_loss.item()

        return total_validation_loss / len(self.validation_data_loader)
