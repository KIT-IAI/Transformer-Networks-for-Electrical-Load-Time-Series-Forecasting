import copy
from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


class TrainingEpoch:

    def __init__(self, epoch_number, training_loss: float, validation_loss):
        self.epoch_number = epoch_number
        self.training_loss = training_loss
        self.validation_loss = validation_loss

    def __str__(self):
        return str(self.__dict__)

    def serialize(self):
        return {
            'epochNumber': self.epoch_number,
            'trainingLoss': self.training_loss,
            'validationLoss': self.validation_loss
        }


class TrainingReport:

    def __init__(self, epochs: List[TrainingEpoch]):
        self.epochs = epochs

    def __str__(self):
        return str(
            {
                'lossCriterion': 'MAE',
                'optimizer': 'AdamW',
                'epochs': [epoch.serialize() for epoch in self.epochs]
            }
        )

    def serialize(self):
        return {
            'lossCriterion': 'MAE',
            'optimizer': 'AdamW',
            'epochs': [epoch.serialize() for epoch in self.epochs]
        }


class Trainer(ABC):
    """
    Provides the functionality to train a given model. The process uses cuda if possible, so a GPU with cuda
    compatability should be available for faster training.
    """

    def __init__(self, train_data_loader: DataLoader, validation_data_loader: DataLoader, model: nn.Module,
                 loss_criterion, optimizer, epochs_count: int, learning_rate_scheduler: StepLR, args):
        """
        Creates a Trainer.

        :param train_data_loader:       loads the training data
        :param validation_data_loader:  loads the validation data
        :param model:                   the model to train
        :param loss_criterion:          can be any criterion to measure the loss of predictions
        :param optimizer:               the algorithm to optimize the weights
        :param epochs_count:            how many epochs are executed
        """
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.epochs_count = epochs_count
        self.learning_rate_scheduler = learning_rate_scheduler
        self.args = args
        self.best_model_state = {}

    def train(self) -> TrainingReport:
        """
        Starts the training process, which runs n epochs.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Used device: ', device)
        self.model = self.model.to(device)

        epochs: List[TrainingEpoch] = []

        epochs_without_validation_loss_decrease = 0
        minimum_average_validation_loss = float('inf')
        for epoch in range(self.epochs_count):
            # training phase
            training_loss = self.train_phase(device)

            # validation phase
            validation_loss = self.validation_phase(device)

            self.learning_rate_scheduler.step()

            if self.args.use_early_stopping:
                if minimum_average_validation_loss <= validation_loss:
                    epochs_without_validation_loss_decrease += 1
                else:
                    epochs_without_validation_loss_decrease = 0
                    minimum_average_validation_loss = validation_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())

                if epochs_without_validation_loss_decrease > self.args.early_stopping_patience:
                    print('Early stopping has happened at epoch', epoch)
                    break

            print('Epoch: ', epoch)
            print('Average training loss: ', training_loss)
            print('Average validation loss: ', validation_loss)

            epochs.append(TrainingEpoch(epoch, training_loss, validation_loss))

        device = 'cpu'
        self.model.load_state_dict(self.best_model_state)  # use the best model
        self.model = self.model.to(device)

        return TrainingReport(epochs)

    @abstractmethod
    def train_phase(self, device: str) -> float:
        """
        Executes the training phase for one epoch.

        :param device: 'cuda' or 'cpu' the preferred way is to use 'cuda' so that the gpu can be used
        :return: the average training loss for the epoch
        """
        pass

    @abstractmethod
    def validation_phase(self, device: str) -> float:
        """
        Executes the validation phase for one epoch.

        :param device: 'cuda' or 'cpu' the preferred way is to use 'cuda' so that the gpu can be used
        :return: the average validation loss for the epoch
        """
        pass
