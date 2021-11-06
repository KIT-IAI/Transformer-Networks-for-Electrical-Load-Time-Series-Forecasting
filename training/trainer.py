import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Provides the functionality to train a given model. The process uses cuda if possible, so a GPU with cuda
    compatability should be available for faster training.
    """

    def __init__(self, train_data_loader: DataLoader, validation_data_loader: DataLoader, model: nn.Module,
                 loss_criterion, optimizer, epochs_count: int, use_early_stopping: bool,
                 early_stopping_patience: int = 0):
        """
        Creates a Trainer.

        :param train_data_loader:       loads the training data
        :param validation_data_loader:  loads the validation data
        :param model:                   the model to train
        :param loss_criterion:          can be any criterion to measure the loss of predictions
        :param optimizer:               the algorithm to optimize the weights
        :param epochs_count:            how many epochs are executed
        :param use_early_stopping:      whether the training process should stop if the validation does not decreases
        :param early_stopping_patience: how many epochs the validation loss has to increase to execute early stopping
        """
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.epochs_count = epochs_count
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience

    def train(self) -> None:
        """
        Starts the training process, which runs n epochs.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Used device: ', device)
        self.model = self.model.to(device)

        epochs_without_validation_loss_decrease = 0
        previous_average_validation_loss = None
        for epoch in range(self.epochs_count):
            # training phase
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

            # validation phase
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

            average_training_loss = total_training_loss / len(self.train_data_loader)
            average_validation_loss = total_validation_loss / len(self.validation_data_loader)

            if self.use_early_stopping:
                if previous_average_validation_loss and previous_average_validation_loss <= average_validation_loss:
                    epochs_without_validation_loss_decrease += 1
                else:
                    epochs_without_validation_loss_decrease = 0

                if epochs_without_validation_loss_decrease > self.early_stopping_patience:
                    print('Early stopping has happened at epoch', epoch)
                    break

            previous_average_validation_loss = average_validation_loss
            print('Epoch: ', epoch)
            print('Average training loss: ', average_training_loss)
            print('Average validation loss: ', average_validation_loss)

        device = 'cpu'
        self.model = self.model.to(device)
