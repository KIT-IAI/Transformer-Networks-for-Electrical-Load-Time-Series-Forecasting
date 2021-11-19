from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

from models.model_type import ModelType
from training.trainer import TrainingReport


class BaseModelWrapper(ABC):

    def __init__(self, model_type: ModelType, args):
        self.model_type = model_type
        self.args = args

    @abstractmethod
    def predict(self, dataset: Dataset) -> (torch.Tensor, torch.Tensor):
        """
        Executes a prediction with the model
        :param dataset: contains the input and target data
        :return: (predicted, target)
        """
        pass

    @abstractmethod
    def train(self, train_dataset: Dataset, validation_dataset: Dataset = None) -> TrainingReport:
        """
        Trains the model.
        :param train_dataset: the dataset for the training of the model
        :param validation_dataset: the dataset for the validation step (is therefore optional)
        :return: a report of the training
        """
        pass
