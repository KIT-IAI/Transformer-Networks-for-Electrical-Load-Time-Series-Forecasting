import argparse
from abc import ABC

import torch

from data_loading.standard_dataset import StandardDataset
from models.model_type import ModelType
from models.wrappers.base_model_wrapper import BaseModelWrapper


class SklearnModelWrapper(BaseModelWrapper, ABC):
    """
    Is a wrapper for sklearn models to train them and predict.
    """

    def __init__(self, model, model_type: ModelType, args: argparse.Namespace):
        super().__init__(model_type, args)
        self.model = model

    def train(self, train_dataset: StandardDataset, validation_dataset: StandardDataset = None) -> None:
        self.model.fit(train_dataset.prepared_time_series_input, train_dataset.prepared_time_series_target)
        return None

    def predict(self, dataset: StandardDataset) -> (torch.Tensor, torch.Tensor):
        return self.model.predict(dataset.prepared_time_series_input), dataset.prepared_time_series_target

    def __str__(self):
        return str(self.model)
