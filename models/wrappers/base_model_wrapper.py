from abc import ABC, abstractmethod

import torch

from models.model_type import ModelType


class BaseModelWrapper(ABC):

    def __init__(self, model_type: ModelType):
        self.model_type = model_type

    @abstractmethod
    def predict(self, inputs) -> torch.Tensor:
        pass
