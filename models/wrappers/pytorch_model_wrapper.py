from abc import ABC

import torch

from models.model_type import ModelType
from models.wrappers.base_model_wrapper import BaseModelWrapper


class PytorchModelWrapper(BaseModelWrapper, ABC):

    def __init__(self, model: torch.nn.Module, model_type: ModelType):
        super().__init__(model_type)
        self.model = model

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def __str__(self):
        return str(self.model)
