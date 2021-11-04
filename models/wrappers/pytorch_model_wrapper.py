from abc import ABC

import torch

from models.wrappers.base_model_wrapper import BaseModelWrapper


class PytorchModelWrapper(BaseModelWrapper, ABC):

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
