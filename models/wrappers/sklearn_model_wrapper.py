from abc import ABC

import torch

from models.model_type import ModelType
from models.wrappers.base_model_wrapper import BaseModelWrapper


class SklearnModelWrapper(BaseModelWrapper, ABC):

    def __init__(self, model, model_type: ModelType):
        super().__init__(model_type)
        self.model = model

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_as_np = inputs.detach().numpy()
        return torch.Tensor(self.model.predict(inputs_as_np))

    def __str__(self):
        return str(self.model)
