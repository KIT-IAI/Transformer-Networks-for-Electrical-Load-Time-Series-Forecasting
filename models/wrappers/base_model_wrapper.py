from abc import ABC, abstractmethod

import torch


class BaseModelWrapper(ABC):

    @abstractmethod
    def predict(self, inputs) -> torch.Tensor:
        pass
