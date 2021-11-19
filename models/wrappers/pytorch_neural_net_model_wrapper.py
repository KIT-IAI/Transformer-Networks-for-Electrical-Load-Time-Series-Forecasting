from abc import ABC

import torch
from torch.nn import L1Loss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, SequentialSampler

from data_loading.standard_dataset import StandardDataset
from models.model_type import ModelType
from models.wrappers.base_model_wrapper import BaseModelWrapper
from training.neural_net_trainer import NeuralNetTrainer
from training.trainer import TrainingReport


class PytorchNeuralNetModelWrapper(BaseModelWrapper, ABC):

    def __init__(self, model: torch.nn.Module, model_type: ModelType, args):
        super().__init__(model_type, args)
        self.model = model

    def train(self, train_dataset: StandardDataset, validation_dataset: StandardDataset = None) -> TrainingReport:
        train_dl = DataLoader(train_dataset, batch_size=80, sampler=SequentialSampler(train_dataset))
        validation_dl = DataLoader(validation_dataset, batch_size=80, sampler=SequentialSampler(validation_dataset))

        criterion = L1Loss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = StepLR(optimizer, self.args.learning_rate_scheduler_step,
                           self.args.learning_rate_scheduler_gamma)
        trainer = NeuralNetTrainer(train_dl, validation_dl, self.model, criterion, optimizer,
                                   self.args.max_epochs, scheduler, self.args.use_early_stopping,
                                   self.args.early_stopping_patience)
        return trainer.train()

    def predict(self, dataset: StandardDataset) -> (torch.Tensor, torch.Tensor):
        device = 'cpu'
        return self.model(dataset.prepared_time_series_input).to(device).detach(), \
            dataset.prepared_time_series_target.to(device).detach()

    def __str__(self):
        return str(self.model)
