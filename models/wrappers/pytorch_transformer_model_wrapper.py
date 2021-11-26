from abc import ABC

import torch
from torch.nn import L1Loss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from data_loading.transformer_dataset import TransformerDataset
from models.model_type import ModelType
from models.wrappers.base_model_wrapper import BaseModelWrapper
from training.trainer import TrainingReport
from training.tranformer_trainer import TransformerTrainer


class PytorchTransformerModelWrapper(BaseModelWrapper, ABC):

    def __init__(self, model: torch.nn.Module, model_type: ModelType, args):
        super().__init__(model_type, args)
        self.model = model

    def train(self, train_dataset: TransformerDataset, validation_dataset: TransformerDataset = None) -> TrainingReport:
        train_dl = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        validation_dl = DataLoader(validation_dataset, batch_size=self.args.batch_size)

        criterion = L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = StepLR(optimizer, self.args.learning_rate_scheduler_step, self.args.learning_rate_scheduler_gamma)

        trainer = TransformerTrainer(train_dl, validation_dl, self.model, criterion, optimizer, self.args.max_epochs,
                                     scheduler, self.args)
        return trainer.train()

    def predict(self, dataset: TransformerDataset) -> (torch.Tensor, torch.Tensor):
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size)

        self.model.eval()
        with torch.no_grad():
            device = 'cpu'
            output = torch.tensor([])
            target = torch.tensor([])
            for batch_index, (encoder_input, decoder_input) in enumerate(data_loader):
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)

                batch_size = encoder_input.shape[0]
                expected = decoder_input[:, self.args.transformer_labels_count:, 0]
                target = torch.cat([target, expected], dim=0)

                u = decoder_input[:, :, 1:]
                o1 = decoder_input[:, :self.args.transformer_labels_count, 0:1]
                o2 = torch.zeros([batch_size, self.args.forecasting_horizon, 1]).to(device)
                adjusted_decoder_input = torch.cat([torch.cat([o1, o2], dim=1), u], dim=2).to(device)
                predicted = torch.reshape(self.model(encoder_input, adjusted_decoder_input),
                                          torch.Size([batch_size, self.args.transformer_labels_count
                                                      + self.args.forecasting_horizon]))
                predicted = predicted[:, self.args.transformer_labels_count:]
                output = torch.cat([output, predicted], dim=0)

        return output, target

    def __str__(self):
        return str(self.model)
