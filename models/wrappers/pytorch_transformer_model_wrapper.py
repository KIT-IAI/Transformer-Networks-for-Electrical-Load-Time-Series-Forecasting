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
        train_dl = DataLoader(train_dataset, batch_size=1000)
        validation_dl = DataLoader(validation_dataset, batch_size=1000)

        criterion = L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.98))
        scheduler = StepLR(optimizer, self.args.learning_rate_scheduler_step, self.args.learning_rate_scheduler_gamma)
        trainer = TransformerTrainer(train_dl, validation_dl, self.model, criterion, optimizer, self.args.max_epochs,
                                     scheduler, self.args.use_early_stopping, self.args.early_stopping_patience)
        return trainer.train()

    def predict(self, dataset: TransformerDataset) -> (torch.Tensor, torch.Tensor):
        data_loader = DataLoader(dataset, batch_size=1000)

        self.model.eval()
        with torch.no_grad():
            device = 'cpu'
            output = torch.tensor([])
            target = torch.tensor([])
            for batch_index, (encoder_input, decoder_input) in enumerate(data_loader):
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                expected = decoder_input[:, :, 0]
                output = torch.cat([output, expected], dim=0)

                start_decoder_input = encoder_input[:, 167:168, :]
                for i in range(0, 24):
                    predicted = self.model(encoder_input, start_decoder_input)
                    known_decoder_input = decoder_input[:, i:i + 1, 1:]
                    new_predicted = predicted[:, i:i + 1, 0:1]
                    predicted = torch.cat([new_predicted, known_decoder_input], dim=2)
                    start_decoder_input = torch.cat([start_decoder_input[:, :, :], predicted], dim=1)
                total_prediction = start_decoder_input[:, 1:, 0]
                target = torch.cat([target, total_prediction], dim=0)

        return output, target

    def __str__(self):
        return str(self.model)
