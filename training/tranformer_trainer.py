from abc import ABC

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from training.trainer import Trainer


class TransformerTrainer(Trainer, ABC):
    """
    Is a Trainer, which can be used to train transformers.
    """

    def __init__(self, train_data_loader: DataLoader, validation_data_loader: DataLoader, model: nn.Module,
                 loss_criterion, optimizer, epochs_count: int, learning_rate_scheduler: StepLR, args):
        """
        Creates a TransformerTrainer.

        :param train_data_loader:       loads the training data
        :param validation_data_loader:  loads the validation data
        :param model:                   the model to train
        :param loss_criterion:          can be any criterion to measure the loss of predictions
        :param optimizer:               the algorithm to optimize the weights
        :param epochs_count:            how many epochs are executed
        """
        super().__init__(train_data_loader, validation_data_loader, model, loss_criterion, optimizer, epochs_count,
                         learning_rate_scheduler, args)

    def train_phase(self, device) -> float:
        self.model.train()
        total_training_loss = 0.0
        for encoder_input, decoder_input in self.train_data_loader:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            self.optimizer.zero_grad()
            if self.args.transformer_use_teacher_forcing:
                expected = decoder_input[:, self.args.transformer_labels_count:, 0].detach().clone()
                decoder_input[:, 1:, 0:1] = decoder_input[:, :-1, 0:1]  # shift one step the target to the right

                decoder_sequence_length = decoder_input.shape[1]
                target_mask = create_mask(decoder_sequence_length).to(device)
                predicted = torch.reshape(
                    self.model(encoder_input, decoder_input, tgt_mask=target_mask),
                    torch.Size([encoder_input.shape[0],
                                self.args.transformer_labels_count + self.args.forecasting_horizon]))
                predicted = predicted[:, self.args.transformer_labels_count - 0:]

            elif self.args.transformer_use_auto_regression:
                predicted, expected = self.execute_model_one_step_ahead(encoder_input, decoder_input, device)

            else:  # generative approach (currently the best)
                predicted, expected = self.execute_model_on_batch(encoder_input, decoder_input, device)
            training_loss = self.loss_criterion(predicted, expected)
            training_loss.backward()
            self.optimizer.step()
            total_training_loss += training_loss.item()

        return total_training_loss / len(self.train_data_loader)

    def validation_phase(self, device) -> float:
        self.model.eval()
        total_validation_loss = 0.0
        with torch.no_grad():
            for batch_index, (encoder_input, decoder_input) in enumerate(self.validation_data_loader):
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)

                if self.args.transformer_use_teacher_forcing:
                    expected = decoder_input[:, self.args.transformer_labels_count:, 0].detach().clone().to(device)
                    decoder_input[:, 1:, 0] = decoder_input[:, :-1, 0]  # shift one step the target to the right

                    start_decoder_input = decoder_input[:, :self.args.transformer_labels_count + 1, :].to(device)
                    for i in range(1, 25):
                        target_mask = create_mask(start_decoder_input.shape[1]).to(device)
                        predicted = self.model(encoder_input, start_decoder_input, tgt_mask=target_mask).to(device)
                        if i == 24:
                            known_decoder_input = torch.zeros(start_decoder_input.shape[0], 1, start_decoder_input.shape[2] - 1).to(device)
                        else:
                            known_decoder_input = decoder_input[:, self.args.transformer_labels_count + i:
                                                                   self.args.transformer_labels_count + i + 1, 1:].to(
                                device)
                        new_predicted = predicted[:,
                                        self.args.transformer_labels_count + i -1:self.args.transformer_labels_count + i,
                                        0:1].to(device)
                        predicted = torch.cat([new_predicted, known_decoder_input], dim=2).to(device)
                        start_decoder_input = torch.cat([start_decoder_input[:, :, :], predicted], dim=1).to(device)
                    predicted = start_decoder_input[:, self.args.transformer_labels_count + 1:, 0].to(device)

                elif self.args.transformer_use_auto_regression:
                    predicted, expected = self.execute_model_one_step_ahead(encoder_input, decoder_input, device)
                    predicted = predicted[:, self.args.transformer_labels_count:]
                    expected = expected[:, self.args.transformer_labels_count:]

                else:  # generative approach (currently the best)
                    predicted, expected = self.execute_model_on_batch(encoder_input, decoder_input, device)
                    predicted = predicted[:, self.args.transformer_labels_count:]
                    expected = expected[:, self.args.transformer_labels_count:]
                validation_loss = self.loss_criterion(predicted, expected)

                if batch_index % 30 == 0:
                    print(batch_index, validation_loss)
                    plt.plot(predicted.to('cpu').detach().numpy()[:, 0], label='pred')
                    plt.plot(expected.to('cpu').detach().numpy()[:, 0], label='expected')
                    plt.legend(loc='upper left')
                    plt.show()
                total_validation_loss += validation_loss.item()

        return total_validation_loss / len(self.validation_data_loader)

    def execute_model_on_batch(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor,
                               device: str) -> [torch.Tensor, torch.Tensor]:
        batch_size = encoder_input.shape[0]
        expected = decoder_input[:, :, 0]
        u = decoder_input[:, :, 1:]
        o1 = decoder_input[:, :self.args.transformer_labels_count, 0:1]
        o2 = torch.zeros([batch_size, self.args.forecasting_horizon, 1]).to(device)
        adjusted_decoder_input = torch.cat([torch.cat([o1, o2], dim=1), u], dim=2).to(device)
        target_mask = create_mask(self.args.transformer_labels_count + self.args.forecasting_horizon).to(device)
        predicted = torch.reshape(self.model(encoder_input, adjusted_decoder_input, tgt_mask=target_mask),
                                  torch.Size([batch_size,
                                              self.args.transformer_labels_count + self.args.forecasting_horizon]))
        target_predicted = predicted[:, :]
        return target_predicted, expected

    def execute_model_one_step_ahead(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor,
                                     device: str) -> [torch.Tensor, torch.Tensor]:
        batch_size = encoder_input.shape[0]
        expected = decoder_input[:, :self.args.transformer_labels_count + 1, 0]
        u = decoder_input[:, :-self.args.forecasting_horizon + 1, 1:]
        o1 = decoder_input[:, :self.args.transformer_labels_count, 0:1]
        o2 = torch.zeros([batch_size, 1, 1]).to(device)
        adjusted_decoder_input = torch.cat([torch.cat([o1, o2], dim=1), u], dim=2).to(device)
        target_mask = create_mask(self.args.transformer_labels_count + 1).to(device)
        predicted = torch.reshape(self.model(encoder_input, adjusted_decoder_input, tgt_mask=target_mask),
                                  torch.Size([batch_size, self.args.transformer_labels_count + 1]))
        target_predicted = predicted[:, :]
        return target_predicted, expected


def create_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask \
        .float() \
        .masked_fill(mask == 0, float('-inf')) \
        .masked_fill(mask == 1, float(0.0))
    return mask
