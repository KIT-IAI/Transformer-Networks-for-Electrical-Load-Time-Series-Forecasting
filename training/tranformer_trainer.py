from abc import ABC

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from training.trainer import Trainer


class TransformerTrainer(Trainer, ABC):
    """
    Is a Trainer, which can be used to train Transformers.
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

        # iterate over the training batches
        for encoder_input, decoder_input in self.train_data_loader:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            #print(encoder_input.shape, decoder_input.shape)
            #print(encoder_input[0])
            #print(decoder_input[0])
            self.optimizer.zero_grad()

            # there are three possible training methods: teacher-forcing, one step ahead and the generative approach
            if self.args.transformer_use_teacher_forcing:
                expected = decoder_input[:, self.args.transformer_labels_count:, 0].detach().clone()

                # shift the target one step to the right (--> model does not learn to copy input)
                decoder_input[:, 1:, 0:1] = decoder_input[:, :-1, 0:1]

                # create a mask for the target of
                decoder_sequence_length = decoder_input.shape[1]
                target_mask = create_mask(decoder_sequence_length).to(device)

                # execute the model for the batch and shape the output
                predicted = self.model(encoder_input, decoder_input, tgt_mask=target_mask)
                predicted = torch.reshape(
                    predicted,
                    torch.Size([encoder_input.shape[0],
                                self.args.transformer_labels_count + self.args.forecasting_horizon]))
                predicted = predicted[:, self.args.transformer_labels_count:]

            elif self.args.transformer_use_auto_regression:
                predicted, expected = self.execute_model_one_step_ahead(encoder_input, decoder_input, device)
                predicted = predicted[:, self.args.transformer_labels_count - 1:]
                expected = expected[:, self.args.transformer_labels_count - 1:]

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

                # there are three possible training methods: teacher-forcing, one step ahead and the generative approach
                if self.args.transformer_use_teacher_forcing:
                    expected = decoder_input[:, self.args.transformer_labels_count:, 0].detach().clone().to(device)
                    decoder_input[:, 1:, 0] = decoder_input[:, :-1, 0]  # shift one step the target to the right

                    # iterate autoregressive for each forecast
                    start_decoder_input = decoder_input[:, :self.args.transformer_labels_count + 1, :].to(device)
                    for i in range(1, 1 + self.args.forecasting_horizon):
                        target_mask = create_mask(start_decoder_input.shape[1]).to(device)
                        predicted = self.model(encoder_input, start_decoder_input, tgt_mask=target_mask).to(device)
                        if i == self.args.forecasting_horizon:
                            known_decoder_input = torch.zeros(start_decoder_input.shape[0], 1,
                                                              start_decoder_input.shape[2] - 1).to(device)
                        else:
                            known_decoder_input = decoder_input[:, self.args.transformer_labels_count + i:
                                                                   self.args.transformer_labels_count + i + 1, 1:] \
                                .to(device)
                        new_predicted = predicted[:, self.args.transformer_labels_count + i - 1
                                                     :self.args.transformer_labels_count + i, 0:1].to(device)
                        predicted = torch.cat([new_predicted, known_decoder_input], dim=2).to(device)
                        start_decoder_input = torch.cat([start_decoder_input, predicted], dim=1).to(device)
                    predicted = start_decoder_input[:, self.args.transformer_labels_count + 1:, 0].to(device)

                elif self.args.transformer_use_auto_regression:
                    predicted, expected = self.execute_model_one_step_ahead(encoder_input, decoder_input, device)
                    predicted = predicted[:, self.args.transformer_labels_count - 1:]
                    expected = expected[:, self.args.transformer_labels_count - 1:]

                else:  # generative approach (currently the best)
                    predicted, expected = self.execute_model_on_batch(encoder_input, decoder_input, device)
                    predicted = predicted[:, self.args.transformer_labels_count:]
                    expected = expected[:, self.args.transformer_labels_count:]

                validation_loss = self.loss_criterion(predicted, expected)
                total_validation_loss += validation_loss.item()

        return total_validation_loss / len(self.validation_data_loader)

    def execute_model_on_batch(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor,
                               device: str) -> [torch.Tensor, torch.Tensor]:
        """
        The model is executed for a batch with the whole input (intended to be used for the generative transformer
        training method.

        :param encoder_input: the raw input of the encoder [Batch, Sequence, Features]
        :param decoder_input: the raw input of the decoder [Batch, Sequence, Features]
        :param device: indicates on which device the output is calculated
        :returns: a tuple of the predicted and expected tensor
        """
        # prepare the input for the decoder (set the future time series values to zero)
        batch_size = encoder_input.shape[0]
        decoder_sequence_length = self.args.transformer_labels_count + self.args.forecasting_horizon
        expected = decoder_input[:, :, 0]
        u = decoder_input[:, :, 1:]
        o1 = decoder_input[:, :self.args.transformer_labels_count, 0:1]
        o2 = torch.zeros([batch_size, self.args.forecasting_horizon, 1]).to(device)
        adjusted_decoder_input = torch.cat([torch.cat([o1, o2], dim=1), u], dim=2).to(device)

        # execute the model with a mask
        target_mask = create_mask(decoder_sequence_length).to(device)
        predicted = self.model(encoder_input, adjusted_decoder_input, tgt_mask=target_mask)
        predicted = torch.reshape(predicted, torch.Size([batch_size, decoder_sequence_length]))
        return predicted, expected

    def execute_model_one_step_ahead(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor,
                                     device: str) -> [torch.Tensor, torch.Tensor]:
        """
        The model is executed for a batch with the whole input (intended to be used for the one step ahead transformer
        training method).

        :param encoder_input: the raw input of the encoder [Batch, Sequence, Features]
        :param decoder_input: the raw input of the decoder [Batch, Sequence, Features]
        :param device: indicates on which device the output is calculated
        :returns: a tuple of the predicted and expected tensor
        """

        # prepare the input for the decoder (shift the time series values one step to the right)
        batch_size = encoder_input.shape[0]
        expected = decoder_input[:, 1:self.args.transformer_labels_count + 1, 0]
        u = decoder_input[:, 1:-self.args.forecasting_horizon + 1, 1:]
        o1 = decoder_input[:, :self.args.transformer_labels_count, 0:1]
        adjusted_decoder_input = torch.cat([o1, u], dim=2).to(device)

        # execute the model with a mask
        target_mask = create_mask(self.args.transformer_labels_count).to(device)
        predicted = self.model(encoder_input, adjusted_decoder_input, tgt_mask=target_mask)
        predicted = torch.reshape(predicted, torch.Size([batch_size, self.args.transformer_labels_count]))
        return predicted, expected


def create_mask(size):
    """
    Creates a quadratic mask of the given size with the triangular upper set to to negative infinity and the half to 0.
    :returns: the 2D mask of the given size
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask \
        .float() \
        .masked_fill(mask == 0, float('-inf')) \
        .masked_fill(mask == 1, float(0.0))
    return mask
