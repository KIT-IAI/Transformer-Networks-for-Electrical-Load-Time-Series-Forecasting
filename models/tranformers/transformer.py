import math

import torch
from torch import nn, Tensor


class ValueEmbedding(nn.Module):

    def __init__(self, d_model: int, time_series_features=1):
        super(ValueEmbedding, self).__init__()
        self.linear = nn.Linear(time_series_features, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Creates from the given tensor a linear projection.
        :param x: the input tensor to project
        :return: the projected tensor
        """
        return self.linear(x)


class TokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=(3,), padding=1, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TimeEmbedding(nn.Module):

    def __init__(self, d_model: int, time_series_features=8):
        super(TimeEmbedding, self).__init__()
        self.linear = nn.Linear(time_series_features, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Creates from the given tensor a linear projection.
        :param x: the input tensor to project
        :return: the projected tensor
        """
        return self.linear(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe.require_grad = False
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:x.size(0)]


class TotalEmbedding(nn.Module):

    def __init__(self, d_model: int, value_features: int, time_features: int, dropout: float):
        super(TotalEmbedding, self).__init__()

        self.value_features = value_features
        self.time_features = time_features

        self.value_embedding = ValueEmbedding(d_model, value_features + time_features)
        self.time_embedding = TimeEmbedding(d_model, time_features)
        self.positional_encoding = PositionalEncoding(d_model)

        self.linear_embedding_weight = nn.Linear(3, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        """
        :param x: tensor of dimension [Batch_Size, Sequence_Length, Features]
        :return: the embedded value
        """
        value_embedded = self.value_embedding(x[:, :, :])
        # time_embedded = self.time_embedding(x[:, :, -self.time_features:])
        pe = self.positional_encoding(x)
        return self.dropout(value_embedded + pe)


class MultipleLinearLayers(nn.Module):

    def __init__(self, d_model: int, d_output):
        super().__init__()
        self.linear_layers = []
        for i in range(0, d_output):
            self.linear_layers.append(nn.Linear(d_model, 1).to('cuda'))

    def forward(self, x):
        output = torch.zeros([x.shape[0], x.shape[1]])
        output = output.to('cuda')

        for i in range(0, x.shape[1]):
            output[:, i:i+1] = self.linear_layers[i](x[:, i, :])
        return output


class TimeSeriesTransformer(nn.Module):

    def __init__(self, d_model: int, input_features_count: int, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, dropout: float, attention_heads: int):
        super().__init__()
        self.transformer = nn.Transformer(d_model, attention_heads, num_encoder_layers, num_decoder_layers,
                                          batch_first=True, dim_feedforward=dim_feedforward, dropout=dropout)
        # self.projection = MultipleLinearLayers(d_model, 48)
        self.projection = nn.Linear(d_model, 1, bias=True)
        self.encoder_embedding = TotalEmbedding(d_model, 1, input_features_count - 1, dropout)
        self.decoder_embedding = TotalEmbedding(d_model, 1, input_features_count - 1, dropout)
        self.relu = nn.ReLU()

    def forward(self, x_enc, x_dec, src_mask=None, tgt_mask=None, dec_enc_mask=None):
        enc_embedding = self.encoder_embedding(x_enc)
        dec_embedding = self.decoder_embedding(x_dec)
        out = self.transformer(enc_embedding, dec_embedding, src_mask=src_mask, tgt_mask=tgt_mask)
        out = self.projection(self.relu(out))
        return out
