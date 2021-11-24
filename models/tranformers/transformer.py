import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):

    def __init__(self, d_model: int, input_features_count: int, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, dropout: float, attention_heads: int):
        super().__init__()
        self.linear = nn.Linear(input_features_count, d_model)
        self.linear2 = nn.Linear(input_features_count, d_model)
        self.transformer = nn.Transformer(d_model, attention_heads, num_encoder_layers, num_decoder_layers,
                                          batch_first=True, dim_feedforward=dim_feedforward, dropout=dropout)
        self.projection = nn.Linear(d_model, 1, bias=True)
        self.positional_encoding_enc = PositionalEncoding(d_model)
        self.positional_encoding_dec = PositionalEncoding(d_model)

    def forward(self, x_enc, x_dec, src_mask=None, tgt_mask=None, dec_enc_mask=None):
        enc_embedding = self.linear(x_enc)
        dec_embedding = self.linear2(x_dec)
        enc_pos_encoded = self.positional_encoding_enc(enc_embedding)
        dec_pos_encoded = self.positional_encoding_dec(dec_embedding)
        out = self.transformer(enc_pos_encoded, dec_pos_encoded, src_mask=src_mask, tgt_mask=tgt_mask)
        out = self.projection(out)
        return out
