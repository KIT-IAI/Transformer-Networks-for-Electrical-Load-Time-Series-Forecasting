from torch import nn


class TimeSeriesTransformer(nn.Module):

    def __init__(self, d_model: int, input_features_count: int, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, dropout: float, attention_heads: int):
        super().__init__()
        self.linear = nn.Linear(input_features_count, d_model)
        self.linear2 = nn.Linear(input_features_count, d_model)
        self.transformer = nn.Transformer(d_model, attention_heads, num_encoder_layers, num_decoder_layers,
                                          batch_first=True, dim_feedforward=dim_feedforward, dropout=dropout)
        self.proj1 = nn.Linear(d_model, 64)
        self.proj2 = nn.Linear(64, 64)
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x_enc, x_dec, src_mask=None, tgt_mask=None, dec_enc_mask=None):
        enc_embedding = self.linear(x_enc)
        dec_embedding = self.linear2(x_dec)
        out = self.transformer(x_enc, x_dec, src_mask=src_mask, tgt_mask=tgt_mask)
        # out = self.proj1(out)
        # out = self.proj2(out)
        out = self.projection(out)
        return out

    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
    #             enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
    #     enc_out = self.enc_embedding(x_enc, x_mark_enc)
    #     enc_out = self.encoder(enc_out)
    #
    #     dec_out = self.dec_embedding(x_dec, x_mark_dec)
    #     dec_out = self.decoder(dec_out, enc_out)
    #     dec_out = self.projection(dec_out)
    #
    #     # dec_out = self.end_conv1(dec_out)
    #     # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
    #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
