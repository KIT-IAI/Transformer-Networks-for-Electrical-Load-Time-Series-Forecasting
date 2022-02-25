from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import MultiheadAttention

from models.tranformers.transformer import TotalEmbedding


#
# This code part is based on the implementation of the Informer paper by Haoyi Zhou, Shanghang Zhang, Jieqi Peng,
# Shuai Zhang, Jianxin Li, Hui Xiong and Wancai Zhang. https://github.com/zhouhaoyi/Informer2020
#


class ProbMask:
    def __init__(self, b, h, l, index, scores, device='cpu'):
        _mask = torch.ones(l, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(b, h, l, scores.shape[-1])
        indicator = _mask_ex[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


def prob_qk(q, k, sample_k, n_top):  # n_top: c*ln(L_q)
    # Q [B, H, L, D]
    b, h, l_k, e = k.shape
    _, _, l_q, _ = q.shape

    # calculate the sampled Q_K
    k_expand = k.unsqueeze(-3).expand(b, h, l_q, l_k, e)
    index_sample = torch.randint(l_k, (l_q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
    k_sample = k_expand[:, :, torch.arange(l_q).unsqueeze(1), index_sample, :]
    q_k_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze(-2)

    # find the Top_k query with sparsity measurement
    m = q_k_sample.max(-1)[0] - torch.div(q_k_sample.sum(-1), l_k)
    m_top = m.topk(n_top, sorted=False)[1]

    # use the reduced Q to calculate Q_K
    q_reduce = q[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], m_top, :]  # factor*ln(L_q)
    q_k = torch.matmul(q_reduce, k.transpose(-2, -1))  # factor*ln(L_q)*L_k

    return q_k, m_top


class ProbAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.n_heads = n_heads
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def _get_initial_context(self, v, l_q):
        b, h, l_v, d = v.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            v_sum = v.mean(dim=-2)
            context = v_sum.unsqueeze(-2).expand(b, h, l_q, v_sum.shape[-1]).clone()
        else:  # use mask
            assert (l_q == l_v)  # requires that L_Q == L_V, i.e. for self-attention only
            context = v.cumsum(dim=-2)
        return context

    def _update_context(self, context_in, v, scores, index, l_q):
        b, h, l_v, d = v.shape

        if self.mask_flag:
            attn_mask = ProbMask(b, h, l_q, index, scores, device=v.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], index, :] \
            = torch.matmul(attn, v).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([b, h, l_v, l_v]) / l_v).type_as(attn).to(attn.device)
            attns[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values):
        b, l, _ = queries.shape
        _, s, _ = keys.shape
        h = self.n_heads

        queries = self.query_projection(queries).view(b, l, h, -1)
        keys = self.key_projection(keys).view(b, s, h, -1)
        values = self.value_projection(values).view(b, s, h, -1)

        b, l_q, h, d = queries.shape
        _, l_k, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        u_part = self.factor * np.ceil(np.log(l_k)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(l_q)).astype('int').item()  # c*ln(L_q)

        u_part = u_part if u_part < l_k else l_k
        u = u if u < l_q else l_q

        scores_top, index = prob_qk(queries, keys, sample_k=u_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(d)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, l_q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, l_q)

        return context.transpose(2, 1).contiguous().view(b, l, -1), attn


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class ConvLayer(nn.Module):
    def __init__(self, c_in, kernel_size: int):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=(kernel_size,),
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x):
        # x [B, L, D]
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns


class Informer(nn.Module):
    def __init__(self, input_features_count: int, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, dropout=0.0,
                 attn='prob', activation='gelu', output_attention=False, distil=True, device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = TotalEmbedding(d_model, 1, input_features_count - 1, dropout)
        self.dec_embedding = TotalEmbedding(d_model, 1, input_features_count - 1, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # ProbAttention(d_model, n_heads, False, attention_dropout=dropout, output_attention=output_attention),
                    MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True, device=device),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model, 3
                ) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # ProbAttention(d_model, n_heads, True, attention_dropout=dropout, output_attention=False),
                    # ProbAttention(d_model, n_heads, False, attention_dropout=dropout, output_attention=False),
                    MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True, device=device),
                    MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True, device=device),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, encoder_input, decoder_input, enc_self_mask=None, tgt_mask=None):
        enc_out = self.enc_embedding(encoder_input)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(decoder_input)
        dec_out = self.decoder(dec_out, enc_out, x_mask=tgt_mask)
        dec_out = self.projection(dec_out)

        return dec_out
