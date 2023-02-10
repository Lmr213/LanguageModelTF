import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch.autograd import Variable


class Embedding(nn.Module):
    def __init__(self, num_hiddens, vocab_size):  # num_hiddens:词嵌入维度
        super(Embedding, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.num_hiddens)  # 缩放作用
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵
        pm = torch.zeros(max_len, num_hiddens)

        # 初始化一个绝对位置编码矩阵, 转换为二维张量:max_len * 1的矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        # 1 * num_hiddens size的变换矩阵
        div_term = torch.exp(torch.arange(0, num_hiddens, 2) * -(math.log(10000.0) / num_hiddens))

        pm[:, 0::2] = torch.sin(position * div_term)
        pm[:, 1::2] = torch.sin(position * div_term)
        pm = pm.unsqueeze(0)  # match the output size of embedding and pm dimension to do addition

        self.register_buffer('pm', pm)

    def forward(self, x):
        x = x + Variable(self.pm[:, x.size(1)], requires_grad=False)
        return self.dropout(x)


class outLayer(nn.Module):
    def __init__(self, num_hiddens, vocab_size):
        super(outLayer, self).__init__()
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, x):
        output = F.log_softmax(self.out(x), dim=-1)  # -1是指在最后一个维度(vocab_size维度)做softmax
        return output


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_size, dropout, max_len=5000)
        self.embedding = Embedding(embed_size, vocab_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.out = outLayer(embed_size, vocab_size)

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = TransformerModel.get_key_padding_mask(src)
        tgt_key_padding_mask = TransformerModel.get_key_padding_mask(tgt)

        # src size: (batch_size, len(src))
        # tgt size: (batch_size, len(tgt))
        # out size: (batch_size, len(seq), embed_size)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Transformer blocks - Out size = (len(seq), batch_size, vocab_size)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask,
                                           src_key_padding_mask=src_key_padding_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask)

        out = self.out(transformer_out)
        return out

    def get_key_padding_mask(tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask
