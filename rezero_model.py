'''
ReZero Model
This code is a modified version of the ReZero source code provided 
by the author's of ReZero. It can be found here:
https://github.com/tbachlechner/ReZero-examples/blob/master/ReZero-Deep_Fast_Transformer.ipynb
'''


import numpy as np
import time
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_

torch.manual_seed(0)

######################################################################
# Define the ReZero Transformer


class ReZeroEncoderLayer(Module):
    r"""ReZero-TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        num_features: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        use_LayerNorm: using either no LayerNorm (dafault=False), or use LayerNorm "pre", or "post"

    Examples::
        >>> encoder_layer = ReZeroModel(num_features=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    def __init__(self, num_features=22, nhead=3, dim_feedforward=2048, dropout=0.1, activation = "relu", 
                 use_LayerNorm = True, init_resweight = 0, resweight_trainable = True):
        super(ReZeroEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(num_features, nhead, dropout=dropout)
        
        # Define the Resisdual Weight for ReZero
        self.resweight = torch.nn.Parameter(torch.Tensor([init_resweight]), requires_grad = resweight_trainable)

        # Implementation of Feedforward model
        self.linear1 = Linear(num_features, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, num_features)
        self.use_LayerNorm = use_LayerNorm
        if self.use_LayerNorm != False:
            self.norm1 = LayerNorm(num_features)
            self.norm2 = LayerNorm(num_features)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh
        
        
        # self.embedding = nn.Embedding(22, 20)
        # self.full_embedding = nn.Embedding(51, 51)
        # self.bn = nn.BatchNorm1d(51)
        

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(ReZeroEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        
        ## IN: bs x 700 x 51
        src2 = src
        if self.use_LayerNorm == "pre":
            src2 = self.norm1(src2)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        # Apply the residual weight to the residual connection. This enables ReZero.
        src2 = self.resweight * src2
        src2 = self.dropout1(src2)
        if self.use_LayerNorm == False:
            src = src + src2
        elif self.use_LayerNorm == "pre":
            src = src + src2
        elif self.use_LayerNorm == "post":
            src = self.norm1(src + src2)
        src2 = src
        if self.use_LayerNorm == "pre":
            src2 = self.norm1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = self.resweight * src2
        src2 = self.dropout2(src2)
        if self.use_LayerNorm == False:
            src = src + src2
        elif self.use_LayerNorm == "pre":
            src = src + src2
        elif self.use_LayerNorm == "post":
            src = self.norm1(src + src2)
        return src



######################################################################
# Define the model

class ReZeroModel(torch.nn.Module):
    def __init__(self, num_features, ntoken=22, nhead=1, nhid=2048, nlayers=1, dropout=0.1):
        super(ReZeroModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        compress_representation = 36
        final_cats = 9
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = ReZeroEncoderLayer(compress_representation, nhead, nhid, dropout, 
            activation = "relu", use_LayerNorm = False, init_resweight = 0, 
            resweight_trainable = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.encoder = torch.nn.Embedding(num_features,ntoken) 
        self.ninp = num_features
        self.decoder = torch.nn.Linear(compress_representation, final_cats)
        self._reset_parameters()
        self.init_weights()
        self.linear_compress = torch.nn.Linear(num_features,compress_representation)
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, device, one_hot_embed):
        src = src.permute(2,0,1) # Correct transformer dims: seq_len x bs x features 
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.linear_compress(src)
        src = src * math.sqrt(self.ninp)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output).permute(1,0,2)
        return output

######################################################################
# Positional Encoding

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)