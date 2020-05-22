
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


class ReZeroModel(Module):
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

    def __init__(self, num_features=51, nhead=3, dim_feedforward=2048, dropout=0.1, activation = "relu", 
                 use_LayerNorm = False, init_resweight = 0, resweight_trainable = True):
        super(ReZeroModel, self).__init__()
        
        self.self_attn = MultiheadAttention(num_features, nhead, dropout=dropout)
        
        # Define the Resisdual Weight for ReZero
        self.resweight = torch.nn.Parameter(torch.Tensor([init_resweight]), requires_grad = resweight_trainable)

        # Implementation of Feedforward model
        self.linear1 = Linear(num_features, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, 9)
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
        
        
        self.embedding = nn.Embedding(22, 22)
        self.full_embedding = nn.Embedding(51, 51)
        self.bn = nn.BatchNorm1d(51)
        

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(ReZeroModel, self).__setstate__(state)
    
    def _generate_square_subsequent_mask(self, mask):
        mask = mask.bool().masked_fill(mask == 1, True)
        mask = mask.masked_fill(mask != True, False)
        return mask #torch.transpose(mask,0,1)

    def forward(self, src, device, one_hot_embed, src_mask=None, src_key_padding_mask=None):
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
        # Set padding mask
        pad = src[:,21,:] 
        pad_mask = self._generate_square_subsequent_mask(pad.to(device))
            
        if (one_hot_embed == True):
            # embed one hot
            one_hot = src[:, 0:22, :].argmax(axis=1)
            embedded = self.embedding(one_hot.long()).permute(0, 2, 1)
            src[:, 0:22, :] = embedded
        
        src = self.bn(src) # in: bs x features x length
        
         
        src = src.permute(2,0,1)  # permute for transformer. Need to undo this at end of fwd pass
        src2 = src
    
        
        if self.use_LayerNorm == "pre":
            src2 = self.norm1(src2)
        src2 = self.self_attn(src2,src2,src2, key_padding_mask=pad_mask)[0]
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
            #src = src + src2
            src = src2
        elif self.use_LayerNorm == "pre":
            src = src + src2
        elif self.use_LayerNorm == "post":
            src = self.norm1(src + src2)
        return src.permute(1,0,2)
