import numpy as np
import torch
import torch.utils.data as utils_data
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from torch import Tensor

class LayerNorm(nn.LayerNorm):
    """Layer Normalization with support for fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    """
    Quick GELU activation function.
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Residual Attention Block.
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Transformer block composed of residual attention blocks.
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class TransformerModel(nn.Module):
    def __init__(self,d_model,nhead,n_layer,dropout=0.1) -> None:
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=d_model*4,batch_first=True,dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=n_layer)
    
    def forward(self,x:Tensor):
        res = self.transformer(x)
        return res

class Model(nn.Module):
    def __init__(self,input_dim,scaled_dim,seg_size,output_dim,d_model,nhead,n_layer,dropout = 0.1) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_dim,scaled_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(scaled_dim*2,scaled_dim)
        )
        # self.l2 = nn.Linear(1,d_model)
        # nn.Sequential(
        #     nn.Linear(1,d_model*2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model*2,d_model)
        # )
        # self.transformer = Transformer(d_model,n_layer,nhead)
        self.transformer = TransformerModel(d_model=d_model,nhead=nhead,n_layer=n_layer,dropout=dropout)
        # self.conv1 = nn.Conv1d(scaled_dim,scaled_dim,prv_dim)
        self.l3 = nn.Sequential(
            nn.Linear(scaled_dim,4*scaled_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*scaled_dim,2*scaled_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*scaled_dim,output_dim)
        )
        # self.gelu2 = nn.GELU()

    def forward(self,x:Tensor):
        x = self.l1(x)
        # x = x.unsqueeze(-1)
        # x = self.l2(x)
        x = self.transformer(x)
        # x = x.permute(0,2,1)
        # x = self.conv1(x)
        # x = x.squeeze(-1)
        x = self.l3(x)
        return x