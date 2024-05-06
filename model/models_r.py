import numpy as np
import torch
import torch.utils
import torch.utils.data as utils_data
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from torch import Tensor

import model.transformer as transformer

class Transformer(nn.Module):
    def __init__(self,d_model:int,nhead:int,n_layer:int,dropout:float=0.1) -> None:
        super().__init__()
        self.RoPEMultiHeadAttention = transformer.RotaryPEMultiHeadAttention(heads=nhead,d_model=d_model,rope_percentage=1.0,dropout=dropout)
        self.ff = transformer.ConvFeedForward(d_model=d_model,d_ff=4*d_model,activation=nn.GELU(),dropout=dropout)
        self.encoderLayer = transformer.TransformerLayer(d_model=d_model,self_attn=self.RoPEMultiHeadAttention,feed_forward=self.ff,dropout=dropout)
        self.encoder = transformer.Encoder(self.encoderLayer,n_layers=n_layer)
        self.src_mask = None
    def forward(self,x:torch.Tensor):
        # if self.src_mask is None or self.src_mask.size(0)!=x.shape[1]:
        #     self.src_mask = transformer.subsequent_mask(x.shape[1]).to(x.device)
        # TODO: 增加随机遮掩
        x = self.encoder(x,self.src_mask)
        return x

class Model(nn.Module):
    def __init__(self,input_dim,output_dim,seg_size,d_model,nhead,n_layer,dropout = 0.1) -> None:
        super().__init__()
        self.affine_eeg = nn.Sequential(
            nn.Linear(input_dim,d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model,d_model)
        )
        self.eeg_transformer = Transformer(d_model=d_model,nhead=nhead,n_layer=n_layer,dropout=dropout)



        self.outputLayer = nn.Sequential(
            nn.Linear(d_model*seg_size,4*d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model,output_dim)
        )

    def forward(self,eeg:Tensor):
        aff_eeg = self.affine_eeg(eeg)
        att_eeg = self.eeg_transformer(aff_eeg)
        att_eeg = att_eeg.contiguous().view(att_eeg.size(0), -1)
        mel_eeg = self.outputLayer(att_eeg)

        return mel_eeg