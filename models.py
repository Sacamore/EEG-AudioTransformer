import numpy as np
import torch
import torch.utils
import torch.utils.data as utils_data
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from torch import Tensor

import transformer

class Transformer(nn.Module):
    def __init__(self,d_model:int,nhead:int,n_layer:int,dropout:float=0.1) -> None:
        super().__init__()
        self.RoPEMultiHeadAttention = transformer.RotaryPEMultiHeadAttention(heads=nhead,d_model=d_model,rope_percentage=1.0,dropout=dropout)
        self.ff = transformer.ConvFeedForward(d_model=d_model,d_ff=4*d_model,activation=nn.GELU(),dropout=dropout)
        self.encoderLayer = transformer.TransformerLayer(d_model=d_model,self_attn=self.RoPEMultiHeadAttention,feed_forward=self.ff,dropout=dropout)
        self.encoder = transformer.Encoder(self.encoderLayer,n_layers=n_layer)
        # self.decoderLayer = transformer.TransformerLayer(d_model=d_model,self_attn=self.RoPEMultiHeadAttention,src_attn=self.RoPEMultiHeadAttention,feed_forward=self.ff,dropout=dropout)
        # self.decoder = transformer.Decoder(self.decoderLayer,n_layers=n_layer)
        # self.transformer = transformer.Transformer(encoder=self.encoder,decoder=self.decoder)
        self.src_mask = None
    def forward(self,x:torch.Tensor):
        # if self.src_mask is None or self.src_mask.size(0)!=x.shape[1]:
        #     self.src_mask = transformer.subsequent_mask(x.shape[1]).to(x.device)
        x = self.encoder(x,self.src_mask)
        return x
        

    

class TransformerModel(nn.Module):
    def __init__(self,d_model,nhead,n_layer,dropout=0.1) -> None:
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=d_model*4,batch_first=True,dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=n_layer)
    
    def forward(self,x:Tensor):
        res = self.transformer(x)
        return res

class Model(nn.Module):
    def __init__(self,input_dim,output_dim,seg_size,d_model,nhead,n_layer,dropout = 0.1) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_dim,2*d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model,d_model)
        )
        self.transformer = Transformer(d_model=d_model,nhead=nhead,n_layer=n_layer,dropout=dropout)
        self.conv1 = nn.Conv1d(d_model,d_model,seg_size//2+1,1,seg_size//4)
        self.maxpool1 = nn.AvgPool1d(seg_size,1)
        self.l3 = nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(4*d_model,2*d_model),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(4*d_model,output_dim)
        )
        # self.gelu2 = nn.GELU()

    def forward(self,x:Tensor):
        x = self.l1(x)
        x = self.transformer(x)
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = x.squeeze(-1)
        x = self.l3(x)
        # x = torch.clamp(x,min = np.log(1e-5),max=None)
        return x