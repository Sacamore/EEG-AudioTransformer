import numpy as np
import torch
import torch.utils.data as utils_data
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from torch import Tensor

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class TransformerModel(nn.Module):
    def __init__(self,d_model,nhead,n_layer,dropout=0.1) -> None:
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=512,batch_first=True,dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=n_layer)
    
    def forward(self,x:Tensor):
        res = self.transformer(x)
        return res

class Model(nn.Module):
    def __init__(self,input_dim,output_dim,d_model,nhead,n_layer,dropout = 0.1) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_dim,d_model*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model,d_model)
        )
        self.transformer = TransformerModel(d_model=d_model,nhead=nhead,n_layer=n_layer,dropout=dropout)
        self.l2 = nn.Sequential(
            nn.Linear(d_model,2*d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(4*d_model,2*d_model),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(2*d_model,output_dim)
        )
        self.gelu2 = nn.GELU()

    def forward(self,x:Tensor):
        x = self.l1(x)
        x = self.transformer(x)
        x = self.l2(x)
        return x