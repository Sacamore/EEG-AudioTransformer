import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy

import model.transformer as transformer

class TransformerEncoder(nn.Module):
    def __init__(self,input_dim:int,d_model:int,nhead:int=4,n_layer:int=2,dropout:float=0.1) -> None:
        super().__init__()
        self.aff_input = nn.Linear(input_dim,d_model)
        self.RoPEMultiHeadAttention = transformer.RotaryPEMultiHeadAttention(heads=nhead,d_model=d_model,rope_percentage=1.0,dropout=dropout)
        self.ff = transformer.ConvFeedForward(d_model=d_model,d_ff=4*d_model,activation=nn.GELU(),dropout=dropout)
        self.encoderLayer = transformer.TransformerLayer(d_model=d_model,self_attn=self.RoPEMultiHeadAttention,feed_forward=self.ff,dropout=dropout)
        self.encoder = transformer.Encoder(self.encoderLayer,n_layers=n_layer)
        self.src_mask = None
    def forward(self,x:torch.Tensor):
        x = self.aff_input(x)
        x = self.encoder(x,self.src_mask)
        return x

class ResidualLayer(nn.Module):
    def __init__(self,input_dim:int,d_model:int,hidden_dim:int) -> None:
        super().__init__()
        self.res_block = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(input_dim,hidden_dim,kernel_size=3,stride=1,padding=1,bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim,d_model,kernel_size=1,stride=1,bias=False)
        )
    def forward(self, x:torch.Tensor):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    def __init__(self,input_dim:int,d_model:int,hidden_dim:int,n_layers:int) -> None:
        super().__init__()
        self.stack = nn.ModuleList([copy.deepcopy(ResidualLayer(input_dim,d_model,hidden_dim)) for _ in range(n_layers)])
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self,x:torch.Tensor):
        for layer in self.stack:
            x = layer(x)
        x = self.relu(x)
        return x


class SemanticDecoder(nn.Module):
    def __init__(self, input_dim:int , class_num: int) -> None:
        super().__init__()
        self.aff_input = nn.Linear(input_dim,input_dim)
        self.classifier = nn.Linear(input_dim,class_num)
    
    def forward(self,input_vq:torch.Tensor):
        feat = self.aff_input(input_vq)
        class_logits = self.classifier(feat)
        return class_logits
    
class FeatEncoder(nn.Module):
    def __init__(self,feat_dim:int,d_model:int) -> None:
        super().__init__()
        self.aff_feat = nn.Linear(feat_dim,d_model)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self,x:torch.Tensor):
        return self.relu(self.aff_feat(x))
    
class FeatDecoder(nn.Module):
    def __init__(self, input_dim:int,output_dim:int,vq_dim:int) -> None:
        super().__init__()
        self.aff_vq = nn.Linear(vq_dim,input_dim)
        self.rec = nn.Linear(input_dim*2,output_dim)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self,feat:torch.Tensor,vq:torch.Tensor):
        aff_vq = self.relu(self.aff_vq(vq))
        rec_feat = torch.cat([aff_vq,feat],dim=-1)
        rec = self.rec(rec_feat)
        return rec

