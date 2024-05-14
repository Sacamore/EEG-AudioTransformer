import numpy as np
from typing import Optional,List

import copy
import torch
import torch.utils.data as utils_data
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建一个 max_len x d_model 的矩阵 P
        pe = np.zeros((max_len, d_model))
        
        # 计算每个位置的值
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # 增加一个维度，以适应 batch 的处理
        pe = pe[np.newaxis, ...]
        
        # 转换为 Tensor
        self.register_buffer('pe',torch.tensor(pe, dtype=torch.float32))
        # self.pe = torch.tensor(pe, dtype=torch.float32)
    
    def forward(self, x):
        # x 的形状应该是 [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x
    
import math
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # self.linear = nn.Embedding(n_vocab, d_model)
        self.d_s = math.sqrt(d_model)
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)
    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]
        return x * self.d_s + pe

class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, heads:int,d_k:int,bias:bool) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model,heads * d_k,bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x:torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape,self.heads,self.d_k)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads:int, d_model:int,dropout:float=0.1,bias:bool=True) -> None:
        super().__init__()
        self.d_k    = d_model//heads
        self.heads  = heads

        self.query  = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.key    = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.value  = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)

        self.softmax = nn.Softmax(dim=2)

        self.output = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout)
        
        self.scale = 1/np.sqrt(self.d_k)

        self.attn = None

    def get_scores(self,query:torch.Tensor,key:torch.Tensor):
        return torch.einsum('bihd,bjhd -> bijh',query,key)
    
    def prepare_mask(self,mask: torch.Tensor, query_shape:List[int],key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[1]
        assert mask.shape[1] == key_shape[1]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[0]

        mask = mask.unsqueeze(0)

        return mask

    def forward(self,*,
                query:torch.Tensor,
                key:torch.Tensor,
                value:torch.Tensor,
                mask:Optional[torch.Tensor] = None):
        
        batch_size,seq_len,_ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask,query.shape,key.shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores =self.get_scores(query,key)*self.scale

        if mask is not None:
            scores = scores.masked_fill(mask==0,float('-inf'))

        attn = self.softmax(scores)

        attn = self.dropout(attn)

        x = torch.einsum('bijh,bjhd -> bihd',attn,value)

        self.attn = attn.detach()

        x = x.reshape(batch_size,seq_len,-1)

        return self.output(x)
    

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int,base: int = 10000) -> None:
        super().__init__()

        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self,x:torch.Tensor):
        if self.cos_cached is not None and x.shape[1] <= self.cos_cached.shape[0]:
            return
        seq_len = x.shape[1]

        theta = 1. / (self.base ** (torch.arange(0,self.d,2).float()/self.d)).to(x.device)

        seq_idx = torch.arange(seq_len,device=x.device).float().to(x.device)

        idx_theta = torch.einsum('n,d -> nd',seq_idx,theta)

        idx_theta2 = torch.cat([idx_theta,idx_theta],dim=1)

        self.cos_cached = idx_theta2.cos()[None,:,None,:]
        self.sin_cached = idx_theta2.sin()[None,:,None,:]

    def _neg_half(self,x:torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[:,:,:,d_2:], x[:,:,:,:d_2]],dim=-1)
    
    def forward(self,x:torch.Tensor):
        self._build_cache(x)
        x_rope,x_pass = x[..., :self.d],x[..., self.d:]

        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[:x.shape[1]]) + (neg_half_x*self.sin_cached[:x.shape[1]])

        return torch.cat((x_rope,x_pass),dim=-1)
    

class RotaryPEMultiHeadAttention(MultiHeadAttention):
    def __init__(self, heads: int, d_model: int,rope_percentage:float = 0.5, dropout: float = 0.1, bias: bool = True) -> None:
        super().__init__(heads, d_model, dropout, bias)
        d_rope = int(self.d_k * rope_percentage)
        self.query_rotary_pe = RotaryPositionalEmbeddings(d_rope)

        self.key_rotary_pe = RotaryPositionalEmbeddings(d_rope)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('bihd,bjhd->bijh',self.query_rotary_pe(query),self.key_rotary_pe(key))
    

class FeedForward(nn.Module):
    def __init__(self,
                d_model:int,
                d_ff:int,
                dropout:float=0.1,
                activation = nn.ReLU(),
                is_gated:bool=False,
                bias1:bool=True,
                bias2:bool=True,
                bias_gate:bool=True):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model,d_ff,bias1)
        self.layer2 = nn.Linear(d_ff,d_model,bias2)

        self.dropout = nn.Dropout(dropout)

        self.activation = activation

        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model,d_ff,bias_gate)

    def forward(self,x:torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g*self.linear_v(x)
        else:
            x = g
        x= self.dropout(x)

        return self.layer2(x)

class ConvFeedForward(FeedForward):
    def __init__(self, d_model: int, d_ff: int, conv_kernel_size: int = 9, conv_padding: int = 4 ,dropout: float = 0.1, activation=nn.ReLU(), is_gated: bool = False, bias1: bool = True, bias2: bool = True, bias_gate: bool = True):
        super().__init__(d_model, d_ff, dropout, activation, is_gated, bias1, bias2, bias_gate)
        self.layer1 = nn.Conv1d(d_model,d_ff,conv_kernel_size,1,conv_padding)
        self.layer2 = nn.Conv1d(d_ff,d_model,1,1)
    def forward(self,x:torch.Tensor):
        x = x.permute(0,2,1)
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g*self.linear_v(x)
        else:
            x = g
        x= self.dropout(x)
        x = self.layer2(x)
        x = x.permute(0,2,1)
        return x

        

class TransformerLayer(nn.Module):
    def __init__(self,*,
                 d_model:int,
                 self_attn:MultiHeadAttention,
                 src_attn:MultiHeadAttention = None,
                 feed_forward:FeedForward,
                 dropout:float=0.1
                 ) -> None:
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
    
    def forward(self,*,
                x:torch.Tensor,
                mask:torch.Tensor,
                src:torch.Tensor = None,
                src_mask: torch.Tensor = None):

        self_attn = self.self_attn(query=x,key=x,value=x,mask=mask)
        x = x + self.dropout(self_attn)
        x = self.norm_self_attn(x)

        if src is not None:
            attn_src = self.src_attn(query = x, key = src, value = src, mask=src_mask)
            x = x + self.dropout(attn_src)
            x = self.norm_src_attn(x)
        
        ff = self.feed_forward(x)
        x = x + self.dropout(ff)
        x = self.norm_ff(x)

        return x

class Encoder(nn.Module):
    def __init__(self,layer:TransformerLayer,n_layers:int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        # self.norm = nn.LayerNorm([layer.size])

    def forward(self,x:torch.Tensor,mask:torch.Tensor):
        for layer in self.layers:
            x = layer(x=x,mask=mask)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, layer: TransformerLayer,n_layers:int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        # self.norm = nn.LayerNorm([layer.size])
    def forward(self, x:torch.Tensor,memory:torch.Tensor,src_mask:torch.Tensor,tgt_mask:torch.Tensor):
        for layer in self.layers:
            x = layer(x=x,mask=tgt_mask,src=memory,src_mask=src_mask)
        
        return x
    
class Transformer(nn.Module):
    def __init__(self,
                 encoder:Encoder,
                 decoder:Decoder,
                 src_embed:nn.Module,
                 tgt_embed:nn.Module
                 ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
    
    def encode(self,src:torch.Tensor,src_mask:torch.Tensor):
        return self.encoder(self.src_embed(src),src_mask)

    def decode(self,memory:torch.Tensor,src_mask:torch.Tensor,tgt:torch.Tensor,tgt_mask:torch.Tensor):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)

    def forward(self,src:torch.Tensor,tgt:torch.Tensor,src_mask:torch.Tensor,tgt_mask:torch.Tensor):
        enc = self.encode(src,src_mask)
        return self.decode(enc,src_mask,tgt,tgt_mask)  

def subsequent_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).to(torch.bool).unsqueeze(-1)
    return mask