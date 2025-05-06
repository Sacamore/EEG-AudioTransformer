import numpy as np

import torch
import torch.utils.data as utils_data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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

def subsequent_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len),diagonal=1)).bool()
    return mask

class TorchTransformer(nn.Module):
    def __init__(self,input_dim,output_dim,d_model,nhead,nlayer,hidden_dim,dropout) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.encoder_embedding = nn.Linear(input_dim,d_model)
        self.decoder_embedding = nn.Linear(output_dim,d_model)
        # self.transformer = nn.Transformer(d_model=d_model,nhead=nhead,num_encoder_layers=nlayer,num_decoder_layers=nlayer,dim_feedforward=hidden_dim,dropout=dropout,batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=hidden_dim,dropout=dropout,batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=hidden_dim,dropout=dropout,batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=nlayer)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,num_layers=nlayer)
        self.generator =nn.Sequential(
            nn.Linear(d_model,d_model*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(d_model*2,output_dim)
        ) 
        

    def forward(self,src,tgt):
        emb_src = self.encoder_embedding(src)
        emb_src = self.positional_encoding(emb_src)

        emb_tgt = self.decoder_embedding(tgt)
        emb_tgt = self.positional_encoding(emb_tgt)

        dec_subseq_mask = subsequent_mask(emb_tgt.shape[1]).to(emb_tgt.device)

        encoded = self.encoder(emb_src)
        decoded = self.decoder(emb_tgt,encoded,dec_subseq_mask)
        
        output = self.generator(decoded)

        return output

    def infer(self,src,tgt):
        self.tgt_len = tgt.shape[1]
        emb_src = self.encoder_embedding(src)
        emb_src = self.positional_encoding(emb_src)
        encoded = self.encoder(emb_src)

        dec_input = tgt
        next_symbol = torch.zeros(src.shape[0],self.output_dim).type_as(src.data)
        for i in range(0,self.tgt_len):
            dec_input[:,i,:] = next_symbol
            emb_dec = self.decoder_embedding(dec_input)
            dec_outputs = self.decoder(emb_dec,encoded)
            output = self.generator(dec_outputs)
            next_symbol = output[:,i,:]

        return output
    

from model.BaseModel import BaseModelHolder

class TransformerModelHolder(BaseModelHolder):
    def __init__(self,d_model,nhead,nlayer,hidden_dim,dropout,lr):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayer = nlayer
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.loss_fn = nn.MSELoss()
        self.lr = lr
    
    def buildModel(self):
        self.model = TorchTransformer(self.input_dim,self.output_dim,self.d_model,self.nhead,self.nlayer,self.hidden_dim,self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
    
    def makeDataloader(self, batch_size, train_data, train_label, test_data, test_label):
        train_encoder_data = train_data
        train_decoder_input_data = np.pad(train_label,((0,0),(1,0),(0,0)),mode='constant',constant_values=0)
        train_decoder_output_data = np.pad(train_label,((0,0),(0,1),(0,0)),mode='constant',constant_values=1)
        
        test_encoder_data = test_data
        test_decoder_input_data = np.pad(np.zeros(test_label.shape),((0,0),(1,0),(0,0)),mode='constant',constant_values=0)
        test_decoder_output_data = np.pad(test_label,((0,0),(0,1),(0,0)),mode='constant',constant_values=1)
        
        self.input_dim = train_encoder_data.shape[-1]
        self.output_dim = train_decoder_output_data.shape[-1]

        train_encoder_data = torch.from_numpy(train_encoder_data)
        train_decoder_input_data = torch.from_numpy(train_decoder_input_data)
        train_decoder_output_data = torch.from_numpy(train_decoder_output_data)
        test_encoder_data = torch.from_numpy(test_encoder_data)
        test_decoder_input_data = torch.from_numpy(test_decoder_input_data)
        test_decoder_output_data = torch.from_numpy(test_decoder_output_data)
        train_dataset = TensorDataset(train_encoder_data,train_decoder_input_data,train_decoder_output_data)
        test_dataset = TensorDataset(test_encoder_data,test_decoder_input_data,test_decoder_output_data)

        self.train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=False)
        self.test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,pin_memory=False)
        
    def train(self):
        self._toTrain([self.model])
        train_loss = {'loss':0}
        for _, (eeg, mel_in, mel_out) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            eeg = self._toDevice(eeg)
            mel_in = self._toDevice(mel_in)
            mel_out = self._toDevice(mel_out)

            decoded_mel = self.model(eeg, mel_in)
            loss = self.loss_fn(decoded_mel, mel_out)
            loss.backward()
            self.optimizer.step()
            train_loss['loss'] += loss.item()
        train_loss['loss'] = train_loss['loss'] / len(self.train_dataloader)
        return train_loss
    
    def predict(self):
        self._toEval([self.model])
        output_mel = []
        test_loss = 0
        with torch.no_grad():
            for _, (eeg, mel_in, mel_out) in enumerate(self.test_dataloader):
                eeg = self._toDevice(eeg)
                mel_in = self._toDevice(mel_in)
                mel_out = self._toDevice(mel_out)
                decoded_mel = self.model.infer(eeg, mel_in)
                test_loss += self.loss_fn(decoded_mel, mel_out).detach().cpu().numpy()
                output_mel.append(decoded_mel.detach().cpu().numpy())

        output_mel = np.concatenate(output_mel, axis=0)
        test_loss = test_loss / len(self.test_dataloader)
        
        return test_loss, output_mel
    
    def loadModel(self,state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        
    def saveModel(self,e):
        state_dict = {
            'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'epoch':e
        }
        return state_dict
        