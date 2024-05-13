import torch
import torch.nn as nn
import torch.nn.functional as F
import model.transformer as transformer
import copy
import numpy as np

class TransformerEncoder(nn.Module):
    def __init__(self,input_dim:int,d_model:int,nhead:int=4,n_layer:int=2,dropout:float=0.1) -> None:
        super().__init__()
        self.aff_input = nn.Linear(input_dim,d_model)
        self.RoPEMultiHeadAttention = transformer.RotaryPEMultiHeadAttention(heads=nhead,d_model=d_model,rope_percentage=1.0,dropout=dropout)
        # self.MultiHeadAttention = transformer.MultiHeadAttention(heads=nhead,d_model=d_model,dropout=dropout)
        self.ff = transformer.ConvFeedForward(d_model=d_model,d_ff=4*d_model,activation=nn.GELU(),dropout=dropout)
        self.encoderLayer = transformer.TransformerLayer(d_model=d_model,self_attn=self.RoPEMultiHeadAttention,feed_forward=self.ff,dropout=dropout)
        self.encoder = transformer.Encoder(self.encoderLayer,n_layers=n_layer)
        self.src_mask = None
    def forward(self,x:torch.Tensor):
        x = self.aff_input(x)
        x = self.encoder(x,self.src_mask)
        return x

class ResidualLayer(nn.Module):
    def __init__(self,input_dim:int,output_dim:int,hidden_dim:int) -> None:
        super().__init__()
        self.res_block = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(input_dim,hidden_dim,kernel_size=3,stride=1,padding=1,bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim,output_dim,kernel_size=1,stride=1,bias=False)
        )
    def forward(self, x:torch.Tensor):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    def __init__(self,input_dim:int,output_dim:int,d_model:int,hidden_dim:int,n_layers:int) -> None:
        super().__init__()
        self.input_block = ResidualLayer(input_dim,d_model,hidden_dim)
        self.stack = nn.ModuleList([copy.deepcopy(ResidualLayer(d_model,d_model,hidden_dim)) for _ in range(n_layers)])
        self.output_block = ResidualLayer(d_model,output_dim,hidden_dim)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self,x:torch.Tensor):
        x = self.input_block(x)
        for layer in self.stack:
            x = layer(x)
        x = self.relu(self.output_block(x))
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost:float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.epsilon = 1e-10
        self.decay = 0.99
        embeddings = torch.Tensor(num_embeddings,embedding_dim)
        embeddings.uniform_(-1/self.num_embeddings,1/self.num_embeddings)
        self.register_buffer("embeddings",embeddings)
        self.register_buffer("ema_count",torch.zeros(num_embeddings))
        self.register_buffer("ema_weight",self.embeddings.clone())
        # self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        # self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.register_buffer('unactivated_count',-torch.ones(num_embeddings))

    def getDistances(self,flat:torch.Tensor):
        return torch.addmm( torch.sum(self.embeddings ** 2,dim=1)+torch.sum(flat ** 2,dim=1,keepdim=True),
                            flat,self.embeddings.T,
                            alpha=-2.0,beta=1.0)

    def forward(self, inputs):
        # Flatten input
        input_shape = inputs.shape
        flat_inputs = inputs.detach().view(-1, self.embedding_dim)

        # Calculate distances
        distances = self.getDistances(flat_inputs)
        # dis_grad = self.getDistances(inputs.reshape(-1, self.embedding_dim))
        # Encoding
        indices = torch.argmin(distances, dim=-1)
        encodings = F.one_hot(indices,self.num_embeddings).float()
        quantized = F.embedding(indices,self.embeddings)

        if self.training:
            self.ema_count = self.decay *self.ema_count +(1-self.decay)*torch.sum(encodings,dim=0)
            tmp = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon)/(tmp + self.num_embeddings * self.epsilon) * tmp
            dw = torch.matmul(encodings.T,flat_inputs)
            self.ema_weight = self.decay * self.ema_weight + (1-self.decay) * dw
            self.embeddings = self.ema_weight / self.ema_count.unsqueeze(-1)

        self.unactivated_count = self.unactivated_count + 1
        for i in indices:
            self.unactivated_count[i.item()] = 0
        for i,x in enumerate(self.unactivated_count):
            if x > 300:
                self.embeddings[i] = torch.Tensor(self.embedding_dim).uniform_(-1/self.embedding_dim,1/self.embedding_dim).cuda()
                self.unactivated_count[i] = 0
        # Reshape quantized to match input shape
        quantized = quantized.view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(),inputs)
        q_latent_loss = F.mse_loss(quantized,inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return loss, quantized #, indices.view(input_shape[0], input_shape[1])

class MelEncoder(nn.Module):
    def __init__(self, input_dim, seg_size, embedding_dim:int=1024):
        super(MelEncoder, self).__init__()
        assert embedding_dim%seg_size == 0
        self.d_model= embedding_dim // seg_size * 4
        self.flat_dim = self.d_model * seg_size
        self.TransformerEncoder = TransformerEncoder(input_dim=input_dim,d_model=self.d_model)
        # self.linear =nn.Sequential(
        #     nn.Linear(self.flat_dim,self.flat_dim//2),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.flat_dim//2,embedding_dim)
        # )
        # self.ResEncoder = ResidualStack(input_dim=1,output_dim=embedding_dim,d_model=embedding_dim//2,hidden_dim=embedding_dim,n_layers=1)

    def forward(self, inputs:torch.Tensor):
        transformer_encoded = self.TransformerEncoder(inputs)
        flat_encoded = transformer_encoded.view(-1,self.flat_dim)
        encoded = flat_encoded#self.linear(flat_encoded)
        return encoded

class MelDecoder(nn.Module):
    def __init__(self,output_dim,seg_size,embedding_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.seg_size = seg_size
        self.d_model= embedding_dim // seg_size * 4
        self.flat_dim = self.d_model * seg_size
        self.decoder = nn.Sequential(
            nn.Linear(self.flat_dim,self.flat_dim//2),
            nn.LeakyReLU(),
            nn.Linear(self.flat_dim//2,output_dim*seg_size)
        )
    def forward(self,vq:torch.Tensor):
        flat_decoded = self.decoder(vq)
        decoded = flat_decoded.view(-1,self.seg_size,self.output_dim)
        return decoded
    
class EEGEncoder(nn.Module):
    def __init__(self, input_dim, seg_size, embedding_dim:int=1024):
        super(EEGEncoder, self).__init__()
        assert embedding_dim%seg_size == 0
        self.d_model= embedding_dim // seg_size * 4
        self.flat_dim = self.d_model * seg_size
        self.TransformerEncoder = TransformerEncoder(input_dim=input_dim,d_model=self.d_model)

    def forward(self, inputs:torch.Tensor):
        transformer_encoded = self.TransformerEncoder(inputs)
        flat_encoded = transformer_encoded.view(-1,self.flat_dim)
        encoded = flat_encoded
        return encoded
    
class CLIP(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('label',torch.arange(0,batch_size))

    def forward(self,encoded_eeg:torch.Tensor,encoded_mel:torch.Tensor):
        batch_size = encoded_eeg.shape[0]
        encoded_eeg = encoded_eeg / encoded_eeg.norm(dim=-1,keepdim=True)
        encoded_mel = encoded_mel / encoded_mel.norm(dim=-1,keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_eeg = logit_scale * torch.matmul(encoded_eeg,encoded_mel.T)
        loss = F.cross_entropy(logits_eeg,self.label[:batch_size])
        return loss