import torch
import torch.nn as nn
import torch.nn.functional as F
import model.transformer as transformer
import copy
import numpy as np


class RoPETransformerEncoder(nn.Module):
    def __init__(self,input_dim:int,d_model:int,nhead:int=4,n_layer:int=2,dropout:float=0.1) -> None:
        super().__init__()
        self.aff_input = nn.Linear(input_dim,d_model)
        self.RoPEMultiHeadAttention = transformer.RotaryPEMultiHeadAttention(heads=nhead,d_model=d_model,rope_percentage=1.0,dropout=dropout)
        # self.MultiHeadAttention = transformer.MultiHeadAttention(heads=nhead,d_model=d_model,dropout=dropout)
        # self.ff = transformer.ConvFeedForward(d_model=d_model,d_ff=4*d_model,activation=nn.GELU(),dropout=dropout)
        self.ff = transformer.FeedForward(d_model=d_model,d_ff=4*d_model,activation=nn.GELU(),dropout=dropout)
        self.encoderLayer = transformer.TransformerLayer(d_model=d_model,self_attn=self.RoPEMultiHeadAttention,feed_forward=self.ff,dropout=dropout)
        self.encoder = transformer.Encoder(self.encoderLayer,n_layers=n_layer)
        self.src_mask = None
    def forward(self,x:torch.Tensor):
        x = self.aff_input(x)
        x = self.encoder(x,self.src_mask)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self,d_model:int,nhead:int=4,n_layer:int=2,dropout:float=0.1) -> None:
        super().__init__()
        # self.aff_input = nn.Linear(input_dim,d_model)
        self.MultiHeadAttention = transformer.MultiHeadAttention(heads=nhead,d_model=d_model,dropout=dropout)
        # self.MultiHeadAttention = transformer.MultiHeadAttention(heads=nhead,d_model=d_model,dropout=dropout)
        self.ff = transformer.FeedForward(d_model=d_model,d_ff=4*d_model,activation=nn.GELU(),dropout=dropout)
        self.encoderLayer = transformer.TransformerLayer(d_model=d_model,self_attn=self.MultiHeadAttention,feed_forward=self.ff,dropout=dropout)
        self.encoder = transformer.Encoder(self.encoderLayer,n_layers=n_layer)
        self.src_mask = None
    def forward(self,x:torch.Tensor):
        x = self.encoder(x,self.src_mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,d_model:int,nhead:int=4,n_layer:int=2,dropout:float=0.1):
        super().__init__()

    

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
    def __init__(self,input_dim:int,d_model:int,hidden_dim:int,n_layers:int) -> None:
        super().__init__()
        # self.input_block = ResidualLayer(input_dim,d_model,hidden_dim)
        self.stack = nn.ModuleList([copy.deepcopy(ResidualLayer(input_dim,d_model,hidden_dim)) for _ in range(n_layers)])
        # self.output_block = ResidualLayer(d_model,output_dim,hidden_dim)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self,x:torch.Tensor):
        # x = self.input_block(x)
        for layer in self.stack:
            x = layer(x)
        x = self.relu(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost:float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.epsilon = 1e-10
        self.decay = 0.99
        self.reset_num = 0
        embeddings = torch.Tensor(num_embeddings,embedding_dim)
        embeddings.uniform_(-1/self.num_embeddings,1/self.num_embeddings)
        self.register_buffer("embeddings",embeddings)
        self.register_buffer("ema_count",torch.zeros(num_embeddings))
        self.register_buffer("ema_weight",self.embeddings.clone())
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07),requires_grad=True)
        # self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        # self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.register_buffer('unactivated_count',-torch.ones(num_embeddings))

    # def cls(self,encoded_eeg:torch.Tensor):
    #     bs,d_model = encoded_eeg.shape
    #     encoded_eeg = encoded_eeg.reshape(-1,self.embedding_dim)
    #     encoded_norm_eeg = encoded_eeg / (encoded_eeg.norm(dim=-1,keepdim=True))#+1e-10)
    #     vq_mel_norm_table = self.embeddings / (self.embeddings.norm(dim=-1,keepdim=True))#+1e-10)
    #     logit_scale = self.logit_scale.exp()
    #     logits = logit_scale * torch.matmul(encoded_norm_eeg,vq_mel_norm_table.t())
    #     indices = logits.argmax(dim=-1)
    #     vq_mel =torch.empty_like(encoded_eeg)
    #     vq_mel = self.embeddings[indices]
    #     vq_mel = vq_mel.reshape(bs,d_model)
    #     return vq_mel

    # def cls_loss(self,encoded_eeg:torch.Tensor,encoded_mel:torch.Tensor):
    #     # bs,d_model = encoded_eeg.shape
    #     flat_encoded_mel = encoded_mel.detach().view(-1, self.embedding_dim)
    #     distances = self.getDistances(flat_encoded_mel)
    #     mel_indices = torch.argmin(distances, dim=-1)
        
    #     flat_encoded_eeg = encoded_eeg.reshape(-1,self.embedding_dim)
    #     encoded_norm_eeg = flat_encoded_eeg / (flat_encoded_eeg.norm(dim=-1,keepdim=True))#+1e-10)
        
    #     vq_mel_norm_table = self.embeddings / (self.embeddings.norm(dim=-1,keepdim=True))#+1e-10)
        
    #     logit_scale = self.logit_scale.exp()
    #     logits = logit_scale * torch.matmul(encoded_norm_eeg,vq_mel_norm_table.t())
    #     eeg_indices = logits.detach().argmax(dim=-1)
    #     acc_rate = torch.eq(eeg_indices,mel_indices).float().mean()
    #     loss = F.cross_entropy(logits,mel_indices)
    #     return loss,acc_rate


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

            activated_indices = []
            unactivated_indices = []

            for i,x in enumerate(self.unactivated_count):
                if x > 300:
                    unactivated_indices.append(i)
                    self.unactivated_count[i] = 0
                elif x>=0 and x<100:
                    activated_indices.append(i)
            
            if len(activated_indices) != 0 and len(unactivated_indices) != 0:
                activated_quantized = F.embedding(torch.tensor(activated_indices,dtype=torch.long).cuda(),self.embeddings)
                self.embeddings[unactivated_indices] = activated_quantized[torch.randint(0,len(activated_indices),(1,len(unactivated_indices)))] + torch.Tensor(len(unactivated_indices),self.embedding_dim).uniform_(-1/self.num_embeddings,1/self.num_embeddings).cuda()
                self.reset_num+=len(unactivated_indices)
            
        # Reshape quantized to match input shape
        quantized = quantized.view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(),inputs)
        q_latent_loss = F.mse_loss(quantized,inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return loss, quantized #, indices.view(-1,1)
class MaskMelEncoder(nn.Module):
    def __init__(self, input_dim,d_model, seg_size):
        super().__init__()
        # assert embedding_dim%seg_size == 0
        # self.d_model= embedding_dim // seg_size * 4
        self.flat_dim = d_model * seg_size
        self.aff = nn.Linear(input_dim,d_model)
        self.TransformerEncoder = TransformerEncoder(d_model=d_model)

    def forward(self, inputs:torch.Tensor,pe:torch.Tensor):
        bs = inputs.shape[0]
        aff = self.aff(inputs)
        aff += pe
        transformer_encoded = self.TransformerEncoder(aff)
        # flat_encoded = transformer_encoded.view(bs,-1)
        # encoded = flat_encoded#self.linear(flat_encoded)
        return transformer_encoded
    
class MaskEEGEncoder(nn.Module):
    def __init__(self, input_dim,d_model, seg_size):
        super().__init__()
        self.flat_dim = d_model * seg_size
        self.aff = nn.Linear(input_dim,d_model)
        self.TransformerEncoder = TransformerEncoder(d_model=d_model)

    def forward(self, inputs:torch.Tensor,pe:torch.Tensor):
        bs = inputs.shape[0]
        aff = self.aff(inputs)
        aff += pe
        transformer_encoded = self.TransformerEncoder(aff)
        # flat_encoded = transformer_encoded.view(bs,-1)
        # encoded = flat_encoded#self.linear(flat_encoded)
        return transformer_encoded

class MaskMelDecoder(nn.Module):
    def __init__(self,output_dim,d_model,seg_size) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.seg_size = seg_size
        self.decoder = TransformerEncoder(d_model=d_model)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model,2*d_model),
            nn.LeakyReLU(),
            nn.Linear(2*d_model,output_dim)
        )
    def forward(self,vq:torch.Tensor):
        decoded = self.decoder(vq)
        res = self.output_layer(decoded)
        # decoded = flat_decoded.view(-1,self.seg_size,self.output_dim)
        return res

class MelEncoder(nn.Module):
    def __init__(self, input_dim,d_model):
        super().__init__()
        # self.flat_dim = d_model * seg_size
        self.TransformerEncoder = RoPETransformerEncoder(input_dim=input_dim,d_model=d_model)
    
    def forward(self, inputs:torch.Tensor):
        bs = inputs.shape[0]
        transformer_encoded = self.TransformerEncoder(inputs)
        flat_encoded = transformer_encoded.view(bs,-1)
        return flat_encoded
class MelLinearDecoder(nn.Module):
    def __init__(self,output_dim,d_model,seg_size) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.seg_size = seg_size
        self.d_model = d_model
        # self.decoder = TransformerEncoder(d_model=d_model)
        self.flat_dim = seg_size*d_model
        self.decoder = nn.Sequential(
            nn.Linear(self.flat_dim,self.flat_dim),
            nn.LeakyReLU(),
            nn.Linear(self.flat_dim,output_dim*seg_size)
        )
    def forward(self,vq:torch.Tensor):
        # vq = vq.view(-1,self.seg_size,self.d_model)
        # decoded = self.decoder(vq)
        decoded =vq.view(-1,self.flat_dim)
        flat_decoded = self.decoder(decoded)
        res = flat_decoded.view(-1,self.seg_size,self.output_dim)
        return res
    
class MelDecoder(nn.Module):
    def __init__(self,output_dim,d_model,seg_size) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.seg_size = seg_size
        self.d_model = d_model
        self.decoder = TransformerEncoder(d_model=d_model)
        self.flat_dim = seg_size*d_model
        self.linear = nn.Sequential(
            nn.Linear(self.flat_dim,self.flat_dim),
            nn.LeakyReLU(),
            nn.Linear(self.flat_dim,output_dim*seg_size)
        )
    def forward(self,vq:torch.Tensor):
        vq = vq.view(-1,self.seg_size,self.d_model)
        decoded = self.decoder(vq)
        decoded =decoded.view(-1,self.flat_dim)
        flat_decoded = self.linear(decoded)
        res = flat_decoded.view(-1,self.seg_size,self.output_dim)
        return res

class EEGDecoder(nn.Module):
    def __init__(self,output_dim,d_model,seg_size) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.seg_size = seg_size
        self.d_model = d_model
        self.decoder = TransformerEncoder(d_model=d_model)
        self.flat_dim = seg_size*d_model
        self.linear = nn.Sequential(
            nn.Linear(self.flat_dim,self.flat_dim),
            nn.LeakyReLU(),
            nn.Linear(self.flat_dim,output_dim*seg_size)
        )
    def forward(self,vq:torch.Tensor):
        vq = vq.view(-1,self.seg_size,self.d_model)
        decoded = self.decoder(vq)
        decoded =decoded.view(-1,self.flat_dim)
        flat_decoded = self.linear(decoded)
        res = flat_decoded.view(-1,self.seg_size,self.output_dim)
        return res

class EEGEncoder(nn.Module):
    def __init__(self, input_dim,d_model,n_layer=2,nhead=2):
        super().__init__()
        # self.flat_dim = d_model * seg_size
        self.TransformerEncoder = RoPETransformerEncoder(input_dim=input_dim,d_model=d_model,n_layer=n_layer,nhead=nhead,dropout=0.1)
        # self.linear = nn.Linear(eeg_d_model,mel_d_model)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1,16,3,1,1),
        #     nn.BatchNorm2d(16),
        #     nn.Conv2d(16,8,5,2,2),
        #     nn.BatchNorm2d(8),
        #     nn.Conv2d(8,4,3,2,1),
        #     nn.BatchNorm2d(4)
        # )

    def forward(self, inputs:torch.Tensor):
        bs = inputs.shape[0]
        transformer_encoded = self.TransformerEncoder(inputs)
        # transformer_encoded = self.linear(transformer_encoded)
        # conv_encoded = transformer_encoded.unsqueeze(1)
        # conv_encoded = self.conv(conv_encoded)
        flat_encoded = transformer_encoded.view(bs,-1)
        return flat_encoded

class EEGConvEncoder(nn.Module):
    def __init__(self, input_dim,d_model,n_layer):
        super().__init__()
        # self.flat_dim = d_model * seg_size
        # self.TransformerEncoder = RoPETransformerEncoder(input_dim=input_dim,d_model=4*d_model)
        self.embed = nn.Linear(input_dim,d_model)
        #B,16,256 -> B,4,4,64
        self.conv = nn.Sequential(
            nn.Conv2d(1,d_model//4,3,1,1),
            nn.BatchNorm2d(d_model//4),
            ResidualStack(d_model//4,d_model//4,d_model//2,n_layer),
        )
        self.pooling = nn.AvgPool2d((1,d_model//4))

    def forward(self, inputs:torch.Tensor):
        bs = inputs.shape[0]
        embed_encoded = self.embed(inputs)
        conv_encoded = embed_encoded.unsqueeze(1)
        conv_encoded = self.conv(conv_encoded)
        conv_encoded = conv_encoded.permute(0,2,3,1)
        pooling_encoded = self.pooling(conv_encoded)
        flat_encoded = pooling_encoded.view(bs,-1)
        return flat_encoded

class CLIP(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('label',torch.arange(0,batch_size))

    def clip_forward(self,encoded_eeg:torch.Tensor,vq_mel_table:torch.Tensor):
        n_embed,embed_dim = vq_mel_table.shape
        bs,d_model = encoded_eeg.shape
        encoded_eeg = encoded_eeg.reshape(-1,embed_dim)

        encoded_norm_eeg = encoded_eeg / encoded_eeg.norm(dim=-1,keepdim=True)
        vq_mel_norm_table = vq_mel_table / vq_mel_table.norm(dim=-1,keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(encoded_norm_eeg,vq_mel_norm_table.t())
        indices = logits.argmax(dim=-1)
        vq_mel =torch.empty_like(encoded_eeg)
        vq_mel = vq_mel_table[indices]
        vq_mel = vq_mel.reshape(bs,d_model)
        return vq_mel

    def forward(self,encoded_eeg:torch.Tensor,encoded_mel:torch.Tensor):
        batch_size = encoded_eeg.shape[0]
        encoded_eeg = encoded_eeg.reshape(batch_size,-1)
        encoded_mel = encoded_mel.reshape(batch_size,-1)
        encoded_eeg = encoded_eeg / encoded_eeg.norm(dim=-1,keepdim=True)
        encoded_mel = encoded_mel / encoded_mel.norm(dim=-1,keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_eeg = logit_scale * torch.matmul(encoded_eeg,encoded_mel.T)
        logits_mel = logits_eeg.T
        e_loss = F.cross_entropy(logits_eeg,self.label[:batch_size])
        m_loss = F.cross_entropy(logits_mel,self.label[:batch_size])
        return e_loss,m_loss
    
class CosSimClassifier(nn.Module):
    def __init__(self,num_embedding:int,embedding_dim:int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer("embeddings",torch.Tensor(num_embedding,embedding_dim))
        self.cls_center = nn.Parameter(torch.Tensor(1,embedding_dim).uniform_(-1/embedding_dim,1/embedding_dim))
        # self.register_buffer("norm_embeddings_t",(self.embeddings / (self.embeddings.norm(dim=-1,keepdim=True))).t())
    def initEmbeddings(self,embeddings):
        self.embeddings = embeddings.detach()
    
    def getDistances(self,flat:torch.Tensor):
        return torch.addmm( torch.sum(self.embeddings ** 2,dim=1)+torch.sum(flat ** 2,dim=1,keepdim=True),
                            flat,self.embeddings.T,
                            alpha=-2.0,beta=1.0)
    
    def cls(self,encoded_eeg:torch.Tensor):
        bs,d_model = encoded_eeg.shape
        flat_encoded_eeg = encoded_eeg.reshape(-1,self.embedding_dim)
        encoded_norm_eeg = self.getNorm(flat_encoded_eeg)
        vq_mel_norm_table = self.getNorm(self.embeddings)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(encoded_norm_eeg,vq_mel_norm_table.t())
        indices = logits.argmax(dim=-1)
        vq_mel =torch.empty_like(encoded_eeg)
        vq_mel = self.embeddings[indices]
        vq_mel = vq_mel.reshape(bs,d_model)
        return vq_mel

    def getNorm(self,flat:torch.Tensor):
        # return (flat-self.cls_center)/(flat-self.cls_center).norm(dim=-1,keepdim=True)
        return flat/flat.norm(dim=-1,keepdim=True)

    def forward(self,encoded_eeg:torch.Tensor,encoded_mel:torch.Tensor):
        # bs,d_model = encoded_eeg.shape
        flat_encoded_mel = encoded_mel.detach().view(-1, self.embedding_dim)
        distances = self.getDistances(flat_encoded_mel)
        mel_indices = torch.argmin(distances, dim=-1)
        
        flat_encoded_eeg = encoded_eeg.reshape(-1,self.embedding_dim)
        encoded_norm_eeg = self.getNorm(flat_encoded_eeg)
        vq_mel_norm_table = self.getNorm(self.embeddings)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(encoded_norm_eeg,vq_mel_norm_table.t())
        eeg_indices = logits.detach().argmax(dim=-1)
        acc_rate = torch.eq(eeg_indices,mel_indices).float().mean()
        loss = F.cross_entropy(logits,mel_indices)
        return loss,acc_rate

class LinearClassifier(nn.Module):
    def __init__(self,num_embedding:int,embedding_dim:int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim),
            nn.LeakyReLU(),
            # nn.Linear(embedding_dim,embedding_dim),
            # nn.LeakyReLU(),
            nn.Linear(embedding_dim,num_embedding),
            # nn.Sigmoid()
            # nn.Tanh()
        )
        self.register_buffer("embeddings",torch.Tensor(num_embedding,embedding_dim))
    def initEmbeddings(self,embeddings):
        self.embeddings = embeddings.detach()
    def getDistances(self,flat:torch.Tensor):
        return torch.addmm( torch.sum(self.embeddings ** 2,dim=1)+torch.sum(flat ** 2,dim=1,keepdim=True),
                            flat,self.embeddings.T,
                            alpha=-2.0,beta=1.0)
    
    def cls(self,encoded_eeg:torch.Tensor):
        bs,d_model = encoded_eeg.shape
        flat_encoded_eeg = encoded_eeg.reshape(-1,self.embedding_dim)
        logits = self.linear(flat_encoded_eeg)
        indices = logits.argmax(dim=-1)
        vq_mel =torch.empty_like(encoded_eeg)
        vq_mel = self.embeddings[indices]
        vq_mel = vq_mel.reshape(bs,d_model)
        return vq_mel

    def forward(self,encoded_eeg:torch.Tensor,encoded_mel:torch.Tensor):
        # bs,d_model = encoded_eeg.shape
        flat_encoded_mel = encoded_mel.detach().view(-1, self.embedding_dim)
        distances = self.getDistances(flat_encoded_mel)
        mel_indices = torch.argmin(distances, dim=-1)
        
        flat_encoded_eeg = encoded_eeg.reshape(-1,self.embedding_dim)
        logits = self.linear(flat_encoded_eeg)
        eeg_indices = logits.detach().argmax(dim=-1)
        acc_rate = torch.eq(eeg_indices,mel_indices).float().mean()
        loss = F.cross_entropy(logits,mel_indices)
        return loss,acc_rate
    
class DistanceClassifier(nn.Module):
    def __init__(self,num_embedding:int,embedding_dim:int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer("embeddings",torch.Tensor(num_embedding,embedding_dim))
    def initEmbeddings(self,embeddings):
        self.embeddings = embeddings.detach()
    def getDistances(self,flat:torch.Tensor):
        return torch.addmm( torch.sum(self.embeddings ** 2,dim=1)+torch.sum(flat ** 2,dim=1,keepdim=True),
                            flat,self.embeddings.T,
                            alpha=-2.0,beta=1.0)
    
    def cls(self,encoded_eeg:torch.Tensor):
        bs,d_model = encoded_eeg.shape
        flat_encoded_eeg = encoded_eeg.reshape(-1,self.embedding_dim)
        logits = -self.getDistances(flat_encoded_eeg)
        indices = logits.argmax(dim=-1)
        vq_mel =torch.empty_like(encoded_eeg)
        vq_mel = self.embeddings[indices]
        vq_mel = vq_mel.reshape(bs,d_model)
        return vq_mel

    def forward(self,encoded_eeg:torch.Tensor,encoded_mel:torch.Tensor):
        # bs,d_model = encoded_eeg.shape
        flat_encoded_mel = encoded_mel.detach().view(-1, self.embedding_dim)
        distances = self.getDistances(flat_encoded_mel)
        mel_indices = torch.argmin(distances, dim=-1)
        
        flat_encoded_eeg = encoded_eeg.reshape(-1,self.embedding_dim)
        logits = -self.getDistances(flat_encoded_eeg) 
        eeg_indices = logits.detach().argmax(dim=-1)
        acc_rate = torch.eq(eeg_indices,mel_indices).float().mean()
        loss = F.cross_entropy(logits,mel_indices)
        return loss,acc_rate