import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
import random
import model.transformer as transformer

random.seed(1234)
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
    def __init__(self, input_dim:int , output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim,input_dim//2),
                                    nn.LeakyReLU(),
                                    nn.Linear(input_dim//2,input_dim//2),
                                    nn.LeakyReLU(),
                                    nn.Linear(input_dim//2,output_dim))
    
    def forward(self,input_vq:torch.Tensor):
        res = self.linear(input_vq)
        return res
    
class FeatEncoder(nn.Module):
    def __init__(self,feat_dim:int,d_model:int) -> None:
        super().__init__()
        self.aff_feat = nn.Linear(feat_dim,d_model)
        self.relu = nn.LeakyReLU()
    def forward(self,x:torch.Tensor):
        return self.relu(self.aff_feat(x))
    
class FeatDecoder(nn.Module):
    def __init__(self, input_dim:int,output_dim:int,vq_dim:int) -> None:
        super().__init__()
        self.aff_vq = nn.Linear(vq_dim,input_dim)
        self.rec = nn.Linear(input_dim*2,output_dim)
        self.relu = nn.LeakyReLU()
    def forward(self,feat:torch.Tensor,vq:torch.Tensor):
        aff_vq = self.relu(self.aff_vq(vq))
        rec_feat = torch.cat([aff_vq,feat],dim=-1)
        rec = self.rec(rec_feat)
        return rec

class CrossVQEmbeddingEMA(nn.Module):
    def __init__(self, n_embedding:int,embedding_dim:int,commitment_cost:float=0.25,decay:float=0.99,epsilon:float=1e-5) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay =decay
        self.epsilon=epsilon

        init_bound = 1/self.embedding_dim
        embedding = torch.Tensor(n_embedding,embedding_dim)

        embedding.uniform_(-init_bound,init_bound)
        self.register_buffer("embedding",embedding)
        self.register_buffer("ema_count",torch.zeros(n_embedding))
        self.register_buffer("ema_weight",self.embedding.clone())
        self.register_buffer("unactivated_count",-torch.ones(n_embedding))

    def getDistances(self,flat:torch.Tensor):
        return torch.addmm( torch.sum(self.embedding ** 2,dim=1)+torch.sum(flat ** 2,dim=1,keepdim=True),
                            flat,self.embedding.T,
                            alpha=-2.0,beta=1.0)

    def vq_embedding(self,semantic:torch.Tensor):
        # B,T,D = semantic.size()
        flat = semantic.detach().reshape(-1,semantic.shape[-1]) #[BxT,D]
        distances = self.getDistances(flat) #[BxT,M]
        indices = torch.argmin(distances.double(),dim=-1) #[BxT,1]
        quantized = F.embedding(indices,self.embedding) #[BxT,D]
        quantized = quantized.view_as(semantic) # [B,T,D]
        quantized = semantic + (quantized - semantic).detach()
        return quantized
    
    def forward(self,mel_semantic:torch.Tensor,eeg_semantic:torch.Tensor):
        M,D = self.embedding.shape
        B,T,D = mel_semantic.shape
        a_flat = mel_semantic.detach().reshape(-1,D) #[BxT,D]
        e_flat = eeg_semantic.detach().reshape(-1,D) #[BxT,D]

        a_dis = self.getDistances(a_flat) # [BxT, M]
        e_dis = self.getDistances(e_flat) # [BxT, M]

        a_dis_grad = self.getDistances(mel_semantic.reshape(-1,D)) # [BxT, M]
        e_dis_grad = self.getDistances(eeg_semantic.reshape(-1,D)) # [BxT, M]

        a_ph = F.softmax(-torch.sqrt(a_dis_grad),dim=1) # [BxT, M]
        a_ph = a_ph.reshape(B,T,M) # [B, T, M]
        a_ph = a_ph.mean(dim=1) # [B, M]

        e_ph = F.softmax(-torch.sqrt(e_dis_grad),dim=1) # [BxT, M]
        e_ph = e_ph.reshape(B,T,M) # [B, T, M]
        e_ph = e_ph.mean(dim=1) # [B, M]

        Scode = a_ph@torch.log(e_ph.T + 1e-10) + e_ph@torch.log(a_ph.T+1e-10) # [B, B]

        MaxScode = torch.max(-Scode)
        EScode = torch.exp(Scode + MaxScode) #[B, B]
        
        EScode_dim1sum = torch.sum(EScode,dim=1) #[B]
        Lcmcm = 0
        for i in range(B):
            Lcmcm -= torch.log(EScode[i,i]/(EScode_dim1sum[i] + self.epsilon))
        Lcmcm /= B

        a_indices = torch.argmin(a_dis,dim=-1) # [BxT, 1]
        a_encodings = F.one_hot(a_indices,M).float() # [BxT, M]
        a_quantized = F.embedding(a_indices,self.embedding) # [BxT, D]
        a_quantized = a_quantized.view_as(mel_semantic) #[B, T, D]

        e_indices = torch.argmin(e_dis,dim=-1)
        e_encodings = F.one_hot(e_indices,M).float()
        e_quantized = F.embedding(e_indices,self.embedding)
        e_quantized = e_quantized.view_as(eeg_semantic) #[B, T, D]

        a_indices_reshape = a_indices.reshape(B,T)
        a_indices_mode = a_indices_reshape.mode(dim=-1,keepdim=False) #[B]

        e_indices_reshape = e_indices.reshape(B,T)
        e_indices_mode = e_indices_reshape.mode(dim=-1,keepdim=False) #[B]

        equal_item = (a_indices_mode.values == e_indices_mode.values)
        equal_num = equal_item.sum()

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1-self.decay) * torch.sum(a_encodings,dim=0) #[M]
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n+M*self.epsilon)*n #[M]
            a_dw = torch.matmul(a_encodings.T,a_flat) #[M, D]
            ae_dw = torch.matmul(a_encodings.T,e_flat) # [M, D]

            self.ema_weight = self.decay * self.ema_weight + (1-self.decay)*(a_dw + ae_dw)*0.5 # [M, D]
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

            self.ema_count = self.decay * self.ema_count + (1-self.decay) * torch.sum(e_encodings,dim=0)  #[M]
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n+M*self.epsilon)*n #[M]
            e_dw = torch.matmul(e_encodings.T,e_flat) # [M, D]
            ea_dw = torch.matmul(e_encodings.T,a_flat) # [M, D]

            self.ema_weight = self.decay * self.ema_weight + (1-self.decay)*(e_dw + ea_dw)*0.5 # [M, D]
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        self.unactivated_count = self.unactivated_count + 1 #[M]
        for index in a_indices:
            self.unactivated_count[index.item()] = 0
        for index in e_indices:
            self.unactivated_count[index.item()] = 0

        activated_indices = []
        unactivated_indices = []
        for i,x in enumerate(self.unactivated_count):
            if x > 300:
                unactivated_indices.append(i)
                self.unactivated_count[i] = 0
            elif x >= 0 and x < 100:
                activated_indices.append(i)

        # activated_quantized = F.embedding(torch.tensor(activated_indices,dtype=torch.long).cuda(),self.embedding) # [?,D]
        for i in unactivated_indices:
            self.embedding[i] = torch.Tensor(self.embedding_dim).uniform_(-1/self.embedding_dim,1/self.embedding_dim).cuda()
            # self.embedding[i] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(self.embedding_dim).uniform_(-1/1024,1/1024).cuda()
        
        cmcm_loss = 0.5 * Lcmcm

        a_e_latent_loss = F.mse_loss(mel_semantic,a_quantized.detach())
        ae_e_latent_loss = F.mse_loss(mel_semantic,e_quantized.detach())
        a_loss = self.commitment_cost*(2.0*a_e_latent_loss +ae_e_latent_loss)

        e_e_latent_loss = F.mse_loss(eeg_semantic,e_quantized.detach())
        ea_e_latent_loss = F.mse_loss(eeg_semantic,a_quantized.detach())
        e_loss = self.commitment_cost*(2.0*e_e_latent_loss +ea_e_latent_loss)
        
        a_quantized = mel_semantic + (a_quantized-mel_semantic).detach()
        e_quantized = eeg_semantic + (e_quantized-eeg_semantic).detach()

        a_avg_probs = torch.mean(a_encodings,dim=0)
        a_perplexity = torch.exp(-torch.sum(a_avg_probs*torch.log(a_avg_probs+1e-10)))

        e_avg_probs = torch.mean(e_encodings,dim=0)
        e_perplexity = torch.exp(-torch.sum(e_avg_probs*torch.log(e_avg_probs+1e-10)))

        return a_quantized,e_quantized,a_loss,e_loss,a_perplexity,e_perplexity,cmcm_loss,equal_num 

class VQVAEEncoder(nn.Module):
    def __init__(self,mel_dim:int,eeg_dim:int,mel_output_dim:int,eeg_output_dim:int,n_embedding:int,embedding_dim:int) -> None:
        super().__init__()
        self.mel_encoder = FeatEncoder(mel_dim,mel_output_dim)
        self.eeg_encoder = FeatEncoder(eeg_dim,eeg_output_dim)
        self.cross_quantizer = CrossVQEmbeddingEMA(n_embedding,embedding_dim)
        self.mel_selfattn = TransformerEncoder(input_dim=mel_dim,d_model=embedding_dim)
        self.eeg_selfattn = TransformerEncoder(input_dim=eeg_dim,d_model=embedding_dim)
    
    def MelVQEncoder(self,mel_feat:torch.Tensor):
        mel_semantic_res = self.mel_selfattn(mel_feat)
        mel_vq = self.cross_quantizer.vq_embedding(mel_semantic_res)
        return mel_vq#,audio_semantic_res
    
    def EEGVQEncoder(self,eeg_feat:torch.Tensor):
        eeg_semantic_res = self.eeg_selfattn(eeg_feat)
        eeg_vq = self.cross_quantizer.vq_embedding(eeg_semantic_res)
        return eeg_vq#,eeg_semantic_res        

    # def vq_forward(self,audio_feat:torch.Tensor,eeg_feat:torch.Tensor):
    #     eeg_vq,eeg_semantic_res = self.EEGVQEncoder(eeg_feat)
    #     audio_vq,audio_semantic_res = self.AudioVQEncoder(audio_feat)
    #     audio_vq_forward_loss = F.mse_loss(audio_semantic_res,audio_vq.detach()) + 0.25*F.mse_loss(audio_semantic_res,eeg_vq.detach())
    #     eeg_vq_forward_loss = F.mse_loss(eeg_semantic_res,eeg_vq.detach()) + 0.25*F.mse_loss(eeg_semantic_res,audio_vq.detach())
    #     return audio_vq_forward_loss,eeg_vq_forward_loss
    
    def forward(self,mel_feat:torch.Tensor,eeg_feat:torch.Tensor):
        eeg_semantic_res = self.eeg_selfattn(eeg_feat)  # [B,T,D]
        mel_semantic_res = self.mel_selfattn(mel_feat)  # [B,T,D]

        eeg_encoder_res = self.eeg_encoder(eeg_feat)    # [B,T,D]
        mel_encoder_res = self.mel_encoder(mel_feat)  # [B,T,D]

        mel_vq,eeg_vq,mel_embedding_loss,eeg_embedding_loss,mel_perplexity,eeg_perplexity,cmcm_loss,equal_num \
            = self.cross_quantizer(mel_semantic_res,eeg_semantic_res)
        
        return  mel_semantic_res,eeg_semantic_res,mel_encoder_res,eeg_encoder_res,\
                mel_vq,eeg_vq,mel_embedding_loss,eeg_embedding_loss,cmcm_loss,equal_num
    
class VQVAEDecoder(nn.Module):
    def __init__(self,mel_dim:int,eeg_dim:int,mel_output_dim:int,eeg_output_dim:int,embedding_dim:int,class_num:int=141) -> None:
        super().__init__()
        self.mel_decoder = FeatDecoder(mel_output_dim,mel_dim,embedding_dim)
        self.eeg_decoder = FeatDecoder(eeg_output_dim,eeg_dim,embedding_dim)
        # self.audio_semantic_decoder = SemanticDecoder(embedding_dim,class_num)
        # self.eeg_semantic_decoder = SemanticDecoder(embedding_dim,class_num)

    def forward(self,mel_feat:torch.Tensor,eeg_feat:torch.Tensor,mel_encoder_res:torch.Tensor,eeg_encoder_res:torch.Tensor,mel_vq:torch.Tensor,eeg_vq:torch.Tensor):
        mel_recon_res = self.mel_decoder(mel_encoder_res,mel_vq)
        eeg_recon_res = self.eeg_decoder(eeg_encoder_res,eeg_vq)
        audio_recon_loss = F.mse_loss(mel_recon_res,mel_feat)
        eeg_recon_loss = F.mse_loss(eeg_recon_res,eeg_feat)
        # audio_class = self.audio_semantic_decoder(audio_vq)
        # eeg_class = self.eeg_semantic_decoder(eeg_vq)

        return audio_recon_loss,eeg_recon_loss#,audio_class,eeg_class
