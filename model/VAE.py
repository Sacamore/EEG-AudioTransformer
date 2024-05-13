import torch
import torch.nn as nn
import torch.nn.functional as F
import model.transformer as transformer

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
    

class VAE(nn.Module):
    def __init__(self, input_dim:int,d_model:int,seg_size:int,nhead:int = 2,n_layer:int=4 ):
        super(VAE, self).__init__()
        self.encoder = TransformerEncoder(input_dim=input_dim,d_model=d_model,nhead=nhead,n_layer=n_layer)
        
        latent_dim = d_model*seg_size
        self.mu = nn.Linear(latent_dim, latent_dim )
        self.logvar = nn.Linear(latent_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        
        return 

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    mse = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld
