import numpy as np
import torch
import torch.utils
import torch.utils.data as utils_data
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from torch import Tensor

from model.CLUB import CLUBSample_group
from model.CPC import Cross_CPC
from model.VQVAE import VQVAEEncoder,VQVAEDecoder

class model(nn.Module):
    def __init__(self,
                 audio_dim:int,
                 eeg_dim:int,
                 audio_output_dim:int,
                 eeg_output_dim:int,
                 n_embedding:int,
                 embedding_dim:int,
                 hidden_dim:int=256,
                 context_dim=256,
                 n_layer = 2,
                 class_num = 141
                 ) -> None:
        super().__init__()
        self.vqvae_encoder = VQVAEEncoder(audio_dim,eeg_dim,audio_output_dim,eeg_output_dim,n_embedding,embedding_dim)
        self.cpc = Cross_CPC(embedding_dim,hidden_dim,context_dim,n_layer)
        self.audio_mi_net = CLUBSample_group(embedding_dim,audio_output_dim,hidden_dim)
        self.eeg_mi_net = CLUBSample_group(embedding_dim,eeg_output_dim,hidden_dim)
        self.vqvae_decoder = VQVAEDecoder(audio_dim,eeg_dim,audio_output_dim,eeg_output_dim,embedding_dim,class_num)

    