import torch
import torch.nn as nn

# 梅尔谱图编码器（同前）
class MelEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (input_dim//4)**2, latent_dim)
        )
    def forward(self, x):
        return self.conv_layers(x)

# EEG编码器（时序处理）
class EEGEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.Conv1d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        return self.temporal_net(x)

# 共享解码器（重建梅尔谱图）
class SharedDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * (output_dim//4)**2),
            nn.Unflatten(1, (64, output_dim//4, output_dim//4)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.decoder(z)