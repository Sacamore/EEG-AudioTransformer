from model.BaseModel import BaseModelHolder
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE

# ------------------ 残差块 + 通道注意力 ------------------
class ResidualSEBlock2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(channels)
        )
        
        # 通道注意力（Squeeze-and-Excitation）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.conv_layers(x)
        
        # 通道注意力加权
        se_weight = self.se(out).unsqueeze(-1).unsqueeze(-1)
        out = out * se_weight
        
        out += residual # 残差连接
        return out


# ------------------ 梅尔谱图编码器 ------------------
class MelEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,2), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        self.res_blocks = nn.Sequential(
            ResidualSEBlock2D(32),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualSEBlock2D(64),
        )
        
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        # x shape: [batch, 1, time, n_mel]
        x = x.unsqueeze(1)             # [batch, 1, time, n_mel]
        x = self.initial_conv(x)      # [batch, 32, time//1, n_mel//2]
        x = self.res_blocks(x)        # [batch, 64, time//2, n_mel//4]
        return self.final_layers(x)   # [batch, latent_dim]

# ------------------ EEG编码器 ------------------
class EEGEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,1), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        self.res_blocks = nn.Sequential(
            ResidualSEBlock2D(32),
            nn.Conv2d(32, 64, kernel_size=(3,input_dim), stride=(2,1), padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualSEBlock2D(64),
        )
        
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        # x shape: [batch, 1, time, electrode]
        x = x.unsqueeze(1)             # [batch, 1, time, electrode]
        x = self.initial_conv(x)      # [batch, 32, time//2, electrode//1]
        x = self.res_blocks(x)        # [batch, 64, time//4, 1]
        return self.final_layers(x)
    
class TimeEmbedding(nn.Module):
    """时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, t):
        # t: [batch]
        half_dim = self.mlp[0].in_features // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)
    
# ------------------ 扩散UNet ------------------
class CondUNet(nn.Module):
    def __init__(self, input_shape, eeg_latent_dim, mel_latent_dim):
        super().__init__()
        channels = [64, 128, 256, 512]  # 通道数配置
        self.time_embed = TimeEmbedding(channels[0])
        
        # 条件投影层
        self.eeg_proj = nn.Linear(eeg_latent_dim, channels[0])
        self.mel_proj = nn.Linear(mel_latent_dim, channels[0])
        
        # 输入层
        self.input_conv = nn.Conv2d(input_shape[0], channels[0], 3, padding=1)
        
        # 下采样阶段
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualSEBlock2D(channels[i]),
                ResidualSEBlock2D(channels[i]),
                nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1)
            ) for i in range(len(channels)-1)
        ])
        
        # 中间层
        self.mid_block = nn.Sequential(
            ResidualSEBlock2D(channels[-1]),
            ResidualSEBlock2D(channels[-1])
        )
        
        # 上采样阶段
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(channels[-i-1], channels[-i-2], 3, stride=2, padding=1, output_padding=1),
                ResidualSEBlock2D(channels[-i-2]),
                ResidualSEBlock2D(channels[-i-2])
            ) for i in range(len(channels)-1)
        ])
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], input_shape[0], 1)
        )

    def forward(self, x, t, eeg_cond, mel_cond):
        # x: 噪声化的梅尔谱图 [B, C, T, F]
        # t: 时间步 [B]
        # eeg_cond: EEG编码 [B, D]
        # mel_cond: 梅尔编码 [B, D]
        
        # 时间嵌入
        t_emb = self.time_embed(t)  # [B, C]
        
        # 条件融合
        cond = self.eeg_proj(eeg_cond) + self.mel_proj(mel_cond)  # [B, C]
        cond_emb = t_emb + cond  # [B, C]
        
        # 输入处理
        h = self.input_conv(x)  # [B, C, T, F]
        h = h + cond_emb.unsqueeze(-1).unsqueeze(-1)  # 空间广播
        
        # 下采样
        skips = []
        for down in self.down_blocks:
            h = down[:-1](h)  # 前两个残差块
            skips.append(h)
            h = down[-1](h)  # 下采样卷积
        
        # 中间处理
        h = self.mid_block(h)
        
        # 上采样
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up[0](h)  # 转置卷积上采样
            h = torch.cat([h, skip], dim=1)  # 跳跃连接
            h = up[1:](h)  # 残差块
        
        return self.output_conv(h)

# ------------------ 共享解码器 ------------------
class SharedDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape_1,output_shape_2):
        super().__init__()
        self.output_shape = (output_shape_1,output_shape_2)  # (time, n_mel)
        
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.BatchNorm1d(latent_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(latent_dim*2, output_shape_1 * output_shape_2),
            nn.Unflatten(1,self.output_shape))

    def forward(self, z):
        z = self.projection(z)          
        return z

# ------------------ 判别器 ------------------
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.net(z)

class CNNDualModelHolder(BaseModelHolder):
    def __init__(self, latent_dim, dropout,lr_g,lr_d):
        super().__init__()
        self.latent_dim = latent_dim      # 潜在空间维度
        self.dropout = dropout
        self.lr_g = lr_g
        self.lr_d = lr_d

    def buildModel(self):
        # 梅尔谱图编码器（输入：[batch, time, n_mel]）
        self.mel_encoder = MelEncoder(self.mel_channels,self.latent_dim).to(self.device)

        # EEG编码器（输入：[batch, time, electrode]）
        self.eeg_encoder = EEGEncoder(self.eeg_channels,self.latent_dim).to(self.device)

        # 共享解码器（输出：[batch, time, n_mel]）
        self.decoder = SharedDecoder(self.latent_dim,self.mel_timesteps,self.mel_channels).to(self.device)

        # 判别器
        self.discriminator = Discriminator(self.latent_dim).to(self.device)

        # 初始化权重
        for net in [self.mel_encoder, self.eeg_encoder, self.decoder, self.discriminator]:
            net.apply(self._weightsInit)
            
        self.optimizer_G = torch.optim.Adam(
            list(self.mel_encoder.parameters()) + 
            list(self.eeg_encoder.parameters()) + 
            list(self.decoder.parameters()),
            lr=self.lr_g
        )
        
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d
        )

    def makeDataloader(self, batch_size, train_data, train_label, test_data, test_label):
        train_label = torch.from_numpy(train_label).float()
        train_data = torch.from_numpy(train_data).float()
        test_label = torch.from_numpy(test_label).float()
        test_data = torch.from_numpy(test_data).float()

        self.mel_timesteps = train_label.shape[-2]  # 梅尔谱图的时间步数（time）
        self.mel_channels = train_label.shape[-1]  # 梅尔谱图的通道数（n_mel）
        self.eeg_channels = train_data.shape[-1]  # EEG的电极数（electrode）

        # 创建DataLoader
        self.train_dataloader = DataLoader(
            TensorDataset(train_label, train_data), 
            batch_size=batch_size, 
            shuffle=True
        )
        self.test_dataloader = DataLoader(
            TensorDataset(test_label, test_data),
            batch_size=batch_size,
            shuffle=False
        )
        self.real_labels = torch.ones(batch_size, 1).to(self.device)
        self.fake_labels = torch.zeros(batch_size, 1).to(self.device)

    def train(self):
        self._toTrain([self.mel_encoder, self.eeg_encoder, self.decoder, self.discriminator])
        train_loss = {
            'total_loss': 0,
            'recon_mel_loss': 0,
            'recon_eeg_loss': 0,
            'g_loss': 0,
            'd_loss': 0,
            # 'c_loss': 0
        }
        
        for mel_batch, eeg_batch in self.train_dataloader:
            self.optimizer_D.zero_grad()
            self.optimizer_G.zero_grad()
            mel_batch = self._toDevice(mel_batch)
            eeg_batch = self._toDevice(eeg_batch)

            # ------------ 编码 ------------
            z_mel = self.mel_encoder(mel_batch)  # [batch, latent_dim]
            z_eeg = self.eeg_encoder(eeg_batch)  # [batch, latent_dim]

            # ------------ 对抗训练 ------------
            # 判别器损失
            d_loss = self.discriminator_loss(z_mel.detach(), z_eeg.detach())
            d_loss.backward()
            self.optimizer_D.step()
            
            # ------------ 重建损失 ------------
            # 这里似乎也应该计算zeeg重建损失？或者仍只使用zmel重建损失保证zmel的质量？
            recon_mel = self.decoder(z_mel)      # [batch, n_mel, 10]
            recon_mel_loss = nn.MSELoss()(recon_mel, mel_batch)
            recon_eeg = self.decoder(z_eeg)
            recon_eeg_loss = nn.MSELoss()(recon_eeg, mel_batch)
            
            
            # ------------ 对比损失 ------------
            #c_loss = self.contrastive_loss(z_mel, z_eeg)
            
            # ------------ 生成器损失 ------------
            # 这里似乎应该是用zeeg，因为zeeg是生成的，zmel是真实的
            g_loss = nn.BCELoss()(self.discriminator(z_eeg), self.real_labels[:z_eeg.size(0)])

            # ------------ 总损失 ------------
            total_loss = recon_mel_loss +  0.1 * g_loss +0.1 * recon_eeg_loss #+ 0.5 * c_loss
            total_loss.backward()
            self.optimizer_G.step()


            train_loss['total_loss'] += total_loss.item()
            train_loss['recon_mel_loss'] += recon_mel_loss.item()
            train_loss['recon_eeg_loss'] += recon_eeg_loss.item()
            train_loss['g_loss'] += g_loss.item()
            train_loss['d_loss'] += d_loss.item()
            # train_loss['c_loss'] += c_loss.item()

        for k in train_loss.keys():
            train_loss[k] /= len(self.train_dataloader)
        
        return train_loss

    def contrastive_loss(self, z_mel, z_eeg, temperature=0.07):
        similarity = torch.matmul(z_mel, z_eeg.T)/temperature  # [batch, batch]
        labels = torch.arange(z_mel.size(0)).to(self.device)
        c_loss = nn.CrossEntropyLoss()(similarity, labels)
        return c_loss

    def discriminator_loss(self, z_mel, z_eeg):
        d_real_loss = nn.BCELoss()(self.discriminator(z_mel), self.real_labels[:z_mel.size(0)])
        d_fake_loss = nn.BCELoss()(self.discriminator(z_eeg), self.fake_labels[:z_eeg.size(0)])
        d_loss = d_real_loss + d_fake_loss
        return d_loss

    def predict(self):
        self._toEval([self.mel_encoder, self.eeg_encoder, self.decoder])
        output_mel = []
        test_loss = 0

        with torch.no_grad():
            for mel_batch, eeg_batch in self.test_dataloader:
                mel_batch = self._toDevice(mel_batch)
                eeg_batch = self._toDevice(eeg_batch)

                # 用EEG生成梅尔谱图
                z_eeg = self.eeg_encoder(eeg_batch)
                recon_mel = self.decoder(z_eeg) 
                
                # 计算损失并保存结果
                test_loss += nn.MSELoss()(recon_mel, mel_batch).item()
                output_mel.append(recon_mel.cpu().numpy())

        output_mel = np.concatenate(output_mel, axis=0)
        return test_loss / len(self.test_dataloader), output_mel
    
    def loadModel(self, state_dict):
        self.mel_encoder.load_state_dict(state_dict['mel_encoder'])
        self.eeg_encoder.load_state_dict(state_dict['eeg_encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self.discriminator.load_state_dict(state_dict['discriminator'])
        self.optimizer_G.load_state_dict(state_dict['optimizer_G'])
        self.optimizer_D.load_state_dict(state_dict['optimizer_D'])
        
    def saveModel(self, e):
        state_dict = {
            'mel_encoder': self.mel_encoder.state_dict(),
            'eeg_encoder': self.eeg_encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'epoch': e
        }
        return state_dict
    
    def getTextLogHeader(self):
        log_header = super().getTextLogHeader()
        log_header.insert(1, 'total_loss')
        log_header.insert(2, 'recon_mel_loss')
        log_header.insert(3, 'recon_eeg_loss')
        log_header.insert(4, 'g_loss')
        log_header.insert(5, 'd_loss')
        # print(log_header)
        return log_header
    
    def getTSNE(self):
        z_eeg_list = []
        z_mel_list = []
        with torch.no_grad():
            for mel_batch, eeg_batch in self.test_dataloader:
                mel_batch = self._toDevice(mel_batch)
                eeg_batch = self._toDevice(eeg_batch)

                # 用EEG生成梅尔谱图
                z_eeg = self.eeg_encoder(eeg_batch)
                z_mel = self.mel_encoder(mel_batch)
                
                z_eeg_list.append(z_eeg.cpu().numpy())
                z_mel_list.append(z_mel.cpu().numpy())
        
        z_eeg = np.concatenate(z_eeg_list, axis=0)
        z_mel = np.concatenate(z_mel_list, axis=0)
        z_combined = TSNE(n_components=2).fit_transform(np.concatenate([z_eeg, z_mel]))
        x_min,x_max = z_combined.min(0),z_combined.max(0)
        z_combined = (z_combined - x_min) / (x_max - x_min)
        z_combined_label = np.concatenate([np.zeros(z_eeg.shape[0]), np.ones(z_mel.shape[0])])
        return z_combined, z_combined_label