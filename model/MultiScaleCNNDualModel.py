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

# ------------------ 时空自注意力模块 ------------------
class SpatioTemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [batch, C, H, W]
        batch, C, H, W = x.shape
        Q = self.query(x).view(batch, C, -1).permute(0, 2, 1)  # [batch, H*W, C]
        K = self.key(x).view(batch, C, -1)                     # [batch, C, H*W]
        V = self.value(x).view(batch, C, -1).permute(0, 2, 1)  # [batch, H*W, C]
        
        attn = torch.matmul(Q, K) / (C ** 0.5)
        attn = self.softmax(attn)                               # [batch, H*W, H*W]
        out = torch.matmul(attn, V).permute(0, 2, 1).view(batch, C, H, W)
        return out + x

# ------------------ 多尺度梅尔谱图编码器 ------------------
class MultiScaleMelEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,2), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # 多尺度特征提取路径
        self.res_path1 = nn.Sequential(
            ResidualSEBlock2D(32),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        self.res_path2 = nn.Sequential(
            ResidualSEBlock2D(64),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,2), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # 特征融合层
        self.final_layers = nn.Sequential(
            nn.Linear(32+64+64, latent_dim)  # 三个尺度的特征拼接
        )

    def forward(self, x):
        # 初始尺度特征
        x = x.unsqueeze(1)
        x0 = self.initial_conv(x)
        scale0 = nn.AdaptiveAvgPool2d(1)(x0).flatten(1)
        
        # 中等尺度特征
        x1 = self.res_path1(x0)
        scale1 = nn.AdaptiveAvgPool2d(1)(x1).flatten(1)
        
        # 深层尺度特征
        x2 = self.res_path2(x1)
        scale2 = nn.AdaptiveAvgPool2d(1)(x2).flatten(1)
        
        # 多尺度特征融合
        combined = torch.cat([scale0, scale1, scale2], dim=1)
        return self.final_layers(combined)

# ------------------ 多尺度EEG编码器 ------------------
class MultiScaleEEGEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # 时间维度多尺度处理
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,1), padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # 短时特征路径
        self.short_term = nn.Sequential(
            ResidualSEBlock2D(32),
            nn.Conv2d(32, 64, kernel_size=(3,input_dim), stride=(2,1), padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # 长时特征路径
        self.long_term = nn.Sequential(
            ResidualSEBlock2D(64),
            nn.Conv2d(64, 64, kernel_size=(7,1), stride=(2,1), padding=(3,0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # 特征融合
        self.final_layers = nn.Sequential(
            nn.Linear(32+64+64, latent_dim)  # 初始+短时+长时特征
        )

    def forward(self, x):
        # 基础特征
        x = x.unsqueeze(1)
        x0 = self.initial_conv(x)
        base_feat = nn.AdaptiveAvgPool2d(1)(x0).flatten(1)
        
        # 短时特征
        x1 = self.short_term(x0)
        short_feat = nn.AdaptiveAvgPool2d(1)(x1).flatten(1)
        
        # 长时特征
        x2 = self.long_term(x1)
        long_feat = nn.AdaptiveAvgPool2d(1)(x2).flatten(1)
        
        # 多尺度融合
        combined = torch.cat([base_feat, short_feat, long_feat], dim=1)
        return self.final_layers(combined)
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

class MultiScaleCNNDualModelHolder(BaseModelHolder):
    def __init__(self, latent_dim, dropout,lr_g,lr_d):
        super().__init__()
        self.latent_dim = latent_dim      # 潜在空间维度
        self.dropout = dropout
        self.lr_g = lr_g
        self.lr_d = lr_d

    def buildModel(self):
        # 梅尔谱图编码器（输入：[batch, time, n_mel]）
        self.mel_encoder = MultiScaleMelEncoder(self.mel_channels,self.latent_dim).to(self.device)

        # EEG编码器（输入：[batch, time, electrode]）
        self.eeg_encoder = MultiScaleEEGEncoder(self.eeg_channels,self.latent_dim).to(self.device)

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