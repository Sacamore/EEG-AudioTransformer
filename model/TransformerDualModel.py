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
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(channels)
        
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 通道注意力加权
        se_weight = self.se(out).unsqueeze(-1).unsqueeze(-1)
        out = out * se_weight
        
        out += residual  # 残差连接
        out = self.relu(out)
        return out

# ------------------ 位置编码 ------------------
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

# ------------------ 梅尔谱图编码器 ------------------
class MelEncoder(nn.Module):
    def __init__(self,input_steps, input_dim, latent_dim,d_model=128,nhead=2,nlayer=2,hidden_dim=128,dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=hidden_dim,dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,nlayer)

        # self.initial_conv = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,2), padding=(1,1)),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.1)
        # )
        
        # self.res_blocks = nn.Sequential(
        #     ResidualSEBlock2D(32),
        #     nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2),
        #     ResidualSEBlock2D(64),
        # )
        
        # self.final_layers = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(64, latent_dim)
        # )

        self.final_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_steps * d_model, latent_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(latent_dim*2, latent_dim)
        )


    def forward(self, x):
        x = self.embedding(x)  # [batch, time, electrode] -> [batch, time, d_model]
        x = self.positional_encoding(x)  # [batch, time, d_model]

        x = self.transformer_encoder(x)
        # x = x.unsqueeze(1)
        # x = self.initial_conv(x)
        # x = self.res_blocks(x)
        
        return self.final_layers(x)

# ------------------ EEG编码器 ------------------
class EEGEncoder(nn.Module):
    def __init__(self,input_steps, input_dim, latent_dim,d_model=128,nhead=2,nlayer=2,hidden_dim=128,dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=hidden_dim,dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,nlayer)

        # self.initial_conv = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,1), padding=(1,1)),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.1)
        # )
        
        # self.res_blocks = nn.Sequential(
        #     ResidualSEBlock2D(32),
        #     nn.Conv2d(32, 64, kernel_size=(3,d_model), stride=(2,1), padding=(1,0)),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2),
        #     ResidualSEBlock2D(64),
        # )
            
        
        # self.final_layers = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(64, latent_dim)
        # )
        self.final_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_steps * d_model, latent_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(latent_dim*2, latent_dim)
        )

    def forward(self, x):
        x = self.embedding(x)  # [batch, time, electrode] -> [batch, time, d_model]
        x = self.positional_encoding(x)  # [batch, time, d_model]

        x = self.transformer_encoder(x)
        # x = x.unsqueeze(1)
        # x = self.initial_conv(x)
        # x = self.res_blocks(x)
        
        return self.final_layers(x)

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
            nn.Unflatten(1,self.output_shape)
        )

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

class TransformerDualModelHolder(BaseModelHolder):
    def __init__(self, latent_dim, dropout,lr_g,lr_d):
        super().__init__()
        self.latent_dim = latent_dim      # 潜在空间维度
        self.dropout = dropout
        self.lr_g = lr_g
        self.lr_d = lr_d

    def buildModel(self):
        # 梅尔谱图编码器（输入：[batch, time, n_mel]）
        self.mel_encoder = MelEncoder(self.mel_timesteps,self.mel_channels,self.latent_dim).to(self.device)

        # EEG编码器（输入：[batch, time, electrode]）
        self.eeg_encoder = EEGEncoder(self.eeg_timesteps,self.eeg_channels,self.latent_dim).to(self.device)

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
        self.eeg_timesteps = train_data.shape[-2]  # EEG的时间步数（time）
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
            # c_loss = self.contrastive_loss(z_mel, z_eeg)
            
            # ------------ 生成器损失 ------------
            # 这里似乎应该是用zeeg，因为zeeg是生成的，zmel是真实的
            g_loss = nn.BCELoss()(self.discriminator(z_eeg), self.real_labels[:z_eeg.size(0)])

            # ------------ 总损失 ------------
            total_loss = recon_mel_loss +  0.1 * g_loss + 0.1 * recon_eeg_loss # +0.5 * c_loss
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
        log_header.insert(2, 'recon_loss')
        log_header.insert(3, 'g_loss')
        log_header.insert(4, 'd_loss')
        log_header.insert(5, 'c_loss')
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