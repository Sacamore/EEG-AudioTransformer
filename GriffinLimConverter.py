import torch
import librosa
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

class GriffinLimConverter:
    def __init__(self, 
                 sample_rate=22050,
                 n_fft=1024,
                 n_mels=80,
                 hop_length=256,
                 win_length=1024,
                 griffin_lim_iters=100):
        """
        参数说明：
        sample_rate: 音频采样率
        n_fft: STFT窗口大小
        n_mels: 梅尔频带数
        hop_length: 帧移
        win_length: 窗口长度
        griffin_lim_iters: Griffin-Lim迭代次数
        """
        self.sr = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop = hop_length
        self.win_length = win_length
        
        # 预计算梅尔滤波器矩阵
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        self.inv_mel_basis = np.linalg.pinv(self.mel_basis)  # 伪逆矩阵
        
        self.griffin_lim_iters = griffin_lim_iters

    def mel_to_audio(self, mel_spec, power=1.2, normalize=True):
        """
        将梅尔频谱转换为音频波形
        参数：
        mel_spec: 梅尔频谱 [n_mels, time] 或 [time, n_mels]
        power: 幅度谱的幂次修正系数（用于补偿梅尔尺度压缩）
        normalize: 是否对输出音频进行峰值归一化
        """
        # 输入格式处理
        if isinstance(mel_spec, torch.Tensor):
            mel_spec = mel_spec.squeeze().cpu().numpy()
        
        if mel_spec.shape[0] == self.n_mels:
            mel_spec = mel_spec.T  # 转换为[time, n_mels]
            
        if np.any(~np.isfinite(mel_spec)):
            raise ValueError("Input mel spectrogram contains invalid values")
    
        mel_spec = np.clip(mel_spec, a_min=0.0, a_max=None)  # 确保非负

        # 梅尔频谱逆变换
        inv_mel = np.dot(self.inv_mel_basis, mel_spec.T)
        inv_mel = np.clip(inv_mel, a_min=1e-8, a_max=None)  # 防止负数
        inv_mel = (inv_mel ** power) * 0.5  # 添加衰减系数

        # 初始化随机相位
        magnitude = np.abs(inv_mel)
        phase_angle = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        complex_spec = magnitude * phase_angle

        # Griffin-Lim相位估计
        for _ in range(self.griffin_lim_iters):
            stft = librosa.istft(complex_spec, hop_length=self.hop, win_length=self.win_length)
            if not np.all(np.isfinite(stft)):
                stft = np.nan_to_num(stft, nan=0.0, posinf=1e-5, neginf=-1e-5)
            est_stft = librosa.stft(stft, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length)
            _, phase = librosa.magphase(est_stft)
            complex_spec = magnitude * phase

        # 最终重建
        audio = librosa.istft(complex_spec, hop_length=self.hop, win_length=self.win_length)
        
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        
        # 后处理
        audio = audio / np.max(np.abs(audio)) * 0.95 if normalize else 1
        return audio

    def visualize(self, mel_spec, audio):
        """可视化结果"""
        fig = plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        librosa.display.specshow(mel_spec.T, 
                                sr=self.sr,
                                hop_length=self.hop,
                                x_axis='time',
                                y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')

        plt.subplot(2, 1, 2)
        plt.plot(np.linspace(0, len(audio)/self.sr, len(audio)), audio)
        plt.xlim(0, len(audio)/self.sr)
        plt.title('Reconstructed Waveform')
        plt.tight_layout()
        plt.close()
        return fig

    def save_wav(self, audio, path):
        """保存音频文件"""
        wavfile.write(path, self.sr, (audio * 32767).astype(np.int16))

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 初始化参数（必须与生成梅尔频谱时的参数一致）
    converter = GriffinLimConverter(
        sample_rate=22050,
        n_fft=1024,
        n_mels=80,
        hop_length=256,
        griffin_lim_iters=100
    )

    # 假设从模型生成的梅尔频谱（示例随机数据）
    # 形状为[time_steps, n_mels]或[n_mels, time_steps]
    dummy_mel = np.random.randn(80, 10)  # 80个梅尔频带，10个时间帧
    
    # 转换为音频
    audio = converter.mel_to_audio(dummy_mel)
    
    # 可视化与保存
    converter.visualize(dummy_mel, audio)
    converter.save_wav(audio, "output.wav")