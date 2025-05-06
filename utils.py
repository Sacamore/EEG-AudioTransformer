import os
import scipy.fftpack
import librosa
import torch
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

def toMFCC(mel_spec):
    # log_mel_spec = np.log(np.clip(mel_spec,0,None)+1e-6)
    return scipy.fftpack.dct(mel_spec,type=2,axis=1,norm='ortho')[:,:13]

def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(file_path, file_name, epoch, obj):
    checkpoint_list = scan_checkpoint(file_path, file_name)
    if(checkpoint_list is not None):
        for cp in checkpoint_list:
            print("Removing old checkpoint: {}".format(cp))
            os.remove(cp)
    file_path = os.path.join(file_path, f'{file_name}_{epoch:06}.pt')
    print("Saving checkpoint to {}".format(file_path))
    torch.save(obj, file_path)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '??????????')
    cp_list = glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)

def overlap_add_segments(segments, hop_size):
    """
    将分段梅尔频谱拼接为完整序列（支持NumPy和PyTorch）
    
    参数：
    segments : 输入分段，形状 [batch, seg_size, n_mel]
    hop_size : 帧移步长（需满足 hop_size <= seg_size）
    
    返回：
    output : 拼接后的序列，形状 [(batch-1)*hop_size + seg_size, n_mel]
    """
    # 输入验证
    assert len(segments.shape) == 3, "输入必须是3维张量[batch, seg_size, n_mel]"
    batch, seg_size, n_mel = segments.shape
    assert hop_size <= seg_size, "帧移不能超过段长度"
    
    # 计算输出长度并初始化容器
    output_length = int( (batch - 1) * hop_size + seg_size)
    output = np.zeros((output_length, n_mel), dtype=segments.dtype)
    
    # 创建叠加计数数组（用于后续归一化）
    overlap_counter = np.zeros(output_length, dtype=np.int32)
    
    # 遍历每个分段
    for i in range(batch):
        # 计算当前段的起始位置
        start = int(i * hop_size)
        end = int(start + seg_size)
        
        # 将当前段叠加到输出
        output[start:end,:] += segments[i,:,:]
        
        # 记录叠加次数
        overlap_counter[start:end] += 1
    
    # 处理重叠区域的归一化（平均叠加）
    overlap_mask = overlap_counter > 1
    output[overlap_mask] = output[overlap_mask] / overlap_counter[overlap_mask, None]
    
    return output

from scipy.stats import pearsonr
def calPCC(flat_x,flat_y):
    pcc = 0
    for d in range(flat_x.shape[1]):
        r,_ = pearsonr(flat_x[:,d],flat_y[:,d])
        pcc = pcc + r
    return pcc/flat_x.shape[1]   

def calMCD(test_mfcc, decoded_test_mfcc):
    diff = test_mfcc - decoded_test_mfcc
    frame_mcds = np.sqrt(np.sum(diff**2, axis=1))
    mcd = np.mean(frame_mcds)
    return mcd
  
def plot_spectrogram(display_spec,sr,hop_len,n_fft):
    fig = plt.figure(figsize=(10,4))
    # print(display_spec.shape,sr,hop_len,n_fft)
    librosa.display.specshow(display_spec.T,sr=sr,hop_length=hop_len,n_fft=n_fft,x_axis='time',y_axis='mel')
    # plt.yticks([0,512,1024,2048,4096,8192])
    plt.colorbar(format='%+2.0f dB')    
    plt.draw()
    plt.close()
    return fig

def plot_TSNE(data,label):
    fig = plt.figure(figsize=(4,4))
    plt.scatter(data[label==0,0],data[label==0,1],c='red')
    plt.scatter(data[label==1,0],data[label==1,1],c='blue')
    plt.draw()
    plt.close()
    return fig

def plot_graph(ori,model,sr,hop_len):
    fig = plt.figure(figsize=(10,4))
    x = np.arange(len(ori))*(hop_len/float(sr))
    plt.plot(x,ori,color='r',label='origin')
    plt.plot(x,model,color = 'b',label='model')
    plt.legend()
    plt.draw()
    plt.close()
    return fig

import argparse

def parseCommand():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',default=2025,type=int)
    
    parser.add_argument('--model_name',default='DefaultModel',type=str)
    
    parser.add_argument('--model_config',default='',type=str)
    parser.add_argument("--data_config",default='common',type=str)
    
    parser.add_argument('--sub',default=None,type=int)
    parser.add_argument('--fold_num',default=0,type=int)
    parser.add_argument('--epoch',default=None,type=int)
    parser.add_argument('--patience',default=20,type=int)
    parser.add_argument('--use_gpu_num',default='0',type=str)
    
    parser.add_argument('--input_data_dir',default='./feat',type=str)
    parser.add_argument('--save_model_dir',default='./res',type=str)
    parser.add_argument('--save_log_dir',default='./logs',type=str)
    
    parser.add_argument('--save_tensorboard',action="store_true")
    parser.add_argument('--save_logtxt',action="store_true")
    parser.add_argument('--save_model',action="store_true")
    parser.add_argument('--summary_interval',default=5,type=int)
    parser.add_argument('--save_interval',default=200,type=int)
    parser.add_argument('--graph_interval',default=50,type=int)

    parser.add_argument('--pretrain_model',default='',type=str)
    
    argu = parser.parse_args()
    print(f"Initializing Training Process: \n \
            model_name: {argu.model_name}\n \
            model_config: {argu.model_config}\n \
            data_config: {argu.data_config}\n \
            use_gpu_num: {argu.use_gpu_num}\n \
            seed: {argu.seed} \n \
            sub: {argu.sub} \n \
            fold: {argu.fold_num}\n \
            pretrain model: {'None' if argu.pretrain_model=='' else argu.pretrain_model}\n \
            save model: {'true' if argu.save_model else 'false'}\n \
            save method: {'tensorboard' if argu.save_tensorboard else ''} {'logtxt' if argu.save_logtxt else ''}\n")
    # print(argu.save_tensorboard)
    # print(argu.save_logtxt)
    
    return argu