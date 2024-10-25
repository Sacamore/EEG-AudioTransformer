import os
import scipy.fftpack
import librosa
import torch
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

def toMFCC(logMel):
    return scipy.fftpack.dct(logMel,type=2,axis=-1,norm='ortho')[:,:13]

def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '??????????')
    cp_list = glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def getFlatMel(spec):
    seg_size = spec.shape[1]
    mel_spec = spec[:,0,:]
    mel_spec = np.pad(spec[:,0,:],((0,seg_size),(0,0)),mode='constant')
    for i in range(1,seg_size):
        mel_spec = mel_spec + np.pad(spec[:,i,:],((i,seg_size-i),(0,0)),mode='constant')
    mel_spec = mel_spec/seg_size
    display_spec = mel_spec[seg_size//2:-seg_size//2,:]
    return display_spec

from scipy.stats import pearsonr
def calPCC(x:torch.Tensor,y:torch.Tensor):
    flat_x = getFlatMel(x.detach().cpu().numpy())
    flat_y = getFlatMel(y.detach().cpu().numpy())
    pcc = 0
    for d in range(flat_x.shape[1]):
        r,_ = pearsonr(flat_x[:,d],flat_y[:,d])
        pcc = pcc + r
    return pcc/flat_x.shape[1]   
  
def plot_spectrogram(spec,sr,hop_len,n_fft):
    fig = plt.figure(figsize=(10,4))
    display_spec = getFlatMel(spec)
    librosa.display.specshow(display_spec.T,sr=sr,hop_length=hop_len,n_fft=n_fft,x_axis='time',y_axis='mel')
    # plt.yticks([0,512,1024,2048,4096,8192])
    plt.colorbar(format='%+2.0f dB')    
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

    parser.add_argument('--config',default='',type=str)
    parser.add_argument('--epoch',default=None,type=int)
    parser.add_argument('--use_gpu_num',default='0',type=str)
    parser.add_argument('--input_data_dir',default='./feat',type=str)
    parser.add_argument('--save_model_dir',default='./res',type=str)
    parser.add_argument('--seed',default=2024,type=int)
    parser.add_argument('--sub',default=None,type=int)
    parser.add_argument('--summary_interval',default=5,type=int)
    parser.add_argument('--save_interval',default=200,type=int)
    parser.add_argument('--graph_interval',default=50,type=int)
    parser.add_argument('--pretrain_model',default='',type=str)
    parser.add_argument('--fold_num',default=0,type=int)
    parser.add_argument('--save_tensorboard',default=True,type=bool)
    parser.add_argument('--save_logtxt',default=False,type=bool)

    argu = parser.parse_args()
    print(f"Initializing Training Process: \n
            config: {argu.config}\n
            use_gpu_num: {argu.use_gpu_num}\n
            seed: {argu.seed} \n
            sub: {argu.sub} \n
            fold: {argu.fold_num}\n
            pretrain model: {'None' if argu.pretrain_model=='' else argu.pretrain_model}\n
            save: {'tensorboard' if argu.save_tensorboard else ''} {'logtxt' if argu.save_logtxt else ''}\n")

    return argu