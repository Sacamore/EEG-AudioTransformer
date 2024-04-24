import os
import scipy.fftpack
import librosa
import torch
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

def toMFCC(logMel):
    return scipy.fftpack.dct(logMel,type=2,axis=1,norm='ortho')[:,:13]

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

def plot_spectrogram(spec,sr,hop_len,n_fft):
    fig = plt.figure(figsize=(10,4))
    librosa.display.specshow(spec.T,sr=sr,hop_length=hop_len,n_fft=n_fft,x_axis='time',y_axis='mel')
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