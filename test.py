import os
import imp
import numpy as np

import utils

words_path = r'./feat/words'
pts = ['sub-%02d'%i for i in range(1,11)]

eeg = dict()
audio = dict()
word = dict()
for pt in pts:
    folder_path = os.path.join(words_path,f'{pt}')
    word[pt] = []
    eeg[pt] = []
    audio[pt] = []
    for filename in os.listdir(folder_path):
        word_info = np.load(os.path.join(folder_path,filename),allow_pickle=True)
        word[pt].append(word_info.item()['label'])
        eeg[pt].append(word_info.item()['eeg'])
        audio[pt].append(word_info.item()['audio'])

window_length = 0.025
frameshift = 0.005
eeg_sample_rate = 1024
audio_sameple_rate = 16000

import librosa
import matplotlib.pyplot as plt

spectrogram = librosa.feature.melspectrogram(y=audio['sub-06'][10].astype('float'),sr=16000,n_fft=400,hop_length=80,center=False,n_mels=80)
spectrogram = librosa.power_to_db(spectrogram,ref=np.max).T
plt.figure()
# test = utils.extractMelSpecs(audio['sub-06'][10],audio_sameple_rate,windowLength=window_length,frameshift=frameshift)
librosa.display.specshow(spectrogram.T,sr=16000,hop_length=80,win_length=400,x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')        
plt.title(f'sub-06-10-origin')
plt.show()