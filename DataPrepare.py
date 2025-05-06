import os
import numpy as np
import librosa.util
import librosa.feature
import scipy.signal
import scipy.fftpack
import random

pts = ['sub-%02d'%i for i in range(1,11)]

class EEGAudioDataset():
    def __init__(self,sub,data_path= r'./feat',win_len = 0.025,frameshift = 0.005,eeg_sr=1024,audio_sr = 16000,n_mels=40,pad_mode = 'constant') -> None:
        self.eeg = []
        self.audio = []
        self.melspec = []
        self.eeg_hg = []
        self.word = []
        self.data_path = data_path
        self.win_len = win_len
        self.frameshift = frameshift
        self.eeg_sr = eeg_sr
        self.audio_sr = audio_sr
        self.pt = sub
        self.n_mels = n_mels
        self.pad_mode = pad_mode
        self.loadData()
        self.preprocessData()
    

    def extractHG(self, data, sr, windowLength=0.025, frameshift=0.005,pad_mode = 'constant'):
        hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]
        
        data = scipy.signal.detrend(data,axis=0)
        numWindows = int(np.round(data.shape[0]/(frameshift*sr)))

        sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
        data = scipy.signal.sosfiltfilt(sos,data,axis=0)

        sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
        data = scipy.signal.sosfiltfilt(sos,data,axis=0)

        sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
        data = scipy.signal.sosfiltfilt(sos,data,axis=0)

        data = np.abs(hilbert3(data))
        return data

    def extractMelSpecs(self, audio, sr, windowLength=0.025, frameshift=0.005,n_mels=40,pad_mode = 'constant'):
        # align to hifigan
        spectrogram = librosa.feature.melspectrogram(y=audio,sr=sr,n_fft=int(windowLength*sr),hop_length=int(frameshift*sr),center=False,n_mels=n_mels)
        spectrogram = np.log(spectrogram + 1e-6)
        return spectrogram.T


    def loadData(self):
        pt = self.pt
        self.folder_path = os.path.join(self.data_path,f'{pt}')
        for filename in os.listdir(self.folder_path):
            word_info = np.load(os.path.join(self.folder_path,filename),allow_pickle=True)
            self.word.append(word_info.item()['label'])
            self.eeg.append(word_info.item()['eeg'])
            self.audio.append(word_info.item()['audio'])

    def preprocessData(self):
        for i in range(len(self.eeg)):
            self.eeg_hg.append(self.extractHG(self.eeg[i],self.eeg_sr,self.win_len,self.frameshift,pad_mode=self.pad_mode))
            self.audio[i] = librosa.util.normalize(self.audio[i]/32767) * 0.95
            self.audio[i] = np.pad(self.audio[i].astype('double'),(int(np.floor((self.win_len-self.frameshift)*self.audio_sr/2)), int(np.ceil((self.win_len-self.frameshift)*self.audio_sr/2))),mode='minimum')
            self.melspec.append(self.extractMelSpecs(self.audio[i],self.audio_sr,self.win_len,self.frameshift,self.n_mels,pad_mode=self.pad_mode))
        self.audio_sr = 1/self.frameshift
    
    def prepareData(self,seg_size,fold_num=0,hop_size = 0.01):

        # get train and test list of words
        train_data_list = []
        train_label_list = []
        test_data_list = []
        test_label_list = [] 
        for i in range(len(self.eeg_hg)):
            if int(i/(len(self.eeg_hg)//10)) == fold_num:
                test_data_list.append(self.eeg_hg[i])
                test_label_list.append(self.melspec[i])
            else:
                train_data_list.append(self.eeg_hg[i])
                train_label_list.append(self.melspec[i])

        # print("train data list shape: ", len(train_data_list))
        # print("test data list shape: ", len(test_data_list))

        train_data = []
        train_label = []
        test_data = []
        test_label = []

        self.ExtractSegments(seg_size, hop_size, train_data_list, train_label_list, train_data, train_label)
        
        self.ExtractSegments(seg_size, hop_size, test_data_list, test_label_list, test_data, test_label)

        train_data, test_data = self.NormalizeSegmentLength(train_data, test_data)        
        train_label, test_label = self.NormalizeSegmentLength(train_label, test_label)

        data_mean = np.mean(train_data)
        data_std = np.std(train_data)
        train_data = (train_data - data_mean) / data_std
        test_data = (test_data - data_mean) / data_std
        # train_dataloader, test_dataloader,input_dim,output_dim = make_dataloader(batch_size, train_data, train_label, test_data, test_label)

        print("train data shape: ", train_data.shape)
        print("test data shape: ", test_data.shape)
        print("train label shape: ", train_label.shape)
        print("test label shape: ", test_label.shape)

        return train_data,train_label,test_data,test_label

    def ExtractSegments(self, seg_size, hop_size, data_list, label_list, data, label):
        for w in range(len(data_list)):
            eeg = data_list[w]
            mel = label_list[w]
            num_win = int((mel.shape[0]/self.audio_sr-2*seg_size)/hop_size)
            for i in range(num_win):
                start_mel = i*hop_size + seg_size
                end_mel = start_mel + seg_size
                start_eeg = start_mel - seg_size
                end_eeg = end_mel
                eegSegment = eeg[round(start_eeg*self.eeg_sr):round(end_eeg*self.eeg_sr),:]
                melSegment = mel[round(start_mel*self.audio_sr):round(end_mel*self.audio_sr),:]
                data.append(eegSegment)
                label.append(melSegment)

    def NormalizeSegmentLength(self, train, test):
        min_data_shape = 1e9
        for data in train:
            if data.shape[0] < min_data_shape:
                min_data_shape = data.shape[0]
        for data in test:
            if data.shape[0] < min_data_shape:
                min_data_shape = data.shape[0]
        for i in range(len(train)):
            if train[i].shape[0] > min_data_shape:
                train[i] = train[i][:min_data_shape,:]
        for i in range(len(test)):
            if test[i].shape[0] > min_data_shape:
                test[i] = test[i][:min_data_shape,:]
        train = np.stack(train,axis=0)
        test = np.stack(test,axis=0)
        return train,test

