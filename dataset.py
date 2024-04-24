import os
import numpy as np
import librosa
import scipy.signal
import scipy.fftpack
import utils

pts = ['sub-%02d'%i for i in range(1,11)]

class EEGAudioDataset():
    def __init__(self,sub,data_path= r'./feat/words',win_len = 0.025,frameshift = 0.005,eeg_sr=1024,audio_sr = 16000) -> None:
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
        self.sub = sub
        self.pt = sub
        self.loadData()
        self.preprocessData()
    
    @staticmethod
    def extractHG(data, sr, windowLength=0.025, frameshift=0.005):
        hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]
        
        data = scipy.signal.detrend(data,axis=0)
        numWindows = int(np.round(data.shape[0]/(frameshift*sr)))

        sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
        data = scipy.signal.sosfiltfilt(sos,data,axis=0)

        sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
        data = scipy.signal.sosfiltfilt(sos,data,axis=0)

        sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
        data = scipy.signal.sosfiltfilt(sos,data,axis=0)

        hg = np.abs(hilbert3(data))
        feat = np.zeros((numWindows,data.shape[1]*2))
        hg = np.pad(hg,((int(np.floor((windowLength-frameshift)*sr/2)), int(np.ceil((windowLength-frameshift)*sr/2))),(0,0)),mode='reflect')
        data = np.pad(data,((int(np.floor((windowLength-frameshift)*sr/2)), int(np.ceil((windowLength-frameshift)*sr/2))),(0,0)),mode='reflect')
        for win in range(numWindows):
            start= int(np.floor((win*frameshift)*sr))
            stop = int(np.floor(start+windowLength*sr))
            if stop > data.shape[0]:
                stop = data.shape[0]
            # feat[win,:] = np.mean(data[start:stop,:],axis=0)
            feat[win,:] = np.concatenate((np.mean(data[start:stop,:],axis=0),np.mean(hg[start:stop,:],axis=0)),axis=0)
        return feat

    @staticmethod
    def extractMelSpecs(audio, sr, windowLength=0.025, frameshift=0.005):
        # align to hifigan
        audio = librosa.util.normalize(audio/32767) * 0.95
        audio = np.pad(audio.astype('float'),(int(np.floor((windowLength-frameshift)*sr/2)), int(np.floor((windowLength-frameshift)*sr/2))),mode='reflect')
        spectrogram = librosa.feature.melspectrogram(y=audio,sr=sr,n_fft=int(windowLength*sr),hop_length=int(frameshift*sr),center=False,n_mels=40)
        spectrogram = np.log(np.clip(spectrogram,a_min=1e-5,a_max=None))
        rms = librosa.feature.rms(y=audio,frame_length=int(windowLength*sr),hop_length=int(frameshift*sr),center=False)
        zcr = librosa.feature.zero_crossing_rate(y=audio,frame_length=int(windowLength*sr),hop_length=int(frameshift*sr),center=False)
        feat = np.concatenate((spectrogram.T,rms.T,zcr.T),axis=1)
        return feat


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
            self.eeg_hg.append(self.extractHG(self.eeg[i],self.eeg_sr,self.win_len,self.frameshift))
            self.melspec.append(self.extractMelSpecs(self.audio[i],self.audio_sr,self.win_len,self.frameshift))
            if self.eeg_hg[i].shape[0]!=self.melspec[i].shape[0]:
            # print(f'{pt}-{i} not align with audio:{audio[i].shape[0]} and eeg:{eeg[i].shape[0]}')
                minlen = min(self.eeg_hg[i].shape[0],self.melspec[i].shape[0])
                self.eeg_hg[i] = self.eeg_hg[i][:minlen,:]
                self.melspec[i] = self.melspec[i][:minlen,:]
    
    def prepareData(self,seg_size,hop_size = 1,train_test_ratio=0.9):
        split_index = int(len(self.eeg_hg)*train_test_ratio)

        # get train and test list of words
        train_data_list = self.eeg_hg[:split_index]
        train_label_list = self.melspec[:split_index]
        test_data_list = self.eeg_hg[split_index:]
        test_label_list = self.melspec[split_index:]

        train_data = []
        test_data = []
        train_label = []
        test_label = []

        pad_width = ((int(np.floor((seg_size-hop_size)/2.0)),int(np.ceil((seg_size-hop_size)/2.0))),(0,0))

        for w in range(len(train_data_list)):
            word = np.pad(train_data_list[w],pad_width,mode='reflect')
            mel = np.pad(train_label_list[w],pad_width,mode='reflect')
            num_win = int(train_data_list[w].shape[0]/float(hop_size))
            for i in range(num_win):
                start = i*hop_size
                end = start + seg_size
                train_data.append(word[start:end,:])
                train_label.append(mel[start:end,:])
        for w in range(len(test_data_list)):
            word = np.pad(test_data_list[w],pad_width,mode='reflect')
            mel = np.pad(test_label_list[w],pad_width,mode='reflect')
            num_win = int(test_data_list[w].shape[0]/float(hop_size))
            for i in range(num_win):
                start = i*hop_size
                end = start + seg_size
                test_data.append(word[start:end,:])
                test_label.append(mel[start:end,:])
        # for data in train_data:
        #     if data.shape[0] != 16:
        #         print(data.shape)

        train_data = np.stack(train_data,axis=0)
        train_label = np.stack(train_label,axis=0)
        test_data = np.stack(test_data,axis=0)
        test_label = np.stack(test_label,axis=0)

        train_data_mean = np.mean(train_data)
        train_data_std = np.std(train_data)

        train_data = (train_data-train_data_mean)/train_data_std
        test_data = (test_data-train_data_mean)/train_data_std

        return train_data,train_label,test_data,test_label