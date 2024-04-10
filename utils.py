import os

import pandas as pd
import numpy as np 
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack

from pynwb import NWBHDF5IO
import MelFilterBank as mel
from collections import OrderedDict 
import tgt
import librosa

#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractHG(data, sr, windowLength=0.05, frameshift=0.01):
    """
    Window data and extract frequency-band envelope #not using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    #Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate second harmonic of line noise
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Create feature space
    data = np.abs(hilbert3(data))
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat

def stackFeatures(features, modelOrder=4, stepSize=5):
    """
    Add temporal context to each window by stacking neighboring feature vectors
    
    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    Returns
    ----------
    featStacked: array (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    featStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() #Add 'F' if stacked the same as matlab
    return featStacked

def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    """
    Downsamples non-numerical data by using the mode
    
    Parameters
    ----------
    labels: array of str
        Label time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels: array of str
        Downsampled labels
    """
    numWindows=int(np.floor((labels.shape[0]-windowLength*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        unique_label,counts = np.unique(labels[start:stop],return_counts=True)
        newLabels[w] = unique_label[np.argmax(counts)].encode("ascii", errors="ignore").decode()
    return newLabels

def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01):
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    Returns
    ----------
    spectrogram: array (numWindows, numFilter)
        Logarithmic mel scaled spectrogram
    """
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    win = np.hanning(np.floor(windowLength*sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        start_audio = int(np.floor((w*frameshift)*sr))
        stop_audio = int(np.floor(start_audio+windowLength*sr))
        a = audio[start_audio:stop_audio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], 80, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram
    # spectrogram = librosa.feature.melspectrogram(y=audio.astype('float'),sr=sr,n_fft=int(np.floor(windowLength*sr)),hop_length=int(np.floor(frameshift*sr)),center=False,n_mels=23)
    # spectrogram = librosa.power_to_db(spectrogram,ref=np.max)
    # return spectrogram.T


def nameVector(elecs, modelOrder=4):
    """
    Creates list of electrode names
    
    Parameters
    ----------
    elecs: array of str
        Original electrode names
    modelOrder: int
        Temporal context stacked prior and after current window
        Will be added as T-modelOrder, T-(modelOrder+1), ...,  T0, ..., T+modelOrder
        to the elctrode names
    Returns
    ----------
    names: array of str
        List of electrodes including contexts, will have size elecs.shape[0]*(2*modelOrder+1)
    """
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))),1,2 * modelOrder +1).T
    for i, off in enumerate(range(-modelOrder,modelOrder+1)):
        names[i,:] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  #Add 'F' if stacked the same as matlab

def dict4wav(words):
    number_mapping = {
        '0': 'nul',
        '1': 'een',
        '2': 'twee',
        '3': 'drie',
        '4': 'vier',
        '5': 'vijf',
        '6': 'zes',
        '7': 'zeven',
        '8': 'acht',
        '9': 'negen',
        '10': 'tien',
        '11': 'elf',
        '12': 'twaalf',
        '13': 'dertien',
        '14': 'veertien',
        '15': 'vijftien',
        '16': 'zestien',
        '17': 'zeventien',
        '18': 'achttien',
        '19': 'negentien',
        '20': 'twintig',
    }
    words = list(OrderedDict.fromkeys(words))
    new_words = []
    for word in words:
        if word == '':
            continue
        if '`' in word:
            word = word.replace('`','\'')
        if word in number_mapping.keys():
            word = number_mapping[word]
        new_words.append(word) 
    return new_words

def readTextGridPhones(path,sub,num):
    tg = tgt.io.read_textgrid(os.path.join(path,f'{sub}','audio',f'{sub}_{num}.TextGrid'))
    # 获取特定标记层
    target_tier = tg.get_tier_by_name('phones')
    # 获取标记层的间隔和标签
    intervals = target_tier.intervals
    interval_list = []
    for interval in intervals:
        interval_dict = dict()
        interval_dict['label'] = interval.text
        interval_dict['start_time'] = interval.start_time
        interval_dict['end_time'] =interval.end_time
        interval_list.append(interval_dict)
    return interval_list

def readTextGridWords(path,sub,num):
    tg = tgt.io.read_textgrid(os.path.join(path,f'{sub}','audio',f'{sub}_{num}.TextGrid'))
    # 获取特定标记层
    target_tier = tg.get_tier_by_name('words')
    # 获取标记层的间隔和标签
    intervals = target_tier.intervals
    interval_dict = dict()
    for interval in intervals:
        interval_dict['label'] = interval.text
        interval_dict['start_time'] = interval.start_time
        interval_dict['end_time'] =interval.end_time
    return interval_dict

def createPhonesData(path,sub,num,output_path):
    # phones_output_path = os.path.join(path,f'{sub}','phones_data')
    # os.makedirs(phones_output_path, exist_ok=True)
    maxlength = 0
    minlength = 999
    phones_info = readTextGridPhones(path,sub,num)
    audio_sample_rate,audio_data = scipy.io.wavfile.read(os.path.join(path,f'{sub}','audio',f'{sub}_{num}.wav'))
    eeg_sample_rate = 100
    eeg_data = np.load(os.path.join(path,f'{sub}','audio',f'{sub}_{num}.npy'))
    for phone_info in phones_info:
        phone_length = phone_info['end_time']-phone_info['start_time']
        maxlength = max(maxlength,phone_length)
        minlength = min(minlength,phone_length)
        phone_info['audio'] = audio_data[int(phone_info['start_time']*audio_sample_rate):int(phone_info['end_time']*audio_sample_rate)]
        phone_info['eeg'] = eeg_data[int(phone_info['start_time']*eeg_sample_rate):int(phone_info['end_time']*eeg_sample_rate)]
        if not os.path.exists(os.path.join(output_path,phone_info["label"])):
            os.makedirs(os.path.join(output_path,phone_info["label"]))
        count = 1
        save_name = f'{phone_info["label"]}_{sub}_{num}_{count}.npy'
        while os.path.exists(os.path.join(output_path,phone_info["label"],save_name)):
            count += 1
            save_name = f'{phone_info["label"]}_{sub}_{num}_{count}.npy'
        np.save(os.path.join(output_path,phone_info["label"],save_name),phone_info)
    return maxlength,minlength

def toMFCC(logMel):
    return scipy.fftpack.dct(logMel,type=2,axis=1,norm='ortho')[:,:13]
