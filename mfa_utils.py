import numpy as np 

from collections import OrderedDict 
import tgt
import scipy.io.wavfile
import os.path

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