import os
import numpy as np
import torch.backends
import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import random

from tensorboardX import SummaryWriter

from model.BaseModel import BaseModelHolder
from model.Transformer import TransformerModelHolder
from model.TransformerEncoder import TransformerEncoderModelHolder
from model.CNNDualModel import CNNDualModelHolder
from model.TransformerDualModel import TransformerDualModelHolder
from model.MultiScaleCNNDualModel import MultiScaleCNNDualModelHolder

from DataPrepare import EEGAudioDataset
from GriffinLimConverter import GriffinLimConverter
from pystoi import stoi

import json
import csv
import gc

class Model:
    def __init__(self,argu):
        self.model_config_path = r'./config/model'
        self.data_config_path = r'./config/data'
        self.argu = argu
        self.pts = ['sub-%02d'%i for i in range(1,11)]
        # open config file
        with open(os.path.join(self.model_config_path,f'{argu.model_config}.json'),'r') as f:
            self.model_cfg = json.load(f)
        
        with open(os.path.join(self.data_config_path,f'{argu.data_config}.json'),'r') as f:
            self.data_cfg = json.load(f)

        # load config 
        self.model_name = argu.model_name
        self.start_epoch = 0
        self.end_epoch = self.model_cfg['epochs'] if argu.epoch is None else argu.epoch

        self.start_sub = 0
        self.end_sub = 10
        if argu.sub is not None:
            self.start_sub = argu.sub-1
            self.end_sub = argu.sub
            
        self.best_loss = float('inf')
        self.patience = argu.patience
        self.patience_counter = 0
        
        self.griffinLimConverter = GriffinLimConverter(
            sample_rate=self.data_cfg['audio_sr'],
            n_fft=int(self.data_cfg['win_len']*self.data_cfg['audio_sr']),
            n_mels=self.data_cfg['n_mels'],
            hop_length=int(self.data_cfg['frame_shift']*self.data_cfg['audio_sr']),
            win_length=int(self.data_cfg['win_len']*self.data_cfg['audio_sr']),
            griffin_lim_iters=100
        )

#region Early Stop
    
    def checkPatience(self, test_loss,pt,fold_num, e):
        if(e <=10):
            return False
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.patience_counter = 0
            self.saveModelCheckpoint(pt,fold_num,e)
        else:
            self.patience_counter += self.argu.summary_interval
            
        return self.patience_counter >= self.patience
                
#endregion

#region Get Model, Optimizer, Loss
    def getModel(self,model:str) -> BaseModelHolder:
        model_cfg = self.model_cfg
        Models = {
            "Transformer":lambda: TransformerModelHolder( 
                d_model=model_cfg['d_model'],nhead=model_cfg['nhead'],
                nlayer=model_cfg['n_layer'],hidden_dim=model_cfg['hidden_dim'],
                dropout=model_cfg['dropout'],lr=model_cfg['lr']
            ),
            "TransformerEncoder":lambda: TransformerEncoderModelHolder( 
                d_model=model_cfg['d_model'],nhead=model_cfg['nhead'],
                nlayer=model_cfg['n_layer'],hidden_dim=model_cfg['hidden_dim'],
                dropout=model_cfg['dropout'],lr=model_cfg['lr']
            ),
            "CNNDual":lambda: CNNDualModelHolder(
                latent_dim=model_cfg['latent_dim'],
                dropout=model_cfg['dropout'],lr_g=model_cfg['lr_g'],lr_d=model_cfg['lr_d']
            ),
            "MultiScaleCNNDual":lambda: MultiScaleCNNDualModelHolder(
                latent_dim=model_cfg['latent_dim'],
                dropout=model_cfg['dropout'],lr_g=model_cfg['lr_g'],lr_d=model_cfg['lr_d']
            ),
            "TransformerDual":lambda: TransformerDualModelHolder(
                latent_dim=model_cfg['latent_dim'],
                dropout=model_cfg['dropout'],lr_g=model_cfg['lr_g'],lr_d=model_cfg['lr_d']
            )
        }
        
        return Models.get(model)

#endregion

    def train(self):
        argu = self.argu
        # model_name = f'{self.model_name}_{argu.fold_num}'
        data_cfg = self.data_cfg
        model_cfg = self.model_cfg
        fold_num = argu.fold_num

        for pt in self.pts[self.start_sub:self.end_sub]:
            self.dataset = EEGAudioDataset(pt,data_path=data_cfg['data_path'],
                                    win_len=data_cfg['win_len'],frameshift=data_cfg['frame_shift'],
                                    eeg_sr=data_cfg['eeg_sr'],audio_sr=data_cfg['audio_sr'],
                                    pad_mode=data_cfg['pad_mode'],n_mels=data_cfg['n_mels'])
            train_data,train_label,test_data,test_label = self.dataset.prepareData(seg_size=model_cfg['seg_size'],fold_num=fold_num,hop_size=model_cfg['hop_size'])

            self.model_holder = self.getModel(model_cfg['model'])()
            self.model_holder.makeDataloader(model_cfg['batch_size'],train_data,train_label,test_data,test_label)
            self.model_holder.buildModel()

            self.test_mel = utils.overlap_add_segments(test_label,model_cfg['hop_size']/data_cfg['frame_shift'])
            self.test_audio = self.griffinLimConverter.mel_to_audio(self.test_mel)
            test_mfcc = utils.toMFCC(self.test_mel)


            self.loadModelCheckpoint(pt,fold_num)

            self.initializeLogging(pt,fold_num)

            for e in range(self.start_epoch, self.end_epoch):
                train_loss = self.model_holder.train()
                gc.collect()
                torch.cuda.empty_cache()

                if e != 0 and e % argu.save_interval == 0:
                    if(utils.scan_checkpoint(f'{argu.save_model_dir}/{pt}/{self.model_name}/{fold_num}',f'{self.model_name}_{fold_num}') is None):
                        self.saveModelCheckpoint(pt,fold_num,e)

                if e % argu.summary_interval == 0:
                    test_loss,output_mel = self.model_holder.predict()
                    flat_mel = utils.overlap_add_segments(output_mel,model_cfg['hop_size']/data_cfg['frame_shift'])
                    mse = np.mean((flat_mel-self.test_mel)**2)
                    pcc = utils.calPCC(flat_mel, self.test_mel)
                    decoded_test_mfcc = utils.toMFCC(flat_mel)
                    mcd = utils.calMCD(test_mfcc, decoded_test_mfcc)
                    decoded_audio = self.griffinLimConverter.mel_to_audio(flat_mel)
                    stoi_score = stoi(decoded_audio, self.test_audio, self.data_cfg['audio_sr'])
                    self.updateLogging(e, train_loss, mse, pcc, mcd, flat_mel, decoded_audio ,stoi_score)
                    
                    # Early stopping check
                    if self.checkPatience(test_loss,pt,fold_num,e):
                        print(f"Early stopping at epoch {e} with best loss {self.best_loss}")
                        break

                    gc.collect()
                    torch.cuda.empty_cache()
                            
            self.finalizeLogging()

# region ModelCheckpoint

    def loadModelCheckpoint(self, pt, fold_num):
        checkpoint = utils.scan_checkpoint(f'{self.argu.save_model_dir}/{pt}/{self.model_name}/{fold_num}',f'{self.model_name}_{fold_num}')
        if checkpoint is not None:
            state_dict = utils.load_checkpoint(checkpoint[-1])
            self.start_epoch = state_dict['epoch']
            if self.start_epoch > self.end_epoch:
                raise Exception(f'Already got a {self.model_name} model trained by {self.end_epoch} rather then {self.start_epoch}')
            self.model_holder.loadModel(state_dict)

    def saveModelCheckpoint(self, pt, fold_num, e):
        if self.argu.save_model is False:
            return
        save_path = f'{self.argu.save_model_dir}/{pt}/{self.model_name}/{fold_num}'
        if(os.path.exists(f'{self.argu.save_model_dir}/{pt}/{self.model_name}') == False):
            os.mkdir(f'{self.argu.save_model_dir}/{pt}/{self.model_name}')
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)
        state_dict = self.model_holder.saveModel(e)
        utils.save_checkpoint(save_path,f'{self.model_name}_{fold_num}', e, state_dict)

# endregion

# region Logging

    def initializeLogging(self, pt, fold_num):
        if os.path.exists(f'{self.argu.save_log_dir}/{pt}/{self.model_name}') == False:
            os.mkdir(f'{self.argu.save_log_dir}/{pt}/{self.model_name}')
        if self.argu.save_tensorboard:
            self.tb_writer = SummaryWriter(f'{self.argu.save_log_dir}/{pt}/{self.model_name}/{fold_num}')
        if self.argu.save_logtxt:
            self.log_file, self.csv_writer = self.initializeTxtlogging(pt, fold_num)

    def updateLogging(self, e, train_loss, test_loss, pcc, mcd,flat_mel,audio, stoi_score):
        if self.argu.save_tensorboard:
            self.updateTensorboardLogging(e, train_loss, test_loss, pcc, mcd, flat_mel,audio, stoi_score)
        if self.argu.save_logtxt:
            write_row = [e]
            for k in train_loss.keys():
                write_row.append(train_loss[k])
            write_row.append(test_loss)
            write_row.append(pcc)
            write_row.append(mcd)
            self.csv_writer.writerow(write_row)

    def finalizeLogging(self):
        if self.argu.save_tensorboard:
            self.tb_writer.close()
        if self.argu.save_logtxt:
            self.log_file.close()

    def updateTensorboardLogging(self, e:int, train_loss:dict, test_loss:float, pcc:float, mcd:float, flat_mel,audio, stoi_score:float):
    
        if np.any(np.isnan(flat_mel)) or np.any(np.isinf(flat_mel)):
            print("flat_mel contains NaN or Inf values.")
            print(f'epoch: {e}, train_loss: {train_loss}, test_loss: {test_loss}, pcc: {pcc}, mcd: {mcd}')
            return
        
        for k in train_loss.keys():
            self.tb_writer.add_scalar(f'train/{k}',train_loss[k],e)
        self.tb_writer.add_scalar(f'test/loss',test_loss,e)
        self.tb_writer.add_scalar(f'test/pcc',pcc,e)
        self.tb_writer.add_scalar(f'test/mcd',mcd,e)
        self.tb_writer.add_figure('melspec',utils.plot_spectrogram(flat_mel,self.data_cfg['audio_sr'],int(self.data_cfg['audio_sr']*self.data_cfg['frame_shift']),int(self.data_cfg['audio_sr']*self.data_cfg['win_len'])),e)
        # print(flat_mel.shape)
        # self.tb_writer.add_figure('melspec',utils.plot_spectrogram(flat_mel,int(1/self.data_cfg['frame_shift']),int(self.model_cfg['hop_size']/self.data_cfg['frame_shift']),int(self.model_cfg['seg_size']/self.data_cfg['frame_shift'])),e)
        self.tb_writer.add_audio('audio',audio, e, sample_rate=self.data_cfg['audio_sr'])
        self.tb_writer.add_scalar(f'test/stoi',stoi_score,e)
        if e%self.argu.graph_interval == 0:
            tsne_data,tsne_label = self.model_holder.getTSNE()
            if(tsne_data is not None):
                self.tb_writer.add_figure('TSNE',utils.plot_TSNE(tsne_data,tsne_label),e)
        if e == 0:
            self.tb_writer.add_figure('origin melspec',utils.plot_spectrogram(self.test_mel,self.data_cfg['audio_sr'],int(self.data_cfg['audio_sr']*self.data_cfg['frame_shift']),int(self.data_cfg['audio_sr']*self.data_cfg['win_len'])))
            self.tb_writer.add_audio('origin audio',self.test_audio,None,sample_rate=self.data_cfg['audio_sr'])
    def initializeTxtlogging(self, pt, fold_num):
        txtLogPath = f'{self.argu.save_log_dir}/{pt}/{self.model_name}/{fold_num}'
        if os.path.exists(txtLogPath) == False:
            os.mkdir(txtLogPath)
        log_header = self.model_holder.getTextLogHeader()
        # print(log_header)
        log_file = open(f'{txtLogPath}/logging.txt', 'w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(log_header)
        return log_file,csv_writer

#endregion

def setDeterministic(argu):
    random.seed(argu.seed)
    np.random.seed(seed=argu.seed)
    torch.manual_seed(argu.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(argu.seed)
    torch.cuda.manual_seed_all(argu.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8'

if __name__ == '__main__':
    argu = utils.parseCommand()
    os.environ["CUDA_VISIBLE_DEVICES"] = argu.use_gpu_num
    # setDeterministic(argu)
    model = Model(argu)
    model.train()
