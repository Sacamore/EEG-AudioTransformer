import os
import numpy as np
import torch.backends
import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from itertools import chain

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
# import tqdm
from tensorboardX import SummaryWriter

from model.VQVAE import EEGEncoder,VectorQuantizer,EEGDecoder

from dataset import EEGAudioDataset
from torch_dct import dct

import json
import argparse

config_path = r'./config'

pts = ['sub-%02d'%i for i in range(1,11)]

def to_eval(models):
    for m in models:
        m.eval()

def to_train(models):
    for m in models:
        m.train()

def train(argu):
    # open config file
    with open(os.path.join(config_path,f'{argu.config}.json'),'r') as f:
        cfg = json.load(f)
        model_cfg = cfg['model_config']
        data_cfg = cfg['data_config']

    # load config 
    model_name = argu.config
    seg_size = model_cfg['seg_size']
    pred_size = model_cfg['pred_size']
    batch_size = model_cfg['batch_size']
    end_epoch = model_cfg['epochs'] if argu.epoch is None else argu.epoch
    lr = model_cfg['lr']
    b1 = model_cfg['b1']
    b2 = model_cfg['b2']
    clip_grad = model_cfg['clip_grad']
    hidden_dim = model_cfg['hidden_dim']
    d_model = model_cfg['d_model']
    embedding_dim = model_cfg['embedding_dim']
    nhead = model_cfg['nhead']
    n_layer = model_cfg['n_layer']
    n_embedding = model_cfg['n_embedding']

    data_path = data_cfg['data_path']
    win_len = data_cfg['win_len']
    frame_shift = data_cfg['frame_shift']
    eeg_sr = data_cfg['eeg_sr']
    audio_sr = data_cfg['audio_sr']
    n_mels = data_cfg['n_mels']
    pad_mode = data_cfg['pad_mode']

    tensor_type = torch.cuda.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_sub = 0
    end_sub = 10
    if argu.sub is not None:
        start_sub = argu.sub-1
        end_sub = argu.sub
    
    for pt in pts[start_sub:end_sub]:
        dataset = EEGAudioDataset(pt,data_path=data_path,win_len=win_len,frameshift=frame_shift,eeg_sr=eeg_sr,audio_sr=audio_sr,pad_mode=pad_mode,n_mels=n_mels)
        train_data,train_label,test_data,test_label = dataset.prepareData(seg_size=seg_size)
        # test_mfcc = utils.toMFCC(utils.getFlatMel(test_label))
        # test_mel = test_data

        input_dim = test_data.shape[-1]
        output_dim = test_label.shape[-1]

        eeg_encoder = EEGEncoder(input_dim=input_dim,d_model=d_model).to(device)
        vector_quantizer = VectorQuantizer(num_embeddings=n_embedding,embedding_dim=embedding_dim).to(device)
        eeg_decoder = EEGDecoder(output_dim=input_dim,d_model=d_model,seg_size=seg_size).to(device)
        
        optimizer = torch.optim.Adam(chain(eeg_encoder.parameters(),vector_quantizer.parameters(),eeg_decoder.parameters()),lr=lr,betas=(b1,b2))
        scheduler = MultiStepLR(optimizer,milestones=[10,20,30],gamma=0.5)

        # TODO:try smooth l1 loss
        l1loss = nn.SmoothL1Loss().double().to(device)
        # loss_fn = lambda x,y:(l1loss(x.double(), y.double())+l1loss(torch.exp(x.double()).double(),torch.exp(y.double()).double())+l1loss(dct(x.double(),norm='ortho').double(),dct(y.double(),norm='ortho').double()))

        start_epoch = 0
        checkpoint = utils.scan_checkpoint(f'{argu.save_model_dir}/{pt}/{model_name}',f'{model_name}')
        if checkpoint is not None:
            state_dict = utils.load_checkpoint(checkpoint)
            start_epoch = state_dict['epoch']
            if start_epoch > end_epoch:
                raise Exception(f'Already got a {model_name} model trained by {end_epoch} rather then {start_epoch}')
            eeg_encoder.load_state_dict(state_dict['eeg_encoder'])
            vector_quantizer.load_state_dict(state_dict['vector_quantizer'])
            eeg_decoder.load_state_dict(state_dict['eeg_decoder'])
            # eeg_optimizer.load_state_dict(state_dict['eeg_optimizer'])
            optimizer.load_state_dict(state_dict['optimizer'])

            
        
        train_data = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label)
        test_data = torch.from_numpy(test_data).to(device).type(tensor_type)
        test_label = torch.from_numpy(test_label).to(device).type(tensor_type)
        train_dataset = TensorDataset(train_data,train_label)
        # test_dataset = TensorDataset(test_data,test_label)

        train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
        # test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)

        writer = SummaryWriter(f'./logs/{pt}/{model_name}')

        for e in range(start_epoch,end_epoch):
            models = [eeg_encoder,vector_quantizer,eeg_decoder]
            to_train(models)
            aver_eeg_loss = 0
            for _, (eeg, mel) in enumerate(train_dataloader):
                optimizer.zero_grad()
                eeg = eeg.to(device)
                eeg = eeg.type(tensor_type) # [B,T,MEL_D]
                encoded_eeg = eeg_encoder(eeg)
                embed_loss,eeg_vq = vector_quantizer(encoded_eeg)
                
                eeg_decoded = eeg_decoder(eeg_vq)
                eeg_loss = l1loss(eeg_decoded,eeg) + embed_loss
                eeg_loss.backward()
                # for model in models:
                #     nn.utils.clip_grad_norm_(model.parameters(),clip_grad)
                optimizer.step()
                aver_eeg_loss += eeg_loss.detach().cpu().numpy()

            scheduler.step()

            if e!=0 and e%argu.save_interval == 0:
                save_path = f'{argu.save_model_dir}/{pt}/{model_name}'
                if os.path.exists(save_path) == False:
                    os.mkdir(save_path)
                state_dict = {
                    # 'eeg_encoder':eeg_encoder.state_dict(),
                    # 'clip':clip.state_dict(),
                    'eeg_encoder':eeg_encoder.state_dict(),
                    'vector_quantizer':vector_quantizer.state_dict(),
                    'eeg_decoder':eeg_decoder.state_dict(),
                    # 'eeg_optimizer':eeg_optimizer.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch':e
                }
                # save_path = f'{argu.save_model_dir}/{pt}/{model_name}/{model_name}_{e:06}.pt'
                torch.save(state_dict,os.path.join(save_path,f'{model_name}_{e:06}.pt'))

            # TODO: add audio, figure by epoch to tensorboard
            if e % argu.summary_interval == 0:
                to_eval(models)
                encoded_eeg = eeg_encoder(test_data)
                _,eeg_vq = vector_quantizer(encoded_eeg)
                test_eeg_outputs = eeg_decoder(eeg_vq)
                test_eeg_loss = l1loss(test_eeg_outputs,test_data).detach().cpu().numpy()
                writer.add_scalar(f'reset number',vector_quantizer.reset_num,e)
                writer.add_scalar(f'loss/test',test_eeg_loss,e)
                writer.add_scalar(f'loss/train',aver_eeg_loss/len(train_dataloader),e)     
                vector_quantizer.reset_num = 0
        writer.close()

def parseCommand():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--config',default='',type=str)
    parser.add_argument('--epoch',default=None,type=int)
    parser.add_argument('--use_gpu_num',default='2',type=str)
    parser.add_argument('--input_data_dir',default='./feat',type=str)
    parser.add_argument('--save_model_dir',default='./res',type=str)
    parser.add_argument('--seed',default=2024,type=int)
    parser.add_argument('--sub',default=None,type=int)
    parser.add_argument('--summary_interval',default=5,type=int)
    parser.add_argument('--save_interval',default=200,type=int)
    parser.add_argument('--graph_interval',default=50,type=int)
    # TODO: add argument to control print interval, summary interval, validate interval

    argu = parser.parse_args()
    return argu

if __name__ == '__main__':
    argu = parseCommand()
    os.environ["CUDA_VISIBLE_DEVICES"] = argu.use_gpu_num
    np.random.seed(seed=argu.seed)
    torch.manual_seed(argu.seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(argu.seed)
    train(argu)
