import os
import numpy as np
import torch.backends
import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.data import TensorDataset, DataLoader
# import tqdm
from tensorboardX import SummaryWriter

import model.models as models
from dataset import EEGAudioDataset
from torch_dct import dct

import json
import argparse

config_path = r'./config'

pts = ['sub-%02d'%i for i in range(1,11)]

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
    weight_decay = model_cfg['weight_decay']
    d_model = model_cfg['d_model']
    num_embeddings = model_cfg['num_embeddings']
    embedding_dim = model_cfg['embedding_dim']
    nhead = model_cfg['nhead']
    n_layer = model_cfg['n_layer']

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
        test_mfcc = utils.toMFCC(utils.getFlatMel(test_label))
        test_mel = test_data

        input_dim = test_data.shape[-1]
        output_dim = test_label.shape[-1]


        model = models.Model(
            input_dim=input_dim,
            output_dim=output_dim,
            seg_size=seg_size,
            pred_size=pred_size,
            d_model=d_model,
            num_embeddings=num_embeddings,
            embedding_dim = embedding_dim,
            nhead=nhead,
            n_layer=n_layer
        ).to(device)
        l1loss = nn.SmoothL1Loss().double().to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(b1,b2),weight_decay=weight_decay)

        # loss_fn = lambda x,y:0.8*(criterion(x[:,:40], y[:,-1,:40])+criterion(torch.exp(x[:,:40]),torch.exp(y[:,-1,:40])))+0.2*criterion(x[:,40],y[:,-1,40]) #+0.15*criterion(x[:,41],y[:,-1,41])
        loss_fn = lambda x,y:(l1loss(x.double(), y.double())+l1loss(torch.exp(x.double()).double(),torch.exp(y.double()).double())+l1loss(dct(x.double(),norm='ortho').double(),dct(y.double(),norm='ortho').double()))

        start_epoch = 0
        checkpoint = utils.scan_checkpoint(f'{argu.save_model_dir}/{pt}/{model_name}',model_name)

        if checkpoint is not None:
            state_dict = utils.load_checkpoint(checkpoint)
            model.load_state_dict(state_dict=state_dict['model_state_dict'])
            optimizer.load_state_dict(state_dict=state_dict['optimizer_state_dict'])
            start_epoch = state_dict['epoch']

        if start_epoch > end_epoch:
            continue
            # raise Exception(f'Already got a {model_name} model trained by {end_epoch} rather then {start_epoch}')

        train_data = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label[:,-pred_size:,:])
        test_data = torch.from_numpy(test_data).to(device).type(tensor_type)
        test_label = torch.from_numpy(test_label[:,-pred_size:,:]).to(device).type(tensor_type)
        train_dataset = TensorDataset(train_data,train_label)

        train_dataloader = DataLoader(dataset=train_dataset,num_workers=0,batch_size=batch_size,shuffle=True,pin_memory=True)

        writer = SummaryWriter(f'./logs/{pt}/{model_name}')

        for e in range(start_epoch,end_epoch):
            model.train()
            aver_loss= 0
            for _, (data, label) in enumerate(train_dataloader):
                data = data.to(device)
                data = data.type(tensor_type)
                label = label.to(device)
                label = label.type(tensor_type)
                outputs = model(data)
                loss = loss_fn(outputs,label)

                aver_loss+=loss.detach().cpu().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if e!=0 and e%argu.save_interval == 0:
                save_path = f'{argu.save_model_dir}/{pt}/{model_name}'
                if os.path.exists(save_path) == False:
                    os.mkdir(save_path)
                state_dict = {
                    'epoch':e,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                }
                torch.save(state_dict,os.path.join(save_path,f'{model_name}_{e:06}.pt'))
            # TODO: add audio, figure by epoch to tensorboard
            if e % argu.summary_interval == 0:
                model.eval()
                test_outputs = model(test_data)
                test_outputs = torch.clamp(test_outputs,min=np.log(1e-5))
                test_loss = loss_fn(test_outputs,test_label).detach().cpu().numpy()
                aver_loss = aver_loss/len(train_dataloader)
                
                pcc = utils.calPCC(test_outputs,test_label)
                writer.add_scalar(f'test pcc',pcc,e)
                writer.add_scalar(f'loss/train',aver_loss,e)
                writer.add_scalar(f'loss/test',test_loss,e)
                model_mfcc = utils.toMFCC(utils.getFlatMel(test_outputs.detach().cpu().numpy()))
                eu_dis = 0
                for i in range(test_mfcc.shape[0]):
                    eu_dis += np.linalg.norm(model_mfcc[i] - test_mfcc[i])
                mcd = eu_dis/test_mfcc.shape[0]
                writer.add_scalar(f'mcd/test',mcd,e)
                if e%argu.graph_interval == 0:
                    if e == 0:
                        writer.add_figure('melspec/origin',utils.plot_spectrogram(test_label.detach().cpu().numpy(),audio_sr,int(audio_sr*frame_shift),int(audio_sr*win_len)))
                    writer.add_figure('melspec/test',utils.plot_spectrogram(test_outputs.detach().cpu().numpy(),audio_sr,int(audio_sr*frame_shift),int(audio_sr*win_len)),e)

        writer.close()

def parseCommand():
    print('Initializing Training Process..')

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
    parser.add_argument('--pretrain_model',default='mel_vqvae',type=str)
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
