import os
import numpy as np
import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.data import TensorDataset, DataLoader
# import tqdm
from tensorboardX import SummaryWriter

import model
from dataset import EEGAudioDataset

import json
import argparse

config_path = r'./config'

pts = ['sub-%02d'%i for i in range(1,11)]

def train(argu):
    # open config file
    with open(os.path.join(config_path,f'{argu.config}.json'),'r') as f:
        cfg = json.load(f)['model_config']

    # load config 
    model_name = argu.config
    seg_size = cfg['prv_frame']
    batch_size = cfg['batch_size']
    end_epoch = cfg['epochs'] if argu.epoch is None else argu.epoch
    lr = cfg['lr']
    b1 = cfg['b1']
    b2 = cfg['b2']
    weight_decay = cfg['weight_decay']
    scaled_dim = cfg['scaled_dim']
    d_model = cfg['d_model']
    nhead = cfg['nhead']
    n_layer = cfg['n_layer']

    tensor_type = torch.cuda.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_sub = 0
    end_sub = 10
    if argu.sub is not None:
        start_sub = argu.sub-1
        end_sub = argu.sub
    
    for pt in pts[start_sub:end_sub]:
        dataset = EEGAudioDataset(pt)
        train_data,train_label,test_data,test_label = dataset.prepareData(seg_size=seg_size)

        test_mfcc = utils.toMFCC(test_label)
        test_mel = test_data

        input_dim = test_data.shape[-1]
        output_dim = test_label.shape[-1]


        model = model.Model(
            input_dim=input_dim,
            scaled_dim = scaled_dim,
            seg_size=seg_size,
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            n_layer=n_layer
        ).to(device)
        criterion = nn.L1Loss(reduction='mean').to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(b1,b2),weight_decay=weight_decay)

        start_epoch = 0
        checkpoint = utils.scan_checkpoint(f'{argu.save_model_dir}/{pt}',model_name)

        if checkpoint is not None:
            state_dict = utils.load_checkpoint(checkpoint)
            model.load_state_dict(state_dict=state_dict['model_state_dict'])
            optimizer.load_state_dict(state_dict=state_dict['optimizer_state_dict'])
            start_epoch = state_dict['epoch']

        if start_epoch > end_epoch:
            raise Exception(f'Already got a {model_name} model trained by {end_epoch} rather then {start_epoch}')

        train_data = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label)
        test_data = torch.from_numpy(test_data).to(device).type(tensor_type)
        test_label = torch.from_numpy(test_label).to(device).type(tensor_type)
        train_dataset = TensorDataset(train_data,train_label)

        train_dataloader = DataLoader(dataset=train_dataset,num_workers=0,batch_size=batch_size,shuffle=True,pin_memory=True)

        writer = SummaryWriter(f'./logs/{pt}/{model_name}')

        for e in range(start_epoch,end_epoch):
            # TODO: rewrite validate
            model.train()
            aver_loss= 0
            for _, (data, label) in enumerate(train_dataloader):
                data = data.to(device)
                data = data.type(tensor_type)
                label = label.to(device)
                label = label.type(tensor_type)
                outputs = model(data)
                loss = criterion(outputs, label)
                aver_loss+=loss.detach().cpu().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # TODO: add audio, figure by epoch to tensorboard
            if e % argu.summary_interval == 0:
                model.eval()
                test_outputs = model(test_data)
                test_loss = criterion(test_outputs, test_label).detach().cpu().numpy()
                aver_loss = aver_loss/len(train_dataloader)
                writer.add_scalar(f'train loss',aver_loss,e)
                writer.add_scalar(f'test loss',test_loss,e)
                model_mfcc = utils.toMFCC(test_outputs.detach().cpu().numpy())
                eu_dis = 0
                for i in range(test_mfcc.shape[0]):
                    eu_dis += np.linalg.norm(model_mfcc[i] - test_mfcc[i])
                mcd = eu_dis/test_mfcc.shape[0]
                writer.add_scalar(f'test mcd',mcd,e)

                for name,param in model.named_parameters():
                    writer.add_histogram(name,param.clone().cpu().data.numpy(),e)
                    if param.grad is not None:
                        writer.add_histogram(name+'/grad',param.grad.clone().cpu().data.numpy(),e)

        # test_outputs = model(test_data).detach().cpu().numpy()
        torch.save({
            'epoch':end_epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict()
            }, f'{argu.save_model_dir}/{pt}/{model_name}_{end_epoch:06}.pt')
        
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
    # TODO: add argument to control print interval, summary interval, validate interval

    argu = parser.parse_args()
    return argu

if __name__ == '__main__':
    argu = parseCommand()
    os.environ["CUDA_VISIBLE_DEVICES"] = argu.use_gpu_num
    torch.manual_seed(argu.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(argu.seed)
    train(argu)
