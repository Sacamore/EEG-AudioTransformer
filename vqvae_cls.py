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

from model.VQVAE import EEGEncoder,EEGConvEncoder,MelEncoder,VectorQuantizer,MelDecoder,MelLinearDecoder,CosSimClassifier,LinearClassifier,DistanceClassifier

from dataset import EEGAudioDataset
from torch_dct import dct
from time import sleep
# from scipy.stats import pearsonr

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

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# def calPCC(x,y):
#     flat_x = utils.getFlatMel(x.detach().cpu().numpy())
#     flat_y = utils.getFlatMel(y.detach().cpu().numpy())
#     pcc = 0
#     for d in range(flat_x.shape[1]):
#         r,_ = pearsonr(flat_x[:,d],flat_y[:,d])
#         pcc = pcc + r
#     return pcc/flat_x.shape[1]   

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
    # mel_d_model = model_cfg['mel_d_model']
    # eeg_d_model = model_cfg['eeg_d_model']
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
        
    if argu.fold_num != 0:
        model_name = f'{model_name}_{argu.fold_num}'

    for pt in pts[start_sub:end_sub]:
        dataset = EEGAudioDataset(pt,data_path=data_path,win_len=win_len,frameshift=frame_shift,eeg_sr=eeg_sr,audio_sr=audio_sr,pad_mode=pad_mode,n_mels=n_mels)
        train_data,train_label,test_data,test_label = dataset.prepareData(seg_size=seg_size,fold_num=argu.fold_num)
        test_mfcc = utils.toMFCC(utils.getFlatMel(test_label))
        test_mel_data = test_label

        input_dim = test_data.shape[-1]
        output_dim = test_label.shape[-1]

        eeg_encoder = EEGConvEncoder(input_dim=input_dim,d_model=d_model,n_layer=n_layer).to(device)
        # eeg_encoder.apply(weights_init)

        mel_encoder = MelEncoder(input_dim=output_dim,d_model=d_model).to(device)
        vector_quantizer = VectorQuantizer(num_embeddings=n_embedding,embedding_dim=embedding_dim).to(device)
        mel_decoder = MelLinearDecoder(output_dim=output_dim,d_model=d_model,seg_size=seg_size).to(device)
        classifier = LinearClassifier(num_embedding=n_embedding,embedding_dim=embedding_dim).to(device)


        eeg_optimizer = torch.optim.Adam(chain(eeg_encoder.parameters(),classifier.parameters()),lr=lr,betas=(b1,b2))
        
        l1loss = nn.SmoothL1Loss().double().to(device)
        loss_fn = lambda x,y:(l1loss(x.double(), y.double())+l1loss(torch.exp(x.double()).double(),torch.exp(y.double()).double())+l1loss(dct(x.double(),norm='ortho').double(),dct(y.double(),norm='ortho').double()))

        start_epoch = 0
        checkpoint = utils.scan_checkpoint(f'{argu.save_model_dir}/{pt}/{model_name}',f'{model_name}')
        if checkpoint is not None:
            state_dict = utils.load_checkpoint(checkpoint)
            start_epoch = state_dict['epoch']
            if start_epoch > end_epoch:
                raise Exception(f'Already got a {model_name} model trained by {end_epoch} rather then {start_epoch}')
            # if operator.eq(state_dict.model_cfg,model_cfg) == False:
            #     raise Exception(f'{model_name} model')
            eeg_encoder.load_state_dict(state_dict['eeg_encoder'])
            mel_encoder.load_state_dict(state_dict['mel_encoder'])
            vector_quantizer.load_state_dict(state_dict['vector_quantizer'])
            mel_decoder.load_state_dict(state_dict['mel_decoder'])
            eeg_optimizer.load_state_dict(state_dict['eeg_optimizer'])
            classifier.load_state_dict(state_dict['classifier'])
        else:
            pretrain_model = utils.scan_checkpoint(f'{argu.save_model_dir}/{pt}/{argu.pretrain_model}',f'{argu.pretrain_model}')
            while pretrain_model is None or pretrain_model!= f'{argu.save_model_dir}/{pt}/{argu.pretrain_model}/{argu.pretrain_model}_001000.pt':
                print(f'waiting:{pretrain_model}')
                sleep(60)
                pretrain_model = utils.scan_checkpoint(f'{argu.save_model_dir}/{pt}/{argu.pretrain_model}',f'{argu.pretrain_model}')
            # if pretrain_model is not None:
            state_dict = utils.load_checkpoint(pretrain_model)
            mel_encoder.load_state_dict(state_dict['mel_encoder'])
            vector_quantizer.load_state_dict(state_dict['vector_quantizer'],strict=False)
            mel_decoder.load_state_dict(state_dict['mel_decoder'])
            # vector_quantizer.logit_scale.grad.zero_()

            classifier.initEmbeddings(vector_quantizer.embeddings)
        
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
            models = [eeg_encoder,classifier]
            to_train(models)
            aver_acc_rate = 0
            aver_cls_loss = 0
            aver_eeg_loss = 0
            for _, (eeg, mel) in enumerate(train_dataloader):
                # optimizer.zero_grad()
                eeg_optimizer.zero_grad()
                eeg = eeg.to(device)
                eeg = eeg.type(tensor_type) # [B,T,EEG_D]
                mel = mel.to(device)
                mel = mel.type(tensor_type) # [B,T,MEL_D]
                encoded_eeg = eeg_encoder(eeg)
                encoded_mel = mel_encoder(mel)
                
                eeg_loss,eeg_acc_rate = classifier(encoded_eeg,encoded_mel)
                eeg_loss.backward()
                for model in models:
                    nn.utils.clip_grad_norm_(model.parameters(),clip_grad)
                eeg_optimizer.step()
                # optimizer.step()
                aver_acc_rate += eeg_acc_rate.detach().cpu().numpy()
                aver_cls_loss += eeg_loss.detach().cpu().numpy()
                with torch.no_grad():
                    eeg_vq = classifier.cls(encoded_eeg)
                    eeg_decoded = mel_decoder(eeg_vq)
                    aver_eeg_loss += loss_fn(eeg_decoded,mel)


            if e!=0 and e%argu.save_interval == 0:
                save_path = f'{argu.save_model_dir}/{pt}/{model_name}'
                if os.path.exists(save_path) == False:
                    os.mkdir(save_path)
                state_dict = {
                    'eeg_encoder':eeg_encoder.state_dict(),
                    # 'clip':clip.state_dict(),
                    'mel_encoder':mel_encoder.state_dict(),
                    'vector_quantizer':vector_quantizer.state_dict(),
                    'mel_decoder':mel_decoder.state_dict(),
                    'classifier':classifier.state_dict(),
                    'eeg_optimizer':eeg_optimizer.state_dict(),
                    # 'optimizer':optimizer.state_dict(),
                    'epoch':e
                }
                # save_path = f'{argu.save_model_dir}/{pt}/{model_name}/{model_name}_{e:06}.pt'
                torch.save(state_dict,os.path.join(save_path,f'{model_name}_{e:06}.pt'))

            # TODO: add audio, figure by epoch to tensorboard
            if e % argu.summary_interval == 0:
                to_eval(models)
                encoded_eeg = eeg_encoder(test_data)
                eeg_vq = classifier.cls(encoded_eeg)
                test_eeg_outputs = mel_decoder(eeg_vq)
                test_eeg_loss = loss_fn(test_eeg_outputs,test_label).detach().cpu().numpy()
            
                pcc = utils.calPCC(test_eeg_outputs,test_label)
                writer.add_scalar(f'test pcc',pcc,e)
                writer.add_scalar(f'test_loss/eeg',test_eeg_loss,e)

                writer.add_scalar(f'train_acc/eeg-vq',aver_acc_rate/len(train_dataloader),e)
                writer.add_scalar(f'train_loss/cls',aver_cls_loss/len(train_dataloader),e)
                writer.add_scalar(f'train_loss/eeg',aver_eeg_loss/len(train_dataloader),e)
                
                test_eeg_mfcc = utils.toMFCC(utils.getFlatMel(test_eeg_outputs.detach().cpu().numpy()))
                eeg_eu_dis = 0
                for i in range(test_mfcc.shape[0]):
                    eeg_eu_dis += np.linalg.norm(test_eeg_mfcc[i] - test_mfcc[i])
                eeg_mcd = eeg_eu_dis/test_mfcc.shape[0]
                writer.add_scalar(f'test_mcd/eeg',eeg_mcd,e)
                if e%argu.graph_interval == 0:
                    writer.add_figure('melspec/eeg',utils.plot_spectrogram(test_eeg_outputs.detach().cpu().numpy(),audio_sr,int(audio_sr*frame_shift),int(audio_sr*win_len)),e)
                    
                    # writer.add_figure('test rms',utils.plot_graph(test_label[:,-1,40].detach().cpu().numpy(),test_outputs[:,-1,40].detach().cpu().numpy(),sr=audio_sr,hop_len=int(audio_sr*frame_shift)),e)
        
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
    parser.add_argument('--fold_num',default=0,type=int)
    # TODO: add argument to control print interval, summary interval, validate interval

    argu = parser.parse_args()
    return argu

if __name__ == '__main__':
    argu = parseCommand()
    os.environ["CUDA_VISIBLE_DEVICES"] = argu.use_gpu_num
    np.random.seed(seed=argu.seed)
    torch.manual_seed(argu.seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(argu.seed)
    train(argu)
