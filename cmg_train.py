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

from model.CLUB import CLUBSample_group
from model.CPC import Cross_CPC
from model.VQVAE import VQVAEEncoder,VQVAEDecoder,SemanticDecoder
from dataset import EEGAudioDataset

import json
import argparse
import operator

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
    nhead = model_cfg['nhead']
    n_layer = model_cfg['n_layer']
    n_embedding = model_cfg['n_embedding']
    mi_iter = model_cfg['mi_iter']

    data_path = data_cfg['data_path']
    win_len = data_cfg['win_len']
    frame_shift = data_cfg['frame_shift']
    eeg_sr = data_cfg['eeg_sr']
    audio_sr = data_cfg['audio_sr']
    pad_mode = data_cfg['pad_mode']

    tensor_type = torch.cuda.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_sub = 0
    end_sub = 10
    if argu.sub is not None:
        start_sub = argu.sub-1
        end_sub = argu.sub
    
    for pt in pts[start_sub:end_sub]:
        dataset = EEGAudioDataset(pt,data_path=data_path,win_len=win_len,frameshift=frame_shift,eeg_sr=eeg_sr,audio_sr=audio_sr,pad_mode=pad_mode)
        train_data,train_label,test_data,test_label = dataset.prepareData(seg_size=seg_size)
        test_mfcc = utils.toMFCC(test_label[:,-1,:40])
        # test_mel = test_data

        input_dim = test_data.shape[-1]
        output_dim = test_label.shape[-1]


        vqvae_encoder = VQVAEEncoder(mel_dim=output_dim,eeg_dim=input_dim,mel_output_dim=d_model,eeg_output_dim=d_model,n_embedding=n_embedding,embedding_dim=d_model).to(device)
        cpc = Cross_CPC(embedding_dim=d_model,hidden_dim=hidden_dim,context_dim=hidden_dim,num_layers=n_layer,predict_step=pred_size).to(device)
        mel_mi_net = CLUBSample_group(x_dim=d_model,y_dim=d_model,hidden_size=hidden_dim).to(device)
        eeg_mi_net = CLUBSample_group(x_dim=d_model,y_dim=d_model,hidden_size=hidden_dim).to(device)
        vqvae_decoder = VQVAEDecoder(mel_dim=output_dim,eeg_dim=input_dim,mel_output_dim=d_model,eeg_output_dim=d_model,embedding_dim=d_model).to(device)
        mel_vq_decoder = SemanticDecoder(input_dim=d_model,output_dim=output_dim).to(device)

        main_optimizer = torch.optim.Adam(chain(vqvae_encoder.parameters(),cpc.parameters(),vqvae_decoder.parameters()),lr=lr,betas=(b1,b2))
        mel_mi_net_optimizer = torch.optim.Adam(mel_mi_net.parameters(),lr=lr,betas=(b1,b2))
        eeg_mi_net_optimizer = torch.optim.Adam(eeg_mi_net.parameters(),lr=lr,betas=(b1,b2))
        mel_vq_decoder_optimizer = torch.optim.Adam(mel_vq_decoder.parameters(),lr=lr,betas=(b1,b2))
        scheduler = MultiStepLR(main_optimizer,milestones=[10,20,30],gamma=0.5)

        criterion = nn.L1Loss().to(device)
        loss_fn = lambda x,y:(criterion(x, y)+criterion(torch.exp(x),torch.exp(y)))

        start_epoch = 0
        checkpoint = utils.scan_checkpoint(f'{argu.save_model_dir}/{pt}/{model_name}',model_name)

        if checkpoint is not None:
            state_dict = utils.load_checkpoint(checkpoint)
            start_epoch = state_dict['epoch']
            if start_epoch > end_epoch:
                raise Exception(f'Already got a {model_name} model trained by {end_epoch} rather then {start_epoch}')
            # if operator.eq(state_dict.model_cfg,model_cfg) == False:
            #     raise Exception(f'{model_name} model')
            vqvae_encoder.load_state_dict(state_dict['vqvae_encoder'])
            cpc.load_state_dict(state_dict['cpc'])
            mel_mi_net.load_state_dict(state_dict['mel_mi_net'])
            eeg_mi_net.load_state_dict(state_dict['eeg_mi_net'])
            vqvae_decoder.load_state_dict(state_dict['vqvae_decoder'])
            mel_vq_decoder.load_state_dict(state_dict['mel_vq_decoder'])
            main_optimizer.load_state_dict(state_dict['main_optimizer'])
            mel_mi_net_optimizer.load_state_dict(state_dict['mel_mi_net_optimizer'])
            eeg_mi_net_optimizer.load_state_dict(state_dict['eeg_mi_net_optimizer'])
            mel_vq_decoder_optimizer.load_state_dict(state_dict['mel_vq_decoder_optimizer'])
            
        
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
            models = [vqvae_encoder,cpc,mel_mi_net,eeg_mi_net,vqvae_decoder,test_outputs]
            to_train(models)
            aver_main_loss = 0
            aver_decoder_loss = 0
            # aver_mi_mel_loss = 0
            # aver_mi_eeg_loss = 0
            for _, (eeg, mel) in enumerate(train_dataloader):
                main_optimizer.zero_grad()
                mel_vq_decoder_optimizer.zero_grad()
                eeg = eeg.to(device)
                eeg = eeg.type(tensor_type)
                mel = mel.to(device)
                mel = mel.type(tensor_type)

                for _ in range(mi_iter):
                    eeg_mi_net_optimizer.zero_grad()
                    mel_mi_net_optimizer.zero_grad()
                    _,_,mel_encoder_res,eeg_encoder_res,mel_vq,eeg_vq,_,_,_,_ = vqvae_encoder(mel,eeg)
                    mel_encoder_res = mel_encoder_res.detach()
                    eeg_encoder_res = eeg_encoder_res.detach()
                    mel_vq = mel_vq.detach()
                    eeg_vq = eeg_vq.detach()

                    lld_mel_loss = -mel_mi_net.loglikeli(mel_vq,mel_encoder_res)
                    lld_mel_loss.backward()
                    mel_mi_net_optimizer.step()

                    lld_eeg_loss = -eeg_mi_net.loglikeli(eeg_vq,eeg_encoder_res)
                    lld_eeg_loss.backward()
                    eeg_mi_net_optimizer.step()

                mel_semantic_res,eeg_semantic_res,mel_encoder_res,eeg_encoder_res,mel_vq,eeg_vq,mel_embedding_loss,eeg_embedding_loss,cmcm_loss,equal_num =vqvae_encoder(mel,eeg)
                mi_mel_loss = mel_mi_net.mi_est(mel_vq,mel_encoder_res)
                mi_eeg_loss = eeg_mi_net.mi_est(eeg_vq,eeg_encoder_res)
                # aa_accuracy,ee_accuracy,ae_accuracy,ea_accuracy,cpc_loss = cpc(mel_semantic_res,eeg_semantic_res)
                cpc_loss = cpc(mel_semantic_res,eeg_semantic_res)
                mel_recon_loss,eeg_recon_loss = vqvae_decoder(mel,eeg,mel_encoder_res,eeg_encoder_res,mel_vq,eeg_vq)
                
                main_loss = mel_recon_loss + eeg_recon_loss + mel_embedding_loss + eeg_embedding_loss + mi_mel_loss + mi_eeg_loss + cpc_loss + cmcm_loss
                main_loss.backward()
                for model in models:
                    nn.utils.clip_grad_norm_(model.parameters(),clip_grad)
                main_optimizer.step()

                with torch.no_grad():
                    eeg_vq = vqvae_encoder.EEGVQEncoder(eeg)
                test_outputs = mel_vq_decoder(eeg_vq)
                mel_vq_decode_loss = loss_fn(test_outputs,mel)
                mel_vq_decode_loss.backward()
                mel_vq_decoder_optimizer.step()

                aver_main_loss += main_loss
                aver_decoder_loss += mel_vq_decode_loss
                # aver_mi_mel_loss += mi_mel_loss
                # aver_mi_eeg_loss += mi_eeg_loss


            scheduler.step()

            if e %argu.save_interval == 0:
                state_dict = {
                    'vqvae_encoder':vqvae_encoder.state_dict(),
                    'cpc':cpc.state_dict(),
                    'mel_mi_net':mel_mi_net.state_dict(),
                    'eeg_mi_net':eeg_mi_net.state_dict(),
                    'vqvae_decoder':vqvae_decoder.state_dict(),
                    'mel_vq_decoder':mel_vq_decoder.state_dict(),
                    'main_optimizer':main_optimizer.state_dict(),
                    'mel_mi_net_optimizer':mel_mi_net_optimizer.state_dict(),
                    'eeg_mi_net_optimizer':eeg_mi_net_optimizer.state_dict(),
                    'mel_vq_decoder_optimizer':mel_vq_decoder_optimizer.state_dict(),
                    'epoch':e
                }
                save_path = f'{argu.save_model_dir}/{pt}/{model_name}/{model_name}_{end_epoch:06}.pt'
                torch.save(state_dict,save_path)

            # TODO: add audio, figure by epoch to tensorboard
            if e % argu.summary_interval == 0:
                to_eval(models)
                test_vq = vqvae_encoder.EEGVQEncoder(test_data)
                # test_encode_res = vqvae_encoder.eeg_encoder(test_data)
                test_outputs = mel_vq_decoder(test_vq)
                test_loss = loss_fn(test_outputs,test_label).detach().cpu().numpy()
                # test_outputs = torch.clamp(test_outputs,min=np.log(1e-5))
                # test_loss = loss_fn(test_outputs,test_label).detach().cpu().numpy()
                aver_main_loss = aver_main_loss/len(train_dataloader)
                aver_decoder_loss = aver_decoder_loss/len(train_dataloader)

                writer.add_scalar(f'train main loss',aver_main_loss,e)
                writer.add_scalar(f'train decoder loss',aver_decoder_loss,e)
                
                writer.add_scalar(f'test loss',test_loss,e)

                model_mfcc = utils.toMFCC(test_outputs[:,:40].detach().cpu().numpy())
                eu_dis = 0
                for i in range(test_mfcc.shape[0]):
                    eu_dis += np.linalg.norm(model_mfcc[i] - test_mfcc[i])
                mcd = eu_dis/test_mfcc.shape[0]
                writer.add_scalar(f'test mcd',mcd,e)
                if e%argu.graph_interval == 0:
                    if e == 0:
                        writer.add_figure('origin melspec',utils.plot_spectrogram(test_label[:,-1,:40].detach().cpu().numpy(),audio_sr,int(audio_sr*frame_shift),int(audio_sr*win_len)))
                    writer.add_figure('test melspec',utils.plot_spectrogram(test_outputs[:,:40].detach().cpu().numpy(),audio_sr,int(audio_sr*frame_shift),int(audio_sr*win_len)),e)
                    writer.add_figure('test rms',utils.plot_graph(test_label[:,-1,40].detach().cpu().numpy(),test_outputs[:,40].detach().cpu().numpy(),sr=audio_sr,hop_len=int(audio_sr*frame_shift)),e)
        
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
