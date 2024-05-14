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

from model.VQVAE import MaskEEGEncoder,MaskMelEncoder,VectorQuantizer,MaskMelDecoder,CLIP

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
    mask_ratio = model_cfg['mask_ratio']
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

        mel_pos_embed = nn.Embedding(seg_size,d_model).to(device)
        eeg_pos_embed = nn.Embedding(seg_size,d_model).to(device)
        mask_embed = nn.Parameter(torch.randn(d_model)).to(device)
        eeg_encoder = MaskEEGEncoder(input_dim=input_dim,d_model=d_model,seg_size=seg_size).to(device)
        clip = CLIP(batch_size).to(device)
        mel_encoder = MaskMelEncoder(input_dim=output_dim,d_model=d_model,seg_size=seg_size).to(device)
        vector_quantizer = VectorQuantizer(num_embeddings=n_embedding,embedding_dim=embedding_dim).to(device)
        mel_decoder = MaskMelDecoder(output_dim=output_dim,d_model=d_model,seg_size=seg_size).to(device)
        
        eeg_optimizer = torch.optim.Adam(chain(eeg_pos_embed.parameters(),eeg_encoder.parameters(),clip.parameters()),lr=lr,betas=(b1,b2))
        optimizer = torch.optim.Adam(chain(mel_pos_embed.parameters(),mel_encoder.parameters(),vector_quantizer.parameters(),mel_decoder.parameters()),lr=lr,betas=(b1,b2))
        scheduler = MultiStepLR(optimizer,milestones=[10,20,30],gamma=0.5)

        # cross_entrophy_loss = nn.CrossEntropyLoss().to(device)
        l1loss = nn.L1Loss().double().to(device)
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
            mel_pos_embed.load_state_dict(state_dict['mel_pos_embed'])
            eeg_pos_embed.load_state_dict(state_dict['eeg_pos_embed'])
            # mask_embed.load_state_dict(state_dict['mask_embed'])
            eeg_encoder.load_state_dict(state_dict['eeg_encoder'])
            clip.load_state_dict(state_dict['clip'])
            mel_encoder.load_state_dict(state_dict['mel_encoder'])
            vector_quantizer.load_state_dict(state_dict['vector_quantizer'])
            mel_decoder.load_state_dict(state_dict['mel_decoder'])
            eeg_optimizer.load_state_dict(state_dict['eeg_optimizer'])
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
            models = [mel_pos_embed,eeg_pos_embed,eeg_encoder,mel_encoder,clip,vector_quantizer,mel_decoder]
            to_train(models)
            aver_mel_loss = 0
            aver_clip_loss = 0
            aver_eeg_loss = 0
            # aver_mi_mel_loss = 0
            # aver_mi_eeg_loss = 0
            for _, (eeg, mel) in enumerate(train_dataloader):
                optimizer.zero_grad()
                eeg_optimizer.zero_grad()
                eeg = eeg.to(device)
                eeg = eeg.type(tensor_type) # [B,T,EEG_D]
                mel = mel.to(device)
                mel = mel.type(tensor_type) # [B,T,MEL_D]

                bs = eeg.shape[0]
                n_masked = int(seg_size*mask_ratio)
                shuffle_indices = torch.rand(bs,seg_size,device=device).argsort()
                masked_ind,unmasked_ind = shuffle_indices[:,:n_masked],shuffle_indices[:,n_masked:]
                batch_ind = torch.arange(bs,device=device).unsqueeze(-1)

                masked_eeg,unmasked_eeg = eeg[batch_ind,masked_ind],eeg[batch_ind,unmasked_ind]
                masked_mel,unmasked_mel = mel[batch_ind,masked_ind],mel[batch_ind,unmasked_ind]

                unmasked_mel_pos_embed = mel_pos_embed(unmasked_ind)
                masked_mel_pos_embed = mel_pos_embed(masked_ind)

                unmasked_eeg_pos_embed = eeg_pos_embed(unmasked_ind)
                masked_eeg_pos_embed = eeg_pos_embed(masked_ind)

                encoded_eeg_token = eeg_encoder(unmasked_eeg,unmasked_eeg_pos_embed)
                encoded_mel_token = mel_encoder(unmasked_mel,unmasked_mel_pos_embed)
                
                eeg_clip_loss,mel_clip_loss = clip(encoded_eeg_token,encoded_mel_token.detach())
                eeg_loss =  eeg_clip_loss + l1loss(encoded_eeg_token,encoded_mel_token.detach())

                embed_loss,mel_vq = vector_quantizer(encoded_mel_token)

                masked_mel_token = mask_embed[None,None,:].repeat(bs,n_masked,1)
                masked_mel_token += masked_mel_pos_embed

                concat_mel_token = torch.cat([masked_mel_token,mel_vq],dim=1)
                dec_mel_input_tokens = torch.empty_like(concat_mel_token, device=device)
                dec_mel_input_tokens[batch_ind, shuffle_indices] = concat_mel_token

                mel_decoded = mel_decoder(dec_mel_input_tokens)
                mel_loss = loss_fn(mel_decoded[batch_ind,masked_ind,:],masked_mel) + embed_loss
                
                mel_loss.backward()
                eeg_loss.backward()
                for model in models:
                    nn.utils.clip_grad_norm_(model.parameters(),clip_grad)
                eeg_optimizer.step()
                optimizer.step()
                aver_mel_loss += mel_loss.detach().cpu().numpy()
                aver_clip_loss += eeg_loss.detach().cpu().numpy()

                with torch.no_grad():
                    vector_quantizer.eval()
                    _,eeg_vq = vector_quantizer(encoded_eeg_token)
                    
                    masked_eeg_token = mask_embed[None,None,:].repeat(bs,n_masked,1)
                    masked_eeg_token += masked_eeg_pos_embed

                    concat_eeg_token = torch.cat([masked_eeg_token,eeg_vq],dim=1)
                    dec_eeg_input_tokens = torch.empty_like(concat_eeg_token, device=device)
                    dec_eeg_input_tokens[batch_ind, shuffle_indices] = concat_eeg_token

                    eeg_decoded = mel_decoder(dec_eeg_input_tokens)
                    vector_quantizer.train()
                    aver_eeg_loss += loss_fn(eeg_decoded,mel)


                # aver_mi_mel_loss += mi_mel_loss
                # aver_mi_eeg_loss += mi_eeg_loss

            scheduler.step()

            if e!=0 and e%argu.save_interval == 0:
                save_path = f'{argu.save_model_dir}/{pt}/{model_name}'
                if os.path.exists(save_path) == False:
                    os.mkdir(save_path)
                state_dict = {
                    'mel_pos_embed':mel_pos_embed.state_dict(),
                    'eeg_pos_embed':eeg_pos_embed.state_dict(),
                    # 'mask_embed':mask_embed.state_dict(),
                    'eeg_encoder':eeg_encoder.state_dict(),
                    'clip':clip.state_dict(),
                    'mel_encoder':mel_encoder.state_dict(),
                    'vector_quantizer':vector_quantizer.state_dict(),
                    'mel_decoder':mel_decoder.state_dict(),
                    'eeg_optimizer':eeg_optimizer.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch':e
                }
                # save_path = f'{argu.save_model_dir}/{pt}/{model_name}/{model_name}_{e:06}.pt'
                torch.save(state_dict,os.path.join(save_path,f'{model_name}_{e:06}.pt'))

            # TODO: add audio, figure by epoch to tensorboard
            if e % argu.summary_interval == 0:
                to_eval(models)
                bs = test_data.shape[0]
                n_masked = int(seg_size*mask_ratio)
                shuffle_indices = torch.rand(bs,seg_size,device=device).argsort()
                masked_ind,unmasked_ind = shuffle_indices[:,:n_masked],shuffle_indices[:,n_masked:]
                batch_ind = torch.arange(bs,device=device).unsqueeze(-1)
                # batch_ind = torch.arange(test_label.shape[0],device=device).unsqueeze(-1)
                
                masked_eeg,unmasked_eeg = test_data[batch_ind,masked_ind],test_data[batch_ind,unmasked_ind]
                masked_mel,unmasked_mel = test_label[batch_ind,masked_ind],test_label[batch_ind,unmasked_ind]

                unmasked_mel_pos_embed = mel_pos_embed(unmasked_ind)
                masked_mel_pos_embed = mel_pos_embed(masked_ind)

                unmasked_eeg_pos_embed = eeg_pos_embed(unmasked_ind)
                masked_eeg_pos_embed = eeg_pos_embed(masked_ind)

                encoded_eeg_token = eeg_encoder(unmasked_eeg,unmasked_eeg_pos_embed)
                encoded_mel_token = mel_encoder(unmasked_mel,unmasked_mel_pos_embed)
                
                # eeg_clip_loss,mel_clip_loss = clip(encoded_eeg_token,encoded_mel_token.detach())
                # eeg_loss =  eeg_clip_loss + l1loss(encoded_eeg_token,encoded_mel_token.detach())

                embed_loss,mel_vq = vector_quantizer(encoded_mel_token)
                _,eeg_vq = vector_quantizer(encoded_eeg_token)

                masked_mel_token = mask_embed[None,None,:].repeat(bs,n_masked,1)
                masked_mel_token += masked_mel_pos_embed

                concat_mel_token = torch.cat([masked_mel_token,mel_vq],dim=1)
                dec_mel_input_tokens = torch.empty_like(concat_mel_token, device=device)
                dec_mel_input_tokens[batch_ind, shuffle_indices] = concat_mel_token

                test_mel_outputs = mel_decoder(dec_mel_input_tokens)
                test_mel_loss = loss_fn(test_mel_outputs,test_label)
                
                masked_eeg_token = mask_embed[None,None,:].repeat(bs,n_masked,1)
                masked_eeg_token += masked_eeg_pos_embed

                concat_eeg_token = torch.cat([masked_eeg_token,eeg_vq],dim=1)
                dec_eeg_input_tokens = torch.empty_like(concat_eeg_token, device=device)
                dec_eeg_input_tokens[batch_ind, shuffle_indices] = concat_eeg_token

                test_eeg_outputs = mel_decoder(dec_eeg_input_tokens)
                test_eeg_loss = loss_fn(test_eeg_outputs,test_label)
                
                writer.add_scalar(f'test_loss/mel',test_mel_loss,e)
                writer.add_scalar(f'test_loss/eeg',test_eeg_loss,e)

                writer.add_scalar(f'train_loss/mel',aver_mel_loss/len(train_dataloader),e)
                writer.add_scalar(f'train_loss/clip',aver_clip_loss/len(train_dataloader),e)
                writer.add_scalar(f'train_loss/eeg',aver_eeg_loss/len(train_dataloader),e)
                

                test_mel_mfcc = utils.toMFCC(test_mel_outputs[:,-1,:40].detach().cpu().numpy())
                test_eeg_mfcc = utils.toMFCC(test_eeg_outputs[:,-1,:].detach().cpu().numpy())
                mel_eu_dis = 0
                eeg_eu_dis = 0
                for i in range(test_mfcc.shape[0]):
                    mel_eu_dis += np.linalg.norm(test_mel_mfcc[i] - test_mfcc[i])
                    eeg_eu_dis += np.linalg.norm(test_eeg_mfcc[i] - test_mfcc[i])
                mel_mcd = mel_eu_dis/test_mfcc.shape[0]
                eeg_mcd = eeg_eu_dis/test_mfcc.shape[0]
                writer.add_scalar(f'test_mcd/mel',mel_mcd,e)
                writer.add_scalar(f'test_mcd/eeg',eeg_mcd,e)
                if e%argu.graph_interval == 0:
                    if e == 0:
                        writer.add_figure('melspec/origin',utils.plot_spectrogram(test_label[:,-1,:].detach().cpu().numpy(),audio_sr,int(audio_sr*frame_shift),int(audio_sr*win_len)))
                    writer.add_figure('melspec/mel',utils.plot_spectrogram(test_mel_outputs[:,-1,:].detach().cpu().numpy(),audio_sr,int(audio_sr*frame_shift),int(audio_sr*win_len)),e)
                    writer.add_figure('melspec/eeg',utils.plot_spectrogram(test_eeg_outputs[:,-1,:].detach().cpu().numpy(),audio_sr,int(audio_sr*frame_shift),int(audio_sr*win_len)),e)
                    
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
