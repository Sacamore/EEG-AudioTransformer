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

from model.VQVAE import MelEncoder,VectorQuantizer,MelDecoder,MelLinearDecoder

from dataset import EEGAudioDataset
from torch_dct import dct

import json
import csv

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
    
    model_name = f'{model_name}_{argu.fold_num}'
    
    for pt in pts[start_sub:end_sub]:
        dataset = EEGAudioDataset(pt,data_path=data_path,win_len=win_len,frameshift=frame_shift,eeg_sr=eeg_sr,audio_sr=audio_sr,pad_mode=pad_mode,n_mels=n_mels)
        train_data,train_label,test_data,test_label = dataset.prepareData(seg_size=seg_size,fold_num=argu.fold_num)
        test_mfcc = utils.toMFCC(utils.getFlatMel(test_label))
        # test_mel = test_data

        input_dim = test_data.shape[-1]
        output_dim = test_label.shape[-1]

        mel_encoder = MelEncoder(input_dim=output_dim,d_model=d_model).to(device)
        vector_quantizer = VectorQuantizer(num_embeddings=n_embedding,embedding_dim=embedding_dim).to(device)
        mel_decoder = MelLinearDecoder(output_dim=output_dim,d_model=d_model,seg_size=seg_size).to(device)
        
        optimizer = torch.optim.Adam(chain(mel_encoder.parameters(),vector_quantizer.parameters(),mel_decoder.parameters()),lr=lr,betas=(b1,b2))
        scheduler = MultiStepLR(optimizer,milestones=[10,20,30],gamma=0.5)

        l1loss = nn.SmoothL1Loss().double().to(device)
        loss_fn = lambda x,y:(l1loss(x.double(), y.double())+l1loss(torch.exp(x.double()).double(),torch.exp(y.double()).double())+l1loss(dct(x.double(),norm='ortho').double(),dct(y.double(),norm='ortho').double()))

        start_epoch = 0
        checkpoint = utils.scan_checkpoint(f'{argu.save_model_dir}/{pt}/{model_name}',f'{model_name}')
        if checkpoint is not None:
            state_dict = utils.load_checkpoint(checkpoint)
            start_epoch = state_dict['epoch']
            if start_epoch > end_epoch:
                raise Exception(f'Already got a {model_name} model trained by {end_epoch} rather then {start_epoch}')
            mel_encoder.load_state_dict(state_dict['mel_encoder'])
            vector_quantizer.load_state_dict(state_dict['vector_quantizer'],strict=False)
            mel_decoder.load_state_dict(state_dict['mel_decoder'])
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

        if argu.save_tensorboard:
            writer = SummaryWriter(f'./logs/{pt}/{model_name}')
        if argu.save_logtxt:
            if os.path.exists(f'./logs/{pt}/{model_name}') == False:
                os.mkdir(f'./logs/{pt}/{model_name}')
            log_header = ["epoch", "test_loss/mel", "train_loss/mel", "reset_number","test_mcd/mel"]
            file_exists = os.path.isfile('./logs/{pt}/{model_name}/log.txt')
            log_file = open('./logs/{pt}/{model_name}/log.txt', 'a', newline='')
            csv_writer = csv.writer(log_file)
            if not file_exists:
                csv_writer.writerow(log_header)

        for e in range(start_epoch,end_epoch):
            models = [mel_encoder,vector_quantizer,mel_decoder]
            to_train(models)
            aver_mel_loss = 0
            for _, (eeg, mel) in enumerate(train_dataloader):
                optimizer.zero_grad()
                mel = mel.to(device)
                mel = mel.type(tensor_type) # [B,T,MEL_D]
                encoded_mel = mel_encoder(mel)
                embed_loss,mel_vq = vector_quantizer(encoded_mel)
                
                mel_decoded = mel_decoder(mel_vq)
                mel_loss = loss_fn(mel_decoded,mel) + embed_loss
                mel_loss.backward()
                for model in models:
                    nn.utils.clip_grad_norm_(model.parameters(),clip_grad)
                optimizer.step()
                aver_mel_loss += mel_loss.detach().cpu().numpy()

            scheduler.step()

            if e!=0 and e%argu.save_interval == 0:
                save_path = f'{argu.save_model_dir}/{pt}/{model_name}'
                if os.path.exists(save_path) == False:
                    os.mkdir(save_path)
                state_dict = {
                    'mel_encoder':mel_encoder.state_dict(),
                    'vector_quantizer':vector_quantizer.state_dict(),
                    'mel_decoder':mel_decoder.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch':e
                }
                torch.save(state_dict,os.path.join(save_path,f'{model_name}_{e:06}.pt'))

            # TODO: add audio, figure by epoch to tensorboard
            if e % argu.summary_interval == 0:
                to_eval(models)
                encoded_mel = mel_encoder(test_label)
                _,mel_vq = vector_quantizer(encoded_mel)
                test_mel_outputs = mel_decoder(mel_vq)
                test_mel_loss = loss_fn(test_mel_outputs,test_label).detach().cpu().numpy()
                vector_quantizer.reset_num = 0
                test_mel_mfcc = utils.toMFCC(utils.getFlatMel(test_mel_outputs.detach().cpu().numpy()))
                mel_eu_dis = 0
                for i in range(test_mfcc.shape[0]):
                    mel_eu_dis += np.linalg.norm(test_mel_mfcc[i] - test_mfcc[i])
                mel_mcd = mel_eu_dis/test_mfcc.shape[0]

                if argu.save_tensorboard:
                    writer.add_scalar(f'test_loss/mel',test_mel_loss,e)
                    writer.add_scalar(f'train_loss/mel',aver_mel_loss/len(train_dataloader),e)     
                    writer.add_scalar(f'reset number',vector_quantizer.reset_num,e)
                    writer.add_scalar(f'test_mcd/mel',mel_mcd,e)
                    if e%argu.graph_interval == 0:
                        if e == 0:
                            writer.add_figure('melspec/origin',utils.plot_spectrogram(test_label.detach().cpu().numpy(),audio_sr,int(audio_sr*frame_shift),int(audio_sr*win_len)))
                        writer.add_figure('melspec/mel',utils.plot_spectrogram(test_mel_outputs.detach().cpu().numpy(),audio_sr,int(audio_sr*frame_shift),int(audio_sr*win_len)),e)

                if argu.save_logtxt:
                    log_data = [e, test_mel_loss, aver_mel_loss/len(train_dataloader), vector_quantizer.reset_num,mel_mcd]
                    csv_writer.writerow(log_data)  # 写入数据行

        if argu.save_tensorboard:
            writer.close()
        if argu.save_logtxt:
            log_file.close()

if __name__ == '__main__':
    argu = utils.parseCommand()
    os.environ["CUDA_VISIBLE_DEVICES"] = argu.use_gpu_num
    np.random.seed(seed=argu.seed)
    torch.manual_seed(argu.seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(argu.seed)
    train(argu)
