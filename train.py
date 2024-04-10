import os
import numpy as np
import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.data import TensorDataset, DataLoader
# import tqdm
from torch.utils.tensorboard import SummaryWriter

import transformer
import json
import matplotlib.pyplot as plt
import librosa.display


feat_path = r'./feat'
config_path = r'./config'

pts = ['sub-%02d'%i for i in range(1,11)]
model_name = 'h4l6p3f40e10000'

with open(os.path.join(config_path,f'{model_name}.json'),'r') as f:
    cfg = json.load(f)['model_config']

prv_frame = cfg['prv_frame']
batch_size = cfg['batch_size']
epochs = cfg['epochs']
lr = cfg['lr']
b1 = cfg['b1']
b2 = cfg['b2']
scaled_dim = cfg['scaled_dim']
d_model = cfg['d_model']
nhead = cfg['nhead']
n_layer = cfg['n_layer']

tensor_type = torch.cuda.FloatTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadData(pt):
    return np.load(os.path.join(feat_path,f'{pt}_data.npy')),np.load(os.path.join(feat_path,f'{pt}_label_40.npy'))

def splitData(total_data,total_label):
    # 随机打乱数据索引
    indices = np.random.permutation(total_data.shape[0])

    # 计算划分索引
    split_index = int(total_data.shape[0] * 0.9)

    # 划分数据集
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_data_list = []
    test_data_list = []

    # 获取训练集和测试集数据
    data_padding = np.zeros((1,total_data.shape[1]))

    for idx in train_indices:
        if idx-prv_frame+1<0:
            tmp = total_data[0:idx+1]
            for _ in range(prv_frame-idx-1):
                tmp=np.insert(tmp,0,data_padding,axis=0)
            train_data_list.append(tmp)
        else:
            train_data_list.append(total_data[idx-prv_frame+1:idx+1])
    for idx in test_indices:
        if idx-prv_frame+1<0:
            tmp = total_data[0:idx+1]
            for _ in range(prv_frame-idx-1):
                tmp=np.insert(tmp,0,data_padding,axis=0)
            test_data_list.append(tmp)
        else:
            test_data_list.append(total_data[idx-prv_frame+1:idx+1])
    train_data = total_data[train_indices]
    train_label = total_label[train_indices]
    # test_data = total_data[test_indices]
    test_label = total_label[test_indices]

    train_data_mean = np.mean(train_data)
    train_data_std = np.std(train_data)

    train_data = np.stack(train_data_list,axis=0)
    test_data = np.stack(test_data_list,axis=0)

    train_data = (train_data-train_data_mean)/train_data_std
    test_data = (test_data-train_data_mean)/train_data_std
    return train_data,train_label,test_data,test_label

for pt in pts:
    total_data,total_label = loadData(pt)
    train_data,train_label,test_data,test_label = splitData(total_data,total_label)

    test_mfcc = utils.toMFCC(test_label)
    test_mel = test_data

    input_dim = test_data.shape[-1]
    output_dim = test_label.shape[-1]


    model = transformer.Model(
        input_dim=input_dim,
        scaled_dim = scaled_dim,
        prv_dim=prv_frame,
        output_dim=output_dim,
        d_model=d_model,
        nhead=nhead,
        n_layer=n_layer
    ).to(device)
    criterion = nn.L1Loss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(b1,b2))


    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    test_data = torch.from_numpy(test_data).to(device).type(tensor_type)
    test_label = torch.from_numpy(test_label).to(device).type(tensor_type)
    train_dataset = TensorDataset(train_data,train_label)

    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)

    writer = SummaryWriter(f'./logs/{pt}/{model_name}')


    for e in range(epochs):
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

    test_outputs = model(test_data).detach().cpu().numpy()
    torch.save({
        'epoch':epochs,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
        }, f'./res/{pt}/{model_name}.pt')
    
    writer.close()

