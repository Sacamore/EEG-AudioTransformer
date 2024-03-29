import os
import numpy as np
import utils

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.data import TensorDataset, DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter

import transformer


feat_path = r'./feat'
pts = ['sub-%02d'%i for i in range(1,11)]

batch_size = 512
epochs = 1000
print_interval = 5
lr = 0.00002
b1 = 0.5
b2 = 0.999
d_model = 256
nhead = 4
n_layer =6
tensor_type = torch.cuda.FloatTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadData(pt):
    return np.load(os.path.join(feat_path,f'{pt}_data.npy')),np.load(os.path.join(feat_path,f'{pt}_label.npy'))

def splitData(total_data,total_label):
    # 随机打乱数据索引
    indices = np.random.permutation(total_data.shape[0])

    # 计算划分索引
    split_index = int(total_data.shape[0] * 0.9)

    # 划分数据集
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # 获取训练集和测试集数据
    train_data = total_data[train_indices]
    train_label = total_label[train_indices]
    test_data = total_data[test_indices]
    test_label = total_label[test_indices]

    train_data_mean = np.mean(train_data)
    train_data_std = np.std(train_data)

    # train_label_mean = np.mean(train_label)
    # train_label_std = np.std(train_label)


    train_data = (train_data-train_data_mean)/train_data_std
    test_data = (test_data-train_data_mean)/train_data_std
    return train_data,train_label,test_data,test_label

for pt in pts:
    total_data,total_label = loadData(pt)
    train_data,train_label,test_data,test_label = splitData(total_data,total_label)


    input_dim = test_data.shape[1]
    output_dim = test_label.shape[1]


    model = transformer.Model(
        input_dim=input_dim,
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
    # test_dataset = TensorDataset(test_data,test_label)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)

    writer = SummaryWriter(f'./logs/{pt}')
    writer.add_graph(model,test_data)
    pbar = tqdm.trange(epochs, desc=f"Epochs")

    for e in pbar:
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

        # if (e + 1) % print_interval == 0:
        model.eval()
        test_outputs = model(test_data)
        test_loss = criterion(test_outputs, test_label).detach().cpu().numpy()
        aver_loss = aver_loss/len(train_dataloader)
        pbar.set_postfix({'average loss':aver_loss,'test loss':test_loss})
        writer.add_scalar('average loss',aver_loss,e)
        writer.add_scalar('test loss',test_loss,e)
        # log_write.write(f'{e}\t\t{aver_loss}\t\t{test_loss}\n')
        # log_write.flush()

    torch.save(model.state_dict(), f'./res/{pt}_model.pth')
    writer.close()

