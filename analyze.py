import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--res_dir', default='./fold10res', type=str)
parser.add_argument('--model_name', default='', type=str)

argu = parser.parse_args()

# 创建保存图片的目录
os.makedirs(f'{argu.res_dir}', exist_ok=True)

# 初始化数据存储
epoch_1000_pcc_data = []
epoch_1000_mcd_eeg_data = []
max_test_pcc_data = []

mean_epoch_1000_pcc = []
mean_epoch_1000_mcd_eeg = []
mean_max_pcc = []

model_name = f'{argu.model_name}'

# 遍历文件夹
for sub_num in range(1, 11):
    sub_folder = f'./logs/sub-{sub_num:02d}'
    
    mean_epoch_1000_pcc_fold = 0
    mean_epoch_1000_mcd_eeg_fold = 0
    
    for vqvae_num in range(10):
        vqvae_log = f'{model_name}_{vqvae_num}.txt'
        log_file_path = os.path.join(sub_folder, model_name, vqvae_log)
        
        # 检查文件是否存在
        if os.path.exists(log_file_path):
            # 读取log.txt文件
            df = pd.read_csv(log_file_path)
            
            # 提取epoch=1000时的test_pcc值
            test_pcc_epoch_1000 = df[df['epoch'] == 1000]['test_pcc'].values
            # 提取epoch=1000时的test_mcd/eeg值
            test_mcd_eeg_epoch_1000 = df[df['epoch'] == 1000]['test_mcd/eeg'].values
            
            if test_pcc_epoch_1000.size > 0:
                epoch_1000_pcc_data.append({
                    'sub': f'sub-{sub_num:02d}',
                    'vqvae': vqvae_num,
                    'test_pcc': test_pcc_epoch_1000[0]
                })
                mean_epoch_1000_pcc_fold += test_pcc_epoch_1000[0]
            
            if test_mcd_eeg_epoch_1000.size > 0:
                epoch_1000_mcd_eeg_data.append({
                    'sub': f'sub-{sub_num:02d}',
                    'vqvae': vqvae_num,
                    'test_mcd_eeg': test_mcd_eeg_epoch_1000[0]
                })
                mean_epoch_1000_mcd_eeg_fold += test_mcd_eeg_epoch_1000[0]
    
    mean_epoch_1000_pcc.append(mean_epoch_1000_pcc_fold / 10)
    mean_epoch_1000_mcd_eeg.append(mean_epoch_1000_mcd_eeg_fold / 10)

# 转换为DataFrame
df_epoch_1000_pcc = pd.DataFrame(epoch_1000_pcc_data)
df_epoch_1000_mcd_eeg = pd.DataFrame(epoch_1000_mcd_eeg_data)
df_max_test_pcc = pd.DataFrame(max_test_pcc_data)

# 绘制并保存PCC图片
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_epoch_1000_pcc, x='sub', y='test_pcc')
plt.title('Test PCC at Epoch 1000')
plt.xlabel('Subject')
plt.ylabel('Test PCC')
plt.savefig(f'./fold10res/{model_name}_pcc.png')
plt.close()

# 绘制并保存MCD/EEG图片
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_epoch_1000_mcd_eeg, x='sub', y='test_mcd_eeg')
plt.title('Test MCD/EEG at Epoch 1000')
plt.xlabel('Subject')
plt.ylabel('Test MCD/EEG')
plt.savefig(f'./fold10res/{model_name}_mcd_eeg.png')
plt.close()

# 保存均值到文本文件
with open(f'./fold10res/{model_name}.txt', 'w', newline='') as file:
    for data in mean_epoch_1000_pcc:
        file.write(f'{data}\n')
    file.write(f'mean PCC: {np.mean(mean_epoch_1000_pcc)}\n')
    
    for data in mean_epoch_1000_mcd_eeg:
        file.write(f'{data}\n')
    file.write(f'mean MCD: {np.mean(mean_epoch_1000_mcd_eeg)}')
