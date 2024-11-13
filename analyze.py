import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 创建保存图片的目录
os.makedirs('./fold10res', exist_ok=True)

# 初始化数据存储
epoch_1000_data = []
max_test_pcc_data = []

mean_epoch_1000_pcc = []
mean_max_pcc = []

model_name = 'vqvae_cls_n512d1024_full'

# 遍历文件夹
for sub_num in range(1, 11):
    sub_folder = f'./logs/sub-{sub_num:02d}'
    
    mean_epoch_1000_pcc_fold = 0
    mean_max_pcc_fold = 0
    for vqvae_num in range(10):
        vqvae_log = f'{model_name}_{vqvae_num}.txt'
        log_file_path = os.path.join(sub_folder,model_name, vqvae_log)
        
        # 检查文件是否存在
        if os.path.exists(log_file_path):
            # 读取log.txt文件
            df = pd.read_csv(log_file_path)
            
            # 提取epoch=1000时的test_pcc值
            test_pcc_epoch_1000 = df[df['epoch'] == 1000]['test_pcc'].values
            if test_pcc_epoch_1000.size > 0:
                epoch_1000_data.append({
                    'sub': f'sub-{sub_num:02d}',
                    'vqvae': vqvae_num,
                    'test_pcc': test_pcc_epoch_1000[0]
                })
            mean_epoch_1000_pcc_fold += test_pcc_epoch_1000[0]
            
            # 提取test_pcc最高的值
            max_test_pcc = df['test_pcc'].max()
            max_test_pcc_data.append({
                'sub': f'sub-{sub_num:02d}',
                'vqvae': vqvae_num,
                'test_pcc': max_test_pcc
            })
            mean_max_pcc_fold+=max_test_pcc
    
    mean_epoch_1000_pcc.append(mean_epoch_1000_pcc_fold/10)
    mean_max_pcc.append(mean_max_pcc_fold/10)

# 转换为DataFrame
df_epoch_1000 = pd.DataFrame(epoch_1000_data)
df_max_test_pcc = pd.DataFrame(max_test_pcc_data)

# 绘制并保存小提琴图
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_epoch_1000, x='sub', y='test_pcc')
plt.title('Test PCC at Epoch 1000')
plt.xlabel('Subject')
plt.ylabel('Test PCC')
plt.savefig(f'./fold10res/{model_name}.png')  # 保存第一张图
plt.close()

with open(f'./fold10res/{model_name}.txt','w',newline='') as file:
    for data in mean_epoch_1000_pcc:
        file.write(f'{data}\n')
    file.write(f'mean:{np.mean(mean_epoch_1000_pcc)}')
