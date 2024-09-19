# EEG-AudioTransformer
我的毕设项目

文件结构  
- config：模型配置文件  
- feat：处理过的数据  
- model: 不同模型文件
- history: 历史文件，现已不再使用


cmg_downstream cmg模型下游训练
cmg_train cmg模型训练
dataset 数据集加载及处理
evaluate.ipynb 模型测试及部分可视化
extract_feature.ipynb 原始数据特征提取
mask_mel_vqvae.py 带遮掩的mel频谱vqvae
mfa_utils.py 使用mfa时调用的工具方法
train_words.ipynb 上古时期的训练方法
train.py 标准transformer训练方法
vqvae_cls.py 使用vqvae训练的分类器部分
vqvae_eeg.py 使用vqvae训练的eeg部分，现已弃用
vqvae_mel.py 使用vqvae训练的mel部分  

使用该项目时，应当：  
1. 调用vqvae_mel.py，训练mel频谱的自编码器  
2. 调用vqvae_cls.py，训练最终的分类器
