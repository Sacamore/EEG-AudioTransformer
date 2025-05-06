import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class BaseModelHolder():
    def __init__(self):
        self.tensor_type = torch.cuda.FloatTensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def buildModel(self):
        pass

    def makeDataloader(self, batch_size, train_data, train_label, test_data, test_label):
        pass

    def train(self, data) -> float:
        pass

    def predict(self, data):
        pass
    
    def _toEval(self,models):
        for m in models:
            m.eval()

    def _toTrain(self,models):
        for m in models:
            m.train()
            
    def _toDevice(self,x):
        return x.to(self.device).type(self.tensor_type)

    def _weightsInit(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def overlap_add_segments(self, segments, hop_size):
        """
        将分段梅尔频谱拼接为完整序列（支持NumPy和PyTorch）
        
        参数：
        segments : 输入分段，形状 [batch, seg_size, n_mel]
        hop_size : 帧移步长（需满足 hop_size <= seg_size）
        
        返回：
        output : 拼接后的序列，形状 [(batch-1)*hop_size + seg_size, n_mel]
        """
        # 输入验证
        assert len(segments.shape) == 3, "输入必须是3维张量[batch, seg_size, n_mel]"
        batch, seg_size, n_mel = segments.shape
        assert hop_size <= seg_size, "帧移不能超过段长度"
        
        # 计算输出长度并初始化容器
        output_length = (batch - 1) * hop_size + seg_size
        output = np.zeros((output_length, n_mel), dtype=segments.dtype)
        
        # 创建叠加计数数组（用于后续归一化）
        overlap_counter = np.zeros(output_length, dtype=np.int32)
        
        # 遍历每个分段
        for i in range(batch):
            # 计算当前段的起始位置
            start = i * hop_size
            end = start + seg_size
            
            # 将当前段叠加到输出
            output[start:end] += segments[i]
            
            # 记录叠加次数
            overlap_counter[start:end] += 1
        
        # 处理重叠区域的归一化（平均叠加）
        overlap_mask = overlap_counter > 1
        output[overlap_mask] = output[overlap_mask] / overlap_counter[overlap_mask, None]
        
        return output
        
    def loadModel(self,state_dict):
        pass
    
    def saveModel(self,e):
        pass
    
    def getTextLogHeader(self):
        log_header = ["epoch", "test_loss",  "test_pcc", "test_mcd"]
        return log_header
    
    def getTSNE(self):
        return None,None
    