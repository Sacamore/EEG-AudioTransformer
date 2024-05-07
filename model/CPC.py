import torch
import torch.nn as nn
import torch.nn.functional as F


class Cross_CPC(nn.Module):
    def __init__(self,embedding_dim:int,hidden_dim:int,context_dim:int,num_layers:int,predict_step:int=1,min_start_step:int=1) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.prediction_step = predict_step
        self.min_start_step = min_start_step

        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax()

        self.audio_lstm = nn.LSTM(embedding_dim,context_dim,num_layers,batch_first=True)
        self.eeg_lstm = nn.LSTM(embedding_dim,context_dim,num_layers,batch_first=True)
        
        self.audio_predictor = nn.ModuleList([
            nn.Linear(context_dim,context_dim) for _ in range(predict_step-1)
        ])

        self.eeg_predictor = nn.ModuleList([
            nn.Linear(context_dim,context_dim) for _ in range(predict_step-1)
        ])


    def forward(self,audio_vq:torch.Tensor,eeg_vq:torch.Tensor):
        B,T,D  = audio_vq.shape
        t_samples = (torch.randint(T - self.prediction_step - self.min_start_step, size=(1,)) + self.min_start_step).long()
        nce = 0

        audio_encode_samples = torch.empty((self.prediction_step,B,self.embedding_dim),device=audio_vq.device).double()
        eeg_encode_samples = torch.empty((self.prediction_step,B,self.embedding_dim),device=eeg_vq.device).double()

        for i in range(1,self.prediction_step+1):
            audio_encode_samples[i-1,:,:] = audio_vq[:,t_samples+i,:].reshape(B,self.embedding_dim)
            eeg_encode_samples[i-1,:,:] = eeg_vq[:,t_samples+i,:].reshape(B,self.embedding_dim)
        
        audio_forward_seq = audio_vq[:,:t_samples+1,:]
        eeg_forward_seq = eeg_vq[:,:t_samples+1,:]

        audio_context,_ = self.audio_lstm(audio_forward_seq)
        eeg_context,_ = self.eeg_lstm(eeg_forward_seq)

        audio_context = audio_context[:,-1,:].reshape(B,self.context_dim)
        eeg_context = eeg_context[:,-1,:].reshape(B,self.context_dim)

        audio_pred = torch.empty((self.prediction_step,B,self.embedding_dim),device=audio_vq.device).double()
        eeg_pred = torch.empty((self.prediction_step,B,self.embedding_dim),device=eeg_vq.device).double

        for i in range(self.prediction_step):
            audio_pred[i] = self.audio_predictor[i](audio_context).transpose
            eeg_pred[i] = self.eeg_predictor[i](eeg_context)
        
        for i in range(self.prediction_step):
            aa = torch.mm(audio_encode_samples[i],audio_pred[i].T)
            ee = torch.mm(eeg_encode_samples[i],eeg_pred[i].T)
            ae = torch.mm(audio_encode_samples[i],eeg_pred[i].T)
            ea = torch.mm(eeg_encode_samples[i],audio_pred[i].T)
            aa_accuracy = torch.sum(torch.eq(torch.argmax(self.softmax(aa),dim = 0),0.))/B
            ee_accuracy = torch.sum(torch.eq(torch.argmax(self.softmax(ee),dim = 0),0.))/B
            ae_accuracy = torch.sum(torch.eq(torch.argmax(self.softmax(ae),dim = 0),0.))/B
            ea_accuracy = torch.sum(torch.eq(torch.argmax(self.softmax(ea),dim = 0),0.))/B
            nce = nce + torch.sum(torch.diag(self.logsoftmax(ae))) + torch.sum(torch.diag(self.logsoftmax(ea)))\
                  +0.1*(torch.sum(torch.diag(self.logsoftmax(aa))) + torch.sum(torch.diag(self.logsoftmax(ee))))
        
        nce = -nce/(B*self.prediction_step)

        return aa_accuracy,ee_accuracy,ae_accuracy,ea_accuracy,nce
        

            

