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


    def forward(self,mel_semantic_res:torch.Tensor,eeg_semantic_res:torch.Tensor):
        B,T,D  = mel_semantic_res.shape
        t_samples = (torch.randint(T - self.prediction_step - self.min_start_step, size=(1,)) + self.min_start_step)
        nce = 0

        mel_encode_samples = torch.empty((self.prediction_step,B,self.embedding_dim),device=mel_semantic_res.device)
        eeg_encode_samples = torch.empty((self.prediction_step,B,self.embedding_dim),device=eeg_semantic_res.device)

        for i in range(1,self.prediction_step+1):
            mel_encode_samples[i-1,:,:] = mel_semantic_res[:,t_samples+i,:].reshape(B,self.embedding_dim)
            eeg_encode_samples[i-1,:,:] = eeg_semantic_res[:,t_samples+i,:].reshape(B,self.embedding_dim)
        
        mel_forward_seq = mel_semantic_res[:,:t_samples+1,:]
        eeg_forward_seq = eeg_semantic_res[:,:t_samples+1,:]

        mel_context,_ = self.audio_lstm(mel_forward_seq)
        eeg_context,_ = self.eeg_lstm(eeg_forward_seq)

        mel_context = mel_context[:,-1,:].reshape(B,self.context_dim)
        eeg_context = eeg_context[:,-1,:].reshape(B,self.context_dim)

        mel_pred = torch.empty((self.prediction_step,B,self.embedding_dim),device=mel_semantic_res.device)
        eeg_pred = torch.empty((self.prediction_step,B,self.embedding_dim),device=eeg_semantic_res.device)

        for i in range(self.prediction_step):
            mel_pred[i] = self.audio_predictor[i](mel_context).transpose
            eeg_pred[i] = self.eeg_predictor[i](eeg_context)
        
        for i in range(self.prediction_step):
            aa = torch.mm(mel_encode_samples[i],mel_pred[i].T)
            ee = torch.mm(eeg_encode_samples[i],eeg_pred[i].T)
            ae = torch.mm(mel_encode_samples[i],eeg_pred[i].T)
            ea = torch.mm(eeg_encode_samples[i],mel_pred[i].T)
            # aa_accuracy = torch.sum(torch.eq(torch.argmax(self.softmax(aa),dim = 0),0.))/B
            # ee_accuracy = torch.sum(torch.eq(torch.argmax(self.softmax(ee),dim = 0),0.))/B
            # ae_accuracy = torch.sum(torch.eq(torch.argmax(self.softmax(ae),dim = 0),0.))/B
            # ea_accuracy = torch.sum(torch.eq(torch.argmax(self.softmax(ea),dim = 0),0.))/B
            nce = nce + torch.sum(torch.diag(self.logsoftmax(ae))) + torch.sum(torch.diag(self.logsoftmax(ea)))\
                  +0.1*(torch.sum(torch.diag(self.logsoftmax(aa))) + torch.sum(torch.diag(self.logsoftmax(ee))))
        
        nce = -nce/(B*self.prediction_step)
        return nce
        # return aa_accuracy,ee_accuracy,ae_accuracy,ea_accuracy,nce
        

            

