import torch
import torch.nn as nn
import torch.nn.functional as F

class CLUBSample_group(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample_group, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.p_mu = nn.Sequential(nn.Linear(self.x_dim, self.hidden_size // 2),
                                  nn.LeakyReLU(),
                                  nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                                  nn.LeakyReLU(),
                                  nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                                  nn.LeakyReLU(),
                                  nn.Linear(self.hidden_size // 2, self.y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(self.x_dim, self.hidden_size // 2),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.hidden_size // 2, self.hidden_size // 2),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.hidden_size // 2, self.y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)  # [B,T,D],[B,T,D]
        mu = mu.reshape(-1, mu.shape[-1]) #[BxT,D]
        logvar = logvar.reshape(-1, logvar.shape[-1]) #[BxT,D]
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])  # [BxT, D)
        res = (mu-y_samples)**2/logvar.exp()
        res = -res - logvar
        res = res.sum(dim=1)
        res = res.mean(dim=0) / 2
        return res
        # return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0) / 2 # C

    def mi_est(self, x_samples, y_samples): 

        mu, logvar = self.get_mu_logvar(x_samples) # [B,T,D],[B,T,D]

        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = - ((mu - y_samples) ** 2).mean(dim=1) / logvar.mean(dim=1).exp()  # [B,D]
        negative = - ((mu - y_samples[random_index]) ** 2).mean(dim=1) / logvar.mean(dim=1).exp()  # [B,D]

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean() / 2
