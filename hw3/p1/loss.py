import torch
import torch.nn as nn

class VAE_Loss(nn.Module):
    def __init__(self, lambda_kld, bce=True):
        super(VAE_Loss,self).__init__()
        self.MSE_Loss = nn.MSELoss()
        self.BCE_Loss = nn.BCELoss()
        self.kld_weight = lambda_kld 

    def forward(self, _recons, _input, mean, logvar):

        mse_loss = self.MSE_Loss(_recons, _input)
        bce_loss = self.BCE_Loss(_recons, _input)

        kld_loss   = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        #loss = bce_loss + self.kld_weight * kld_loss
        loss = mse_loss + self.kld_weight * kld_loss
        return loss, mse_loss, kld_loss


