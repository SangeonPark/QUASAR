import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class lhc_encoder(BaseNet):

    def __init__(self, rep_dim=10):
        super().__init__()

        self.rep_dim = rep_dim
    
        self.encoder = nn.Sequential(
            nn.Linear(30, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 48),
            nn.LeakyReLU(True),            
            nn.Linear(48, rep_dim))
        
    def forward(self, x):
        x = self.encoder(x)
        return x


class lhc_decoder(BaseNet):

    def __init__(self, rep_dim=10):
        super().__init__()

        self.rep_dim = rep_dim

        self.decoder = nn.Sequential(
            nn.Linear(rep_dim, 48),
            nn.LeakyReLU(True),
            nn.Linear(48, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 30),
            nn.Sigmoid())

    def forward(self, x):
        x = self.decoder(x)
        return x


class lhc_autoencoder(BaseNet):

    def __init__(self, rep_dim=10):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = lhc_encoder(rep_dim=rep_dim)
        self.decoder = lhc_decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
