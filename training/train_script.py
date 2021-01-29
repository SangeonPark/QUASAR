#!/usr/bin/env python
# coding: utf-8

# In[1]:

print('here')
import numpy as np
import scipy as sp
import scipy.stats
import itertools
import logging
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as utils
import math
import time
import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal

import torch.nn as nn
import torch.nn.init as init
import sys

sys.path.append("../new_flows")
from flows import RealNVP, Planar, MAF
from models import NormalizingFlowModel


# ## Load and process the data

# In[2]:


df_rndbkg = pd.read_hdf("/data/t3home000/spark/QUASAR/preprocessing/conventional_tau_rnd.h5")


# In[3]:


dt = df_rndbkg.values
correct = (dt[:,3]>0) &(dt[:,19]>0) & (dt[:,1]>0) & (dt[:,2]>0)
dt = dt[correct]

for i in range(13,19):
    dt[:,i] = dt[:,i]/dt[:,3]

for i in range(29,35):
    dt[:,i] = dt[:,i]/(dt[:,19])


correct = (dt[:,0]>=2800)
dt = dt[correct]


#Y = dt[:,[3,4,5,6,11,12,19,20,21,22,27,28]]
idx = dt[:,-1]
bkg_idx = np.where(idx==0)[0]
signal_idx = np.where(idx==1)[0]
dt = dt[bkg_idx]
#Y = dt[:,[3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22,27,28,29,30,31,32,33,34]]
#Y = dt[:,[3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22,27,28,29,30,31,32,33,34]]


# In[4]:


print(dt.shape)


# In[5]:


Y = dt[:,[3,4,5,6,11,12,19,20,21,22,27,28]]


# In[6]:


print(Y.shape)


# In[7]:



#if whichbkg == 'rndbkg':
#    dt = df_rndbkg.values
#    correct = (dt[:,3]>0) &(dt[:,19]>0) &(dt[:,1]>0) &(dt[:,2]>0)&(dt[:,16]>0)&(dt[:,32]>0)
#    dt = dt[correct]
#    for i in range(13,19):
#        dt[:,i] = dt[:,i]/dt[:,3]
#
#    for i in range(29,35):
#        dt[:,i] = dt[:,i]/(dt[:,19])
#
#    correct = (dt[:,16]>0) &(dt[:,29]>=0) &(dt[:,29]<=1)&(dt[:,30]>=0) &(dt[:,30]<=1)&(dt[:,31]>=0) &(dt[:,31]<=1)&(dt[:,32]>=0) &(dt[:,32]<=1)&(dt[:,33]>=0) &(dt[:,33]<=1)&(dt[:,34]>=-0.01) &(dt[:,34]<=1)
#    dt = dt[correct]
#    correct = (dt[:,0]>=2800)
#    dt = dt[correct]
#    #Y = dt[:,[3,4,5,6,11,12,19,20,21,22,27,28]]
#    idx = dt[:,-1]
#    bkg_idx = np.where(idx==0)[0]
#    signal_idx = np.where(idx==1)[0]
#    dt = dt[bkg_idx]
#    #Y = dt[:,[3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22,27,28,29,30,31,32,33,34]]
#    Y = dt[:,[3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22,27,28,29,30,31,32,33,34]]


# In[8]:


bkg_mean = []
bkg_std = []


# In[9]:


for i in range(12):
    mean = np.mean(Y[:,i])
    std = np.std(Y[:,i])
    bkg_mean.append(mean)
    bkg_std.append(std)
    Y[:,i] = (Y[:,i]-mean)/std


# In[10]:




total_PureBkg = torch.tensor(Y)

total_PureBkg_selection = total_PureBkg




bs = 800
bkgAE_train_iterator = utils.DataLoader(total_PureBkg_selection, batch_size=bs, shuffle=True)
bkgAE_test_iterator = utils.DataLoader(total_PureBkg_selection, batch_size=bs)


# ## Build the model

# In[17]:


####MAF
class VAE_NF(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.dim = D
        self.K = K
        self.encoder = nn.Sequential(
            nn.Linear(12, 50),
            nn.LeakyReLU(True),
            nn.Linear(50, 30),
            nn.LeakyReLU(True),
            nn.Linear(30, 20),
            nn.LeakyReLU(True),
            nn.Linear(20, D * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(D, 20),
            nn.LeakyReLU(True),
            nn.Linear(20, 30),
            nn.LeakyReLU(True),
            nn.Linear(30, 50),
            nn.LeakyReLU(True),
            nn.Linear(50, 12)
        )

        flow_init = MAF(dim=D)
        flows_init = [flow_init for _ in range(K)]
        prior = MultivariateNormal(torch.zeros(D).cuda(), torch.eye(D).cuda())
        self.flows = NormalizingFlowModel(prior, flows_init)

    def forward(self, x):
        # Run Encoder and get NF params
        enc = self.encoder(x)
        mu = enc[:, :self.dim]
        log_var = enc[:, self.dim: self.dim * 2]

        # Re-parametrize
        sigma = (log_var * .5).exp()
        z = mu + sigma * torch.randn_like(sigma)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Construct more expressive posterior with NF

        z_k, _, sum_ladj = self.flows(z)

        kl_div = kl_div / x.size(0) - sum_ladj.mean()  # mean over batch

        # Run Decoder
        x_prime = self.decoder(z_k)
        return x_prime, kl_div


# ## Creating Instance¶

# In[19]:


def train():
    global n_steps
    train_loss = []
    model.train()

    for batch_idx, x in enumerate(bkgAE_train_iterator):
        start_time = time.time()

        x = x.float().cuda()

        x_tilde, kl_div = model(x)



        mseloss = nn.MSELoss(size_average=False)

        huberloss = nn.SmoothL1Loss(size_average=False)


        #loss_recons = F.binary_cross_entropy(x_tilde, x, size_average=False) / x.size(0)
        loss_recons = mseloss(x_tilde,x ) / x.size(0)

        #loss_recons = huberloss(x_tilde,x ) / x.size(0)
        loss = loss_recons + beta * kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append([loss_recons.item(), kl_div.item()])

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {:5.3f} ms/batch beta:{}'.format(
                batch_idx * len(x), 50000,
                PRINT_INTERVAL * batch_idx / 50000,
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                1000 * (time.time() - start_time),
                beta

            ))

        n_steps += 1


# In[20]:


def evaluate(split='valid'):
    global n_steps
    start_time = time.time()
    val_loss = []
    model.eval()

    with torch.no_grad():
        for batch_idx, x in enumerate(bkgAE_test_iterator):

            x = x.float().cuda()

            x_tilde, kl_div = model(x)
            mseloss = nn.MSELoss(size_average=False)
            #loss_recons = F.binary_cross_entropy(x_tilde, x, size_average=False) / x.size(0)
            huberloss = nn.SmoothL1Loss(size_average=False)


            #loss_recons = F.binary_cross_entropy(x_tilde, x, size_average=False) / x.size(0)
            loss_recons = mseloss(x_tilde,x ) / x.size(0)
            #loss_recons = huberloss(x_tilde,x ) / x.size(0)
            loss = loss_recons + beta * kl_div

            val_loss.append(loss.item())
            #writer.add_scalar('loss/{}/ELBO'.format(split), loss.item(), n_steps)
            #writer.add_scalar('loss/{}/reconstruction'.format(split), loss_recons.item(), n_steps)
            #writer.add_scalar('loss/{}/KL'.format(split), kl_div.item(), n_steps)

    print('\nEvaluation Completed ({})!\tLoss: {:5.4f} Time: {:5.3f} s'.format(
        split,
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


# In[21]:


#version = 1
#zdim = [1,3,5,7,9,10]
#nflow = [1,3,5,7,9,10]
#lrs = [1e-3,1e-4,1e-5,1e-6,1e-7]
#betas = [0.1,0.5,1.0,2.0,10.0]#

#version = 0
#zdim = [2,4,6,8,10]
#nflow = [2,4,6,8,10]
#lrs = [1e-3,1e-4,1e-5,1e-6,1e-7]
#betas = [0.1,0.5,1.0,2.0,10.0]


# In[22]:


version = 0
#zdim = [2,4,6,8,10]
zdim = [4,6,8,10]
nflow = [2,4,6,8,10]
lrs = [5e-3, 1e-3,5e-4, 1e-4,1e-5,1e-6,1e-7]
betas = [0.1,0.5,1.0,2.0,10.0]


# In[43]:


N_EPOCHS = 30
PRINT_INTERVAL = 400
NUM_WORKERS = 4
n_steps = 0


# In[44]:


import re


# In[ ]:


for Z_DIM in zdim:
    for N_FLOWS in nflow:
        for beta in betas:
            model = VAE_NF(N_FLOWS, Z_DIM).cuda()
            ae_def = {
                        "type":"bkg",
                        "trainon":"rndbkg",
                        "features":"12features",
                        "architecture":"MAF",
                        "selection":"mjjcut",
                        "trainloss":"MSELoss",
                        "beta":f"beta{re.sub('[.,]', 'p', str(beta))}",
                        "zdimnflow":f"z{Z_DIM}f{N_FLOWS}",
                        "version":f"ver{version}"

                     }

            BEST_LOSS = float('inf')
            for LR in lrs:
                optimizer = optim.Adam(model.parameters(), lr=LR)
                LAST_SAVED = -1
                PATIENCE_COUNT = 0
                for epoch in range(1, 1000):
                    print("Epoch {}:".format(epoch))
                    train()
                    cur_loss = evaluate()
                    print(cur_loss)

                    if cur_loss <= BEST_LOSS:
                        PATIENCE_COUNT = 0
                        BEST_LOSS = cur_loss
                        LAST_SAVED = epoch
                        print("Saving model!")
                        torch.save(model.state_dict(),f"/data/t3home000/spark/QUASAR/weights/{ae_def['type']}_{ae_def['trainon']}_{ae_def['features']}_{ae_def['architecture']}_{ae_def['selection']}_{ae_def['trainloss']}_{ae_def['beta']}_{ae_def['zdimnflow']}_{ae_def['version']}.h5")

                    else:
                        PATIENCE_COUNT += 1
                        print("Not saving model! Last saved: {}".format(LAST_SAVED))
                        if PATIENCE_COUNT > 10:
                            print(f"############Patience Limit Reached with LR={LR}, Best Loss={BEST_LOSS}")
                            break

                model.load_state_dict(torch.load(f"/data/t3home000/spark/QUASAR/weights/{ae_def['type']}_{ae_def['trainon']}_{ae_def['features']}_{ae_def['architecture']}_{ae_def['selection']}_{ae_def['trainloss']}_{ae_def['beta']}_{ae_def['zdimnflow']}_{ae_def['version']}.h5"))



version = 0
#zdim = [2,4,6,8,10]
zdim = [2]
nflow = [8,10]
lrs = [5e-3, 1e-3,1e-4,1e-5,1e-6,1e-7]
betas = [0.1,0.5,1.0,2.0,10.0]

for Z_DIM in zdim:
    for N_FLOWS in nflow:
        for beta in betas:
            model = VAE_NF(N_FLOWS, Z_DIM).cuda()
            ae_def = {
                        "type":"bkg",
                        "trainon":"rndbkg",
                        "features":"12features",
                        "architecture":"MAF",
                        "selection":"mjjcut",
                        "trainloss":"MSELoss",
                        "beta":f"beta{re.sub('[.,]', 'p', str(beta))}",
                        "zdimnflow":f"z{Z_DIM}f{N_FLOWS}",
                        "version":f"ver{version}"

                     }

            BEST_LOSS = float('inf')
            for LR in lrs:
                optimizer = optim.Adam(model.parameters(), lr=LR)
                LAST_SAVED = -1
                PATIENCE_COUNT = 0
                for epoch in range(1, 1000):
                    print("Epoch {}:".format(epoch))
                    train()
                    cur_loss = evaluate()

                    if cur_loss <= BEST_LOSS:
                        PATIENCE_COUNT = 0
                        BEST_LOSS = cur_loss
                        LAST_SAVED = epoch
                        print("Saving model!")
                        torch.save(model.state_dict(),f"/data/t3home000/spark/QUASAR/weights/{ae_def['type']}_{ae_def['trainon']}_{ae_def['features']}_{ae_def['architecture']}_{ae_def['selection']}_{ae_def['trainloss']}_{ae_def['beta']}_{ae_def['zdimnflow']}_{ae_def['version']}.h5")

                    else:
                        PATIENCE_COUNT += 1
                        print("Not saving model! Last saved: {}".format(LAST_SAVED))
                        if PATIENCE_COUNT > 10:
                            print(f"############Patience Limit Reached with LR={LR}, Best Loss={BEST_LOSS}")
                            break

                model.load_state_dict(torch.load(f"/data/t3home000/spark/QUASAR/weights/{ae_def['type']}_{ae_def['trainon']}_{ae_def['features']}_{ae_def['architecture']}_{ae_def['selection']}_{ae_def['trainloss']}_{ae_def['beta']}_{ae_def['zdimnflow']}_{ae_def['version']}.h5"))

