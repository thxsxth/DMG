import torch
import numpy as np
import pandas as pd
import datetime as dt
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import os
import glob
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import torch.nn.init as weight_init
import scipy
from torch.distributions.multivariate_normal import MultivariateNormal

import torch.distributions as D

device='cuda' if torch.cuda.is_available() else 'cpu'


class DRMG(nn.Module):
  """ Encodes x_{:t} 
      The job of the Encoder is to remember the past observations.
     
  """

  def __init__(self,input_dim,h_dim,n_layers=12,a_dim=2,dropout=0.0,out_dim=36,full_covar=True,mlp_dim=128):
      
      super(DRMG,self).__init__()
      
      self.rnn=nn.GRU(input_dim,h_dim,n_layers,batch_first=True)
      
      self.dropout=dropout  
      self.n_layers=n_layers
      self.hidden_dim=h_dim
      self.full_covar=full_covar
      self.init_weights()
      self.out_dim=out_dim

      self.means_in=nn.Linear(h_dim+a_dim,mlp_dim)
      #We will send the means through a small mlp
      self.mlp=nn.ModuleList([nn.Sequential(nn.Linear(mlp_dim,mlp_dim),nn.ELU()) for i in range(3)])
      
      """
      !!! Since we want to output a mixture of Gaussians need to output both means and Covs
      """
      self.means_out=nn.Linear(mlp_dim,out_dim*2)
      
      self.components=nn.Linear(h_dim+a_dim,2)
      
      
      """
      Again need to output both sets of covaraince matrices
      """
      self.obs_covar=nn.Linear(h_dim+a_dim,out_dim*out_dim*2)
      self.diag=nn.Linear(h_dim+a_dim,out_dim*2) #The diagonals of a covariance, we will make this positive using softplus

      


  def  init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)

  def forward(self,obs,obs_lens,acts,init_h=None, noise=False):

    
    batch_size, max_len, freq=obs.size()  
    obs_lens=torch.LongTensor(obs_lens).to(device)
    obs_lens_sorted, indices = obs_lens.sort(descending=True)
    obs_sorted = obs.index_select(0, indices).to(device)
    
    packed_obs=pack_padded_sequence(obs_sorted,obs_lens_sorted.data.tolist(),batch_first=True)
    # if init_h is None:
    #     init_h=self.init_h

     
    
    hids, h_n = self.rnn(packed_obs) # hids: [B x T x H]  
                                                  # h_n: [num_layers*B*H)
    _, inv_indices = indices.sort()

    hids, lens = pad_packed_sequence(hids, batch_first=True)         
    hids = hids.index_select(0, inv_indices) #B*T*H
    
    """
    Concat hids with actions
    
    """
    hids=torch.cat([hids,acts],dim=-1)  #B*T*(h+a)

    sigma=self.obs_covar(hids).view(batch_size,max_len,2,self.out_dim,self.out_dim)   #B*T*(O*(2*O))-->B*T*2*(0*O)
    diags=self.diag(hids).view((batch_size,max_len,2,self.out_dim))  #B*T*(2*O)-->B*T*2*0

    cov=((torch.tril(torch.ones(batch_size,max_len,2,self.out_dim,self.out_dim,device=device),
    diagonal=-1)*sigma)+torch.diag_embed(F.softplus(diags))).to(device)




    ### Find the means
    means_in=F.elu(self.means_in(hids))
    

    for layer in self.mlp:
      means_in=layer(means_in)
     
    means=self.means_out(means_in).view(batch_size,max_len,2,self.out_dim)  #B*T*(O*2)---> B*T*2*0
    print('Means Shape :',means.shape)
    comps=D.Independent(MultivariateNormal(means,scale_tril=cov),0)

    ##Make sure the number of components are here in 2nd dim
    # print('Comps Sample shape',comps.sample().shape,'Batch Shape', comps.batch_shape)
    mixes=self.components(hids).view(batch_size,-1,2) #B*T*2
    # print(mixes.shape)

    

    mix=D.Categorical(logits=mixes)

    
    return D.MixtureSameFamily(mix,comps)