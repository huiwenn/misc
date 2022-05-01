import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable 

#import argoverse

import sys
import os
sys.path.append(os.path.dirname(__file__))

from EquiCtsConv import *
from EquiLinear import *
from .rho_reg_ECCO_corrected import ECCONetwork


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PECCONetworkMulti(nn.Module):
    def __init__(self, 
                 radius_scale = 40,
                 encoder_hidden_size = 22, 
                 layer_channels = [8, 16, 16, 16, 3],
                 modes=2
                 ):
        super(PECCONetworkMulti, self).__init__()

        self.modes = modes
        self.models = []
        for i in range(modes):
            thismode =  ECCONetwork(radius_scale=radius_scale,
                        layer_channels=layer_channels, 
                        encoder_hidden_size=encoder_hidden_size,
                        correction_scale=72)
            self.models.append(thismode)
            setattr(self, 'mode'+str(i), thismode)

        # lstm
        self.lstm_layers = 2
        self.hidden_size = encoder_hidden_size
        self.lstm = nn.LSTM(input_size=2, hidden_size=encoder_hidden_size,
                          num_layers=self.lstm_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(encoder_hidden_size*self.modes, 24) #fully connected 1
        self.fc = nn.Linear(24, self.modes) #fully connected last layer
        self.relu = nn.ReLU()
        self.m = nn.Softmax() 

    
    def forward(self, inputs, states=None):

        #inputs expected to be a list of len=modes
        
        pr_pos, pr_vel, sigma, statess = [], [], [], []
        hidden_states = []

        for i in range(self.modes): 
            if states:
                pr_pos1, pr_vel1, m1, states1 = self.models[i](inputs[i], states[i])
            else:
                pr_pos1, pr_vel1, m1, states1 = self.models[i](inputs[i])
            
            pr_m1 = m1[...,:2,:]
            
            pr_pos.append(pr_pos1)
            pr_vel.append(pr_vel1)
            sigma.append(pr_m1)
            statess.append(states1)
        
            # put in inputs here
            h_0 = Variable(torch.zeros(self.lstm_layers, pr_pos1.size(1), self.hidden_size,device=device)) #hidden state
            c_0 = Variable(torch.zeros(self.lstm_layers, pr_pos1.size(1), self.hidden_size, device=device)) #internal state
            # Propagate input through LSTM
            if states:
                #print('state shapes', states[i][0][...,1:,:].shape,inputs[i][1].shape, inputs[i][3].shape, pr_vel1.shape)
                x = torch.cat([states[i][0][...,1:,:],inputs[i][1], inputs[i][3].unsqueeze(-2), pr_vel1.unsqueeze(-2)], dim=-2)
            else:
                #print('nostate shapes', inputs[i][1].shape, inputs[i][3].unsqueeze(-2).shape, pr_vel1.unsqueeze(-2).shape)
                x = torch.cat([inputs[i][1],inputs[i][3].unsqueeze(-2), pr_vel1.unsqueeze(-2)], dim=-2)
            

            output, (hn, cn) = self.lstm(x.squeeze(), (h_0, c_0)) #lstm with input, hidden, and internal state
            #print('h_0',h_0.shape)
            #print('hn', hn.shape)
            hn = hn[-1] #.view(-1, self.hidden_size) #reshaping the data for Dense layer next
            #print('hn', hn.shape)

            hidden = self.relu(hn)
            hidden_states.append(hidden)
            #print('hidden', hidden.shape)
        
        if self.modes > 1:
            out = torch.cat(hidden_states, dim=1)
            out = self.fc_1(out) #first Dense
            out = self.relu(out) #relu
            p = self.fc(out) #Final Output
            p_norm = self.m(p).unsqueeze(0) # add dimention for batch
        else:
            p_norm = torch.ones(pr_pos[0].size(0), pr_pos[0].size(1), 1, device=device)

        print('p_norm', p_norm.shape, p_norm)
        return pr_pos, pr_vel, sigma, statess, p_norm 




