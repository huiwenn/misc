import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argoverse

import sys
import os
sys.path.append(os.path.dirname(__file__))

from EquiCtsConv import *
from EquiLinear import *


class VehicleEncoder(nn.Module):
    def __init__(self, 
                 num_radii = 3, 
                 num_theta = 16, 
                 reg_dim = 8,
                 radius_scale = 40,
                 timestep = 0.1,
                 in_channel = 19, 
                 layer_channels = [8, 16, 16]
                 ):
        super(VehicleEncoder, self).__init__()
        
        # init parameters
        
        self.num_radii = num_radii
        self.num_theta = num_theta
        self.reg_dim = reg_dim
        self.radius_scale = radius_scale
        self.timestep = timestep
        self.layer_channels = layer_channels
        
        self.in_channel = in_channel
        self.activation = F.relu
        # self.relu_shift = torch.nn.parameter.Parameter(torch.tensor(0.2))
        # relu_shift = torch.tensor(0.2)
        # self.register_buffer('relu_shift', relu_shift)
        
        # create continuous convolution and fully-connected layers
        
        convs = []
        denses = []
        # c_in, c_out, radius, num_radii, num_theta
        self.conv_vehicle = EquiCtsConv2dRho1ToReg(in_channels = self.in_channel, 
                                                 out_channels = self.layer_channels[0],
                                                 num_radii = self.num_radii, 
                                                 num_theta = self.num_theta,
                                                 radius = self.radius_scale, 
                                                 k = self.reg_dim)

        self.dense_vehicle = nn.Sequential(
            EquiLinearRho1ToReg(self.reg_dim), 
            EquiLinearRegToReg(self.in_channel, self.layer_channels[0], self.reg_dim)
        )
        
        # concat conv_obstacle, conv_fluid, dense_fluid
        in_ch = 2 * self.layer_channels[0] 
        for i in range(1, len(self.layer_channels)):
            out_ch = self.layer_channels[i]
            dense = EquiLinearRegToReg(in_ch, out_ch, self.reg_dim)
            denses.append(dense)
            conv = EquiCtsConv2dRegToReg(in_channels = in_ch, 
                                         out_channels = out_ch,
                                         num_radii = self.num_radii, 
                                         num_theta = self.num_theta,
                                         radius = self.radius_scale, 
                                         k = self.reg_dim)
            convs.append(conv)
            in_ch = self.layer_channels[i]
        
        self.convs = nn.ModuleList(convs)
        self.denses = nn.ModuleList(denses)
        

    def encode(self, p, vehicle_feats, car_mask):

        output_conv_vehicle = self.conv_vehicle(p, p, vehicle_feats, car_mask)
        output_dense_vehicle = self.dense_vehicle(vehicle_feats)
        
        output = torch.cat((output_conv_vehicle, output_dense_vehicle), -2)
        
        for conv, dense in zip(self.convs, self.denses):
            # pass input features to conv and fully-connected layers
            # mags = (torch.sum(output**2,axis=-1) + 1e-6).unsqueeze(-1)
            # in_feats = output/mags * self.activation(mags - self.relu_shift)
            in_feats = self.activation(output)
            output_conv = conv(p, p, in_feats, car_mask)
            output_dense = dense(in_feats)
            
            if output_dense.shape[-2] == output.shape[-2]:
                output = output_conv + output_dense + output
            else:
                output = output_conv + output_dense

        output = self.activation(output)
        
        return output
    
    def forward(self, inputs):
        """ inputs: 8 elems tuple
        p0_enc, v0_enc, p0, v0, feats, box, box_feats
        v0_enc: [batch, num_part, timestamps, 2]
        """
        p0_enc, v0_enc, p0, v0, car_mask = inputs
            
        feats = torch.cat((v0.unsqueeze(-2), v0_enc), -2)

        hidden_feature = self.encode(p0, feats, car_mask)

        return hidden_feature
    

class MapEncoder(nn.Module):
    def __init__(self, 
                 num_radii = 3, 
                 num_theta = 16, 
                 reg_dim = 8,
                 radius_scale = 40,
                 hidden_size: int = 8
                 ):
        super(MapEncoder, self).__init__()
        
        # init parameters
        
        self.num_radii = num_radii
        self.num_theta = num_theta
        self.reg_dim = reg_dim
        self.radius_scale = radius_scale
        self.hidden_size = hidden_size
        
        self.activation = F.relu
        # self.relu_shift = torch.nn.parameter.Parameter(torch.tensor(0.2))
        # relu_shift = torch.tensor(0.2)
        # self.register_buffer('relu_shift', relu_shift)
        
        # create continuous convolution and fully-connected layers
        
        # c_in, c_out, radius, num_radii, num_theta
        self.conv_map = EquiCtsConv2dRho1ToReg(in_channels = 1, 
                                               out_channels = self.hidden_size,
                                               num_radii = self.num_radii, 
                                               num_theta = self.num_theta,
                                               radius = self.radius_scale, 
                                               k = self.reg_dim)

    def forward(self, p, map_p, map_feat, map_mask):
        
        output = self.conv_map(map_p, p, map_feat.unsqueeze(-2), map_mask)
        
        output = self.activation(output)

        return output
    
    
class ModeDecoder(nn.Module):
    def __init__(self, vehicle_hidden=16, map_hidden=8, reg_dim=8, modes=6):
        super(ModeDecoder, self).__init__()
        in_channel = vehicle_hidden + map_hidden
        self.modes = modes
        if modes == 1:
            self.mode_decoder = lambda x: None
        else:
            self.mode_decoder = EquiLinearRegToReg(in_channel, modes, reg_dim)
        
    def forward(self, feat):
        """
        feat: shape (batch, num_vehicles, v+m, reg_dim)
        
        return: shape (batch, modes)
        """
        if self.modes == 1:
            return self.mode_decoder(feat)
        else:
            # mode_pred, _ = self.mode_decoder(feat).norm(dim=-1).topk(k=1, dim=1)
            # mode_pred = mode_pred.squeeze(1)
            mode_pred = self.mode_decoder(feat).norm(dim=-1).permute(0,2,1)
            mode_pred = F.max_pool1d(mode_pred, mode_pred.shape[-1]).squeeze(-1)
            return F.softmax(mode_pred, -1)
        
        
class TrajectoryDecoder(nn.Module):
    def __init__(self, 
                 num_radii = 3, 
                 num_theta = 16, 
                 reg_dim = 8,
                 radius_scale = 40,
                 timestep = 0.1,
                 vehicle_hidden = 16, 
                 map_hidden = 8,
                 layer_channels = [8, 8, 3], 
                 predict_window = 30, 
                 map_encoder = None
                 ):
        super(TrajectoryDecoder, self).__init__()
        
        # init parameters
        
        self.num_radii = num_radii
        self.num_theta = num_theta
        self.reg_dim = reg_dim
        self.radius_scale = radius_scale
        self.timestep = timestep
        self.predict_window = predict_window
        self.vehicle_hidden = vehicle_hidden
        self.map_hidden = map_hidden
        self.layer_channels = layer_channels
        
        self.in_channel = vehicle_hidden + map_hidden
        self.activation = F.relu
        # self.relu_shift = torch.nn.parameter.Parameter(torch.tensor(0.2))
        # relu_shift = torch.tensor(0.2)
        # self.register_buffer('relu_shift', relu_shift)
        
        # create continuous convolution and fully-connected layers
        
        convs = []
        denses = []
        # c_in, c_out, radius, num_radii, num_theta
        self.conv_vehicle = EquiCtsConv2dRegToReg(in_channels = self.in_channel, 
                                                   out_channels = self.layer_channels[0],
                                                   num_radii = self.num_radii, 
                                                   num_theta = self.num_theta,
                                                   radius = self.radius_scale, 
                                                   k = self.reg_dim)
        
        if map_encoder:
            self.map_encoder = map_encoder
        else:
            self.map_encoder = MapEncoder(num_radii = num_radii, 
                                          num_theta = num_theta, 
                                          reg_dim = reg_dim,
                                          radius_scale = radius_scale,
                                          hidden_size = map_hidden)

        self.dense_vehicle = EquiLinearRegToReg(self.in_channel, self.layer_channels[0], self.reg_dim)
        
        # concat conv_obstacle, conv_fluid, dense_fluid
        in_ch = self.layer_channels[0] 
        for i in range(1, len(self.layer_channels)):
            out_ch = self.layer_channels[i]
            dense = EquiLinearRegToReg(in_ch, out_ch, self.reg_dim)
            denses.append(dense)
            conv = EquiCtsConv2dRegToReg(in_channels = in_ch, 
                                         out_channels = out_ch,
                                         num_radii = self.num_radii, 
                                         num_theta = self.num_theta,
                                         radius = self.radius_scale, 
                                         k = self.reg_dim)
            convs.append(conv)
            in_ch = self.layer_channels[i]
        
        self.convs = nn.ModuleList(convs)
        self.denses = nn.ModuleList(denses)
        
        self.dense_back = EquiLinearRegToReg(self.layer_channels[-1], vehicle_hidden, self.reg_dim)
        self.dense_reg2rho1 = EquiLinearRegToRho1(self.reg_dim)
        
    def decode(self, p, feat, map_p, map_feat, car_mask, map_mask):
        output_conv_vehicle = self.conv_vehicle(p, p, feat, car_mask)
        output_dense_vehicle = self.dense_vehicle(feat)
        
        output = output_conv_vehicle + output_dense_vehicle
        
        for conv, dense in zip(self.convs, self.denses):
            # pass input features to conv and fully-connected layers
            # mags = (torch.sum(output**2,axis=-1) + 1e-6).unsqueeze(-1)
            # in_feats = output/mags * self.activation(mags - self.relu_shift)
            in_feats = self.activation(output)
            output_conv = conv(p, p, in_feats, car_mask)
            output_dense = dense(in_feats)
            
            if output_dense.shape[-2] == output.shape[-2]:
                output = output_conv + output_dense + output
            else:
                output = output_conv + output_dense

        output = self.activation(output)

        return output
    
    def forward(self, p, feat, map_p, map_feat, car_mask, map_mask):
        outputs = []
        
        output = self.decode(p, feat, map_p, map_feat, car_mask, map_mask)
        delta_p_dist = self.dense_reg2rho1(output)
        pred = delta_p_dist.clone()
        pred[...,0,:] = pred[...,0,:] + p
        outputs.append(pred)
        
        for t in range(1, self.predict_window):
            p = p + delta_p_dist[...,0,:]
            back_feat = torch.tanh(self.dense_back(output))
            
            feat = feat[...,:self.vehicle_hidden,:] * back_feat
            encode_map = self.map_encoder(p, map_p, map_feat, map_mask)
            feat = torch.cat([feat, encode_map], dim=-2)
            
        
            output = self.decode(p, feat, map_p, map_feat, car_mask, map_mask)
            delta_p_dist = self.dense_reg2rho1(output)
            pred = delta_p_dist.clone()
            pred[...,0,:] = pred[...,0,:] + p
            outputs.append(pred)
            
        return outputs
    
    def reset_predict_window(self, window):
        self.predict_window = window
        
        
class MultiModePECCO(nn.Module):
    def __init__(self, 
                 num_radii = 3, 
                 num_theta = 16, 
                 reg_dim = 8,
                 radius_scale = 40,
                 timestep = 0.1,
                 in_channel = 19,
                 map_hidden = 8, 
                 encoder_channels = [8, 16, 16],
                 decoder_channels = [8, 3], 
                 predict_window = 30, 
                 modes = 6):
        super(MultiModePECCO, self).__init__()
        
        self.modes = modes
        
        self.vehicle_encoder = VehicleEncoder(num_radii = num_radii, 
                                              num_theta = num_theta, 
                                              reg_dim = reg_dim,
                                              radius_scale = radius_scale,
                                              timestep = timestep,
                                              in_channel = in_channel, 
                                              layer_channels = encoder_channels)
        
        self.map_encoder = MapEncoder(num_radii = num_radii, 
                                      num_theta = num_theta, 
                                      reg_dim = reg_dim,
                                      radius_scale = radius_scale,
                                      hidden_size = map_hidden)
        
        self.mode_decoder = ModeDecoder(vehicle_hidden=encoder_channels[-1], 
                                        map_hidden=map_hidden, 
                                        reg_dim=reg_dim, modes=modes)
        
        self.traj_decoder = []
        for m in range(modes):
            traj_decoder_m = TrajectoryDecoder(num_radii = num_radii, 
                                               num_theta = num_theta, 
                                               reg_dim = reg_dim,
                                               radius_scale = radius_scale,
                                               timestep = timestep,
                                               vehicle_hidden = encoder_channels[-1], 
                                               map_hidden = map_hidden,
                                               layer_channels = decoder_channels, 
                                               predict_window = predict_window, 
                                               map_encoder = self.map_encoder)
            self.traj_decoder.append(traj_decoder_m)
            
        self.traj_decoder = nn.ModuleList(self.traj_decoder)
        
    def forward(self, inputs):
        p_enc, v_enc, p, v, map_p, map_feat, car_mask, map_mask = inputs
        
        vehicle_hidden = self.vehicle_encoder((p_enc, v_enc, p, v, car_mask))
        map_hidden = self.map_encoder(p, map_p, map_feat, map_mask)
        
        feat = torch.cat([vehicle_hidden, map_hidden], dim=-2)
        
        traj_preds = []
        for m in range(self.modes):
            traj_pred = self.traj_decoder[m](p, feat, map_p, map_feat, car_mask, map_mask)
            traj_preds.append(traj_pred)
            
        mode_pred = self.mode_decoder(feat)
        
        return traj_preds, mode_pred
    
    def reset_predict_window(self, window):
        for m in range(self.modes):
            self.traj_decoder[m].reset_predict_window(window)
            