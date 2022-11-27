import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import argoverse

import sys
import os
sys.path.append(os.path.dirname(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ParticlesNetwork(nn.Module):
    def __init__(self, 
                 kernel_sizes = [4, 4, 4],
                 radius_scale = 40,
                 coordinate_mapping = 'ball_to_cube',
                 interpolation = 'linear',
                 use_window = True,
                 particle_radius = 0.5,
                 timestep = 0.1,
                 encoder_hidden_size = 66 # 19*3+8+1
                 ):
        super(ParticlesNetwork, self).__init__()
        
        # init parameters
        
        self.kernel_sizes = kernel_sizes
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.timestep = timestep
        self.layer_channels = [32, 64, 64, 64, 6]
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        
        self.encoder_hidden_size = encoder_hidden_size
        
        self.in_channel = encoder_hidden_size
        
        # create continuous convolution and fully-connected layers
        
        convs = []
        denses = []
        
        self.conv_fluid = CtsConv(in_channels = self.in_channel, 
                                  out_channels = self.layer_channels[0],
                                  kernel_sizes = self.kernel_sizes,
                                  radius = self.radius_scale)
        
        self.conv_obstacle = CtsConv(in_channels = 3, 
                                     out_channels = self.layer_channels[0],
                                     kernel_sizes = self.kernel_sizes,
                                     radius = self.radius_scale)
        
        self.dense_fluid = nn.Linear(self.in_channel, self.layer_channels[0])
        
        # concat conv_obstacle, conv_fluid, dense_fluid
        in_ch = 2 * self.layer_channels[0]
        for i in range(1, len(self.layer_channels)):
            out_ch = self.layer_channels[i]
            dense = nn.Linear(in_ch, out_ch)
            denses.append(dense)
            conv = CtsConv(in_channels = in_ch, 
                           out_channels = out_ch,
                           kernel_sizes = self.kernel_sizes,
                           radius = self.radius_scale)
            convs.append(conv)
            in_ch = self.layer_channels[i]
        
        self.convs = nn.ModuleList(convs)
        self.denses = nn.ModuleList(denses)
        
            
    def update_pos_vel(self, p0, v0, a):
        """Apply acceleration and integrate position and velocity.
        Assume the particle has constant acceleration during timestep.
        Return particle's position and velocity after 1 unit timestep."""
        
        dt = self.timestep
        v1 = v0 + dt * a
        p1 = p0 + dt * (v0 + v1) / 2
        return p1, v1

    def apply_correction(self, p0, p1, correction):
        """Apply the position correction
        p0, p1: the position of the particle before/after basic integration. """
        dt = self.timestep
        p_corrected = p1 + correction
        v_corrected = (p_corrected - p0) / dt
        return p_corrected, v_corrected
    
    def dense_forward(self, in_feats, dense_layer):
        flatten_in_feats = in_feats.reshape(
            in_feats.shape[0] * in_feats.shape[1], in_feats.shape[2])
        flatten_output = dense_layer(flatten_in_feats)
        return flatten_output.reshape(in_feats.shape[0], in_feats.shape[1], -1)

    def compute_correction(self, p, v, other_feats, fluid_mask):
        """Precondition: p and v were updated with accerlation"""

        # compute the extent of the filters (the diameter) and the fluid features
        filter_extent = torch.tensor(self.filter_extent)
        fluid_feats = [torch.ones_like(p[:,:, 0:1]), v] 

        if not other_feats is None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, -1)

        # compute the correction by accumulating the output through the network layers
        #print('p', p.shape, 'fluid_feats', fluid_feats.shape, 'mask', fluid_mask.shape)
        output_conv_fluid = self.conv_fluid(p, p, fluid_feats, fluid_mask)
        output_dense_fluid = self.dense_forward(fluid_feats, self.dense_fluid)

        feats = torch.cat((output_conv_fluid, output_dense_fluid), -1)
        # self.outputs = [feats]
        output = feats
        
        for conv, dense in zip(self.convs, self.denses):
            # pass input features to conv and fully-connected layers
            in_feats = F.relu(output)
            output_conv = conv(p, p, in_feats, fluid_mask)
            output_dense = self.dense_forward(in_feats, dense)
            
            # if last dim size of output from cur dense layer is same as last dim size of output
            # current output should be based off on previous output
            if output_dense.shape[-1] == output.shape[-1]:
                output = output_conv + output_dense + output
            else:
                output = output_conv + output_dense
            # self.outputs.append(output)

        # compute the number of fluid particle neighbors.
        # this info is used in the loss function during training.
        # TODO: test this block of code
        self.num_fluid_neighbors = torch.sum(fluid_mask, dim = -1) - 1
    
        # self.last_features = self.outputs[-2]

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * output
        return self.pos_correction
    
    def forward(self, inputs, states=None):
        """ inputs: 8 elems tuple
        p0_enc, v0_enc, p0, v0, a, feats, box, box_feats
        Computes 1 simulation timestep"""

        p0_enc, v0_enc, p0, v0, a, other_feats, fluid_mask = inputs
        other_feats = torch.flatten(other_feats, -2, -1)
        

        if states is None:
            if other_feats is None:
                feats = v0_enc.reshape(*v0_enc.shape[:2], -1)
            else:
                feats = torch.cat((other_feats, v0_enc.reshape(*v0_enc.shape[:2], -1)), -1)
                #print('start',feats.shape)
        else:
            if other_feats is None:
                feats = v0_enc.reshape(*v0_enc.shape[:2], -1)
                feats = torch.cat((states[0][...,3:], feats), -1)
            else:
                feats = torch.cat((other_feats, states[0][...,3:], v0_enc.reshape(*v0_enc.shape[:2], -1)), -1)
                #print('feats',feats.shape)

        # print('p0',p0.shape)
        # print('v0', v0.shape)
        # print('a', a.shape)
        p1, v1 = self.update_pos_vel(p0, v0, a)
        # print('p1',p1.shape)
        # print('v1', v1.shape)
        # print('feats',feats.shape)
        # print('mask', fluid_mask.shape)
        pos_correction = self.compute_correction(p1, v1, feats, fluid_mask)
        pc = torch.cat([pos_correction[...,:2], torch.zeros(*pos_correction.shape[:-1], 1, device=p1.device)], -1)
        p_corrected, v_corrected = self.apply_correction(p0, p1, pc)
        m_matrix = pos_correction[..., 2:].reshape(*pos_correction.shape[:-1], 2,2)

        newfeats = feats[..., other_feats.shape[-1]:]
        #print('newfeats',newfeats.shape)
        return p_corrected, v_corrected, m_matrix, (newfeats, None)



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import pi
import gc
torch.manual_seed(2020)

class CtsConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, radius, normalize_attention=False, 
                 layer_name=None):
        super(CtsConv, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.kernel = torch.nn.parameter.Parameter(
            self.init_kernel(in_channels, out_channels, kernel_sizes), requires_grad=True
        )
        if layer_name is not None:
            self.register_parameter(layer_name, self.kernel)
        self.radius = radius
        self.normalize_attention=normalize_attention
        
    def init_kernel(self, in_channels, out_channels, kernel_sizes):
        kernel = torch.rand(out_channels, in_channels, *kernel_sizes)
        kernel -= 0.5
        k = 1 / torch.sqrt(torch.tensor(in_channels, dtype=torch.float))
        kernel *= 1 * k
        return kernel
    
    def Lambda(self, vec): 
        # Sphere to Grid
        """
        xy = vec[...,0:2] #Spatial Coord
        # Convert to Polar
        r = torch.sqrt(torch.sum(xy ** 2, -1))
        # Stretch Radius
        s = self.stretch(xy[...,0], xy[...,1])
        # Convert to Rectangular
        out = [xy[...,0] * s, xy[...,1] * s, vec[...,2]]
        out = torch.stack(out, -1)
        """
        
        x, y = vec[...,0], vec[...,1]
        x_out, y_out = self.map_polar_sqr(x, y)
        
        out = torch.stack([x_out, y_out, vec[...,2]], axis=-1)
        
        return out
    
    def map_polar_sqr(self, x, y, epsilon=1e-9):

        r = torch.sqrt(x ** 2 + y ** 2 + epsilon)

        cond1 = (x == 0.) & (y == 0.)
        cond2 = (torch.abs(y) <= torch.abs(x)) & (~cond1)
        cond3 = ~(cond1 | cond2)
        
        x_out = torch.zeros(*x.shape, device=self.kernel.device)
        y_out = torch.zeros(*x.shape, device=self.kernel.device)
        
        x_out[cond1] = 0.
        y_out[cond1] = 0.

        x_out[cond2] = torch.sign(x[cond2]) * r[cond2]
        y_out[cond2] = 4 / pi * torch.sign(x[cond2]) * r[cond2] * torch.atan(y[cond2] / x[cond2])

        x_out[cond3] = 4 / pi * torch.sign(y[cond3]) * r[cond3] * torch.atan(x[cond3] / y[cond3])
        y_out[cond3] = torch.sign(y[cond3]) * r[cond3]

        return x_out, y_out
    
    def InterpolateKernelUnit(self, kernel, pos):
        """
        @kernel: [c_out, c_in=feat_dim, x, y, z] -> [batch, C=c_out*c_in, x, y, z]
        @pos: [batch, num, 3] -> [batch, num, 1, 1, 3]
        
        return out: [batch, C=c_out*c_in, num, 1, 1] -> [batch, num, c_out, c_in]
        """
        
        kernels = kernel.reshape(-1, *kernel.shape[2:]).unsqueeze(0)
        kernels = kernels.expand((pos.shape[0], *kernels.shape[1:]))
        grid = pos.unsqueeze(2).unsqueeze(2)
        out = F.grid_sample(kernels, grid, padding_mode='zeros', 
                            mode='bilinear', align_corners=False)
        out = out.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        out = out.reshape(*pos.shape[0:2], *kernel.shape[0:2])
        
        return out
    
    def GetAttention(self, relative_field):
        r = torch.sum(relative_field ** 2, axis=-1)
        return torch.relu((1 - r) ** 3).unsqueeze(-1)
    
    def ContinuousConvUnit(
        self, kernel, field, center, field_feat, 
        field_mask, ctr_feat=None, normalize_attention=False
    ):
        """
        @kernel: [1, feat_dim, depth=3, width=3, height=3]
        @field: [batch, num, pos_dim=3]
        @center: [batch, 1, pos_dim=3]
        @field_feat: [batch, num, c_in=feat_dim]
        @ctr_feat: [batch, 1, feat_dim]
        @field_mask: [batch, num, 1]
        """
        relative_field = (field - center) / self.radius
        
        attention = self.GetAttention(relative_field) * field_mask
        # attention: [batch, num, 1]
        
        psi = torch.sum(attention, axis=1) if normalize_attention else 1
        
        scaled_field = self.Lambda(relative_field)
        
        kernel_on_field = self.InterpolateKernelUnit(kernel, scaled_field)
        # kernel_on_field: [batch, num, c_out, c_in]
        
        out = torch.einsum('bnoi,bni->bo', kernel_on_field, field_feat*attention)
        # out: [batch, c_out]
        
        return out / psi
    
    def InterpolateKernel(self, kernel, pos):
        """
        @kernel_sizes = [kernel_size, kernel_size, kernel_size]
        @kernel: [c_out, c_in=feat_dim, kernel_size, kernel_size, kernel_size] 
                  -> [batch, C=c_out*c_in, x, y, z]
        @pos: [batch, num_m, num_n, 3] -> [batch, num_m, num_n, 1, 3]
        
        return out: [batch, C=c_out*c_in, num_m, num_n, 1] -> [batch, num_m, num_n, c_out, c_in]
        """
        
        kernels = kernel.reshape(-1, *kernel.shape[2:]).unsqueeze(0)
        kernels = kernels.expand((pos.shape[0], *kernels.shape[1:]))
        # kernels: [batch, C=c_out*c_in, x, y, z]
                 
        out = F.grid_sample(kernels, pos.unsqueeze(-2), padding_mode='zeros', 
                            mode='bilinear', align_corners=False)
        del kernels
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # pos.unsqueeze(-2): [batch, num_m, num_n, 1, 3]
        out = out.squeeze(-1).permute(0, 2, 3, 1)
        out = out.reshape(*pos.shape[:-1], *kernel.shape[0:2])
        
        return out
    
    def ContinuousConv(
        self, kernel, field, center, field_feat, 
        field_mask, ctr_feat=None
    ):
        """
        @kernel: [c_out, c_in=feat_dim, kernel_size, kernel_size, kernel_size]
        @field: [batch, num_n, pos_dim=3] -> [batch, 1, num_n, pos_dim]
        @center: [batch, num_m, pos_dim=3] -> [batch, num_m, 1, pos_dim]
        @field_feat: [batch, num_n, c_in=feat_dim] -> [batch, 1, num_n, c_in]
        @ctr_feat: [batch, 1, feat_dim]
        @field_mask: [batch, num_n, 1]
        """
        relative_field = (field.unsqueeze(1) - center.unsqueeze(2)) / self.radius
        # relative_field: [batch, num_m, num_n, pos_dim]
        
        attention = self.GetAttention(relative_field) * field_mask.unsqueeze(1).unsqueeze(-1)
        # attention: [batch, num_m, num_n, 1]
        
        psi = torch.sum(attention, axis=2) + 1 if self.normalize_attention else 1
        
        scaled_field = self.Lambda(relative_field)
        # scaled_field: [batch, num_m, num_n, pos_dim]
        del relative_field
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        
        kernel_on_field = self.InterpolateKernel(kernel, scaled_field)
        # kernel_on_field: [batch, num_m, num_n, c_out, c_in]
        
        #print('kernel', kernel_on_field.shape)
        #print('feat', (field_feat.unsqueeze(1)*attention).shape)
        out = torch.einsum('bmnoi,bbmni->bmo', kernel_on_field, field_feat.unsqueeze(1)*attention)
        # unsqueezed_feat: [batch, 1, num_n, c_in]
        # out: [batch, num_m, c_out]
        
        return out / psi
    
    def forward(
        self, field, center, field_feat, 
        field_mask, ctr_feat=None
    ):
        out = self.ContinuousConv(
            self.kernel, field, center, field_feat, field_mask, ctr_feat
        )
        return out
        
    def extra_repr(self):
        return 'input_channels={}, output_channels={}, kernel_size={}'.format(
            self.in_channels, self.out_channels, self.kernel_sizes
        )


class RelCtsConv(CtsConv):
    def __init__(self, in_channels, out_channels, kernel_sizes, radius, normalize_attention=False, 
                 layer_name=None):
        super(RelCtsConv, self).__init__(in_channels, out_channels, kernel_sizes, 
                                         radius, normalize_attention, layer_name)
        
    def ContinuousConv(
        self, kernel, field, center, field_feat, 
        field_mask, ctr_feat=None
    ):
        """
        @kernel: [c_out, c_in=feat_dim, kernel_size, kernel_size, kernel_size]
        @field: [batch, num_n, pos_dim=3] -> [batch, 1, num_n, pos_dim]
        @center: [batch, num_m, pos_dim=3] -> [batch, num_m, 1, pos_dim]
        @field_feat: [batch, num_n, c_in=feat_dim] -> [batch, 1, num_n, c_in]
        @ctr_feat: [batch, 1, feat_dim]
        @field_mask: [batch, num_n, 1]
        """
        
        relative_field = (field.unsqueeze(1) - center.unsqueeze(2)) / self.radius
        # relative_field: [batch, num_m, num_n, pos_dim]
        
        attention = self.GetAttention(relative_field) * field_mask.unsqueeze(1).unsqueeze(-1)
        # attention: [batch, num_m, num_n, 1]
        
        psi = torch.sum(attention, axis=2) + 1 if self.normalize_attention else 1
        
        scaled_field = self.Lambda(relative_field)
        # scaled_field: [batch, num_m, num_n, pos_dim]
        del relative_field
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        
        kernel_on_field = self.InterpolateKernel(kernel, scaled_field)
        # kernel_on_field: [batch, num_m, num_n, c_out, c_in]
        
        field_feat = field_feat.unsqueeze(1) - ctr_feat.unsqueeze(2)

        out = torch.einsum('bmnoi,bmni->bmo', kernel_on_field, field_feat*attention)
        # unsqueezed_feat: [batch, 1, num_n, c_in]
        # out: [batch, num_m, c_out]
        
        return out / psi
    
    
    def forward(
        self, field, center, field_feat, 
        field_mask, ctr_feat
    ):
        field_feat
        out = self.ContinuousConv(
            self.kernel, field, center, field_feat, field_mask, ctr_feat
        )
        return out
        
    def extra_repr(self):
        return 'input_channels={}, output_channels={}, kernel_size={}'.format(
            self.in_channels, self.out_channels, self.kernel_sizes
        )