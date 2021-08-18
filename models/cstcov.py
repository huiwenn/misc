import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argoverse
import open3d.ml.torch as ml3d
import sys
import os
sys.path.append(os.path.dirname(__file__))

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CstcovModel(nn.Module):
    def __init__(self, 
            kernel_size=[4, 4, 4],
            radius_scale=40,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,
            timestep= 0.1,
            encoder_hidden_size = 19, 
            layer_channels = [32, 64, 64, 64,3]
            ):
        super().__init__()
        
        # init parameters
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.timestep = timestep

        self.radius_scale = radius_scale
        self.layer_channels = layer_channels
        # self.filter_extent = np.float32(self.radius_scale * 6 * self.particle_radius)
        
        self.encoder_hidden_size = encoder_hidden_size
        self.in_channel = self.encoder_hidden_size
        self.activation = F.relu
        # self.relu_shift = torch.nn.parameter.Parameter(torch.tensor(0.2))
        relu_shift = torch.tensor(0.2)
        self.register_buffer('relu_shift', relu_shift)
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)

        # create continuous convolution and fully-connected layers

        self._all_convs = []

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv

            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = conv_fn(kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)

            self._all_convs.append((name, conv))
            return conv

        self.conv_fluid = Conv(name="conv0_fluid",
                                in_channels=self.in_channel,
                                filters=self.layer_channels[0],
                                activation=None)
        self.conv_obstacle = Conv(name="conv0_obstacle",
                                   in_channels=1,
                                   filters=self.layer_channels[0],
                                   activation=None)
        self.dense_fluid = torch.nn.Linear(in_features=self.in_channel,
                                            out_features=self.layer_channels[0])
        torch.nn.init.xavier_uniform_(self.dense_fluid.weight)
        torch.nn.init.zeros_(self.dense_fluid.bias)
        
        self.convs = []
        self.denses = []
        # concat conv_obstacle, conv_fluid, dense_fluid
        in_ch = 3 * self.layer_channels[0] 
        for i in range(1, len(self.layer_channels)):
            out_ch = self.layer_channels[i]
            dense = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense.weight)
            torch.nn.init.zeros_(dense.bias)
            setattr(self, 'dense{0}'.format(i), dense)
            conv = Conv(name='conv{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None)
            setattr(self, 'conv{0}'.format(i), conv)
            self.denses.append(dense)
            self.convs.append(conv)
            in_ch = self.layer_channels[i]
        
        self.convs = nn.ModuleList(self.convs)
        self.denses = nn.ModuleList(self.denses)
        
            
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

    def compute_correction(self, p, v, other_feats, box, box_feats, fluid_mask, box_mask):
        """Precondition: p and v were updated with accerlation"""

        fluid_feats = [v.unsqueeze(-2)]
        
        if not other_feats is None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, -2)
        
        # compute the correction by accumulating the output through the network layers
        output_conv_fluid = self.conv_fluid(fluid_feats, p, p, self.filter_extent)
        output_dense_fluid = self.dense_fluid(fluid_feats)

        output_conv_obstacle = self.conv_obstacle(box, p, box_feats.unsqueeze(-2), box_mask)
        
        feats = torch.cat((output_conv_obstacle, output_conv_fluid, output_dense_fluid), -2)
        # self.outputs = [feats]
        output = feats
        
        for conv, dense in zip(self.convs, self.denses):
            # pass input features to conv and fully-connected layers
            # mags = (torch.sum(output**2,axis=-1) + 1e-6).unsqueeze(-1)
            # in_feats = output/mags * self.activation(mags - self.relu_shift)
            in_feats = self.activation(output)
            # in_feats = output
            output_conv = conv(in_feats, p, p, self.filter_extent)
            output_dense = dense(in_feats)
            
            # if last dim size of output from cur dense layer is same as last dim size of output
            # current output should be based off on previous output
            if output_dense.shape[-2] == output.shape[-2]:
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
        # scale in pecco is (1.0 / 128) 
        self.pos_correction = (1.0 / 4) * output
        self.pos_correction[...,0,:] = (1.0 / 16) * self.pos_correction[...,0,:]
        return self.pos_correction
    
    def forward(self, inputs, states=None):
        """ inputs: 8 elems tuple
        p0_enc, v0_enc, p0, v0, a, feats, box, box_feats
        v0_enc: [batch, num_part, timestamps, 2]
        Computes 1 simulation timestep"""
        p0_enc, v0_enc, p0, v0, a, other_feats, box, box_feats, fluid_mask, box_mask = inputs
        
        if states is None:
            if other_feats is None:
                feats = v0_enc
            else:
                feats = torch.cat((other_feats, v0_enc), -2)
        else:
            if other_feats is None:
                feats = v0_enc
                feats = torch.cat((states[0][...,1:,:], feats), -2)
            else:
                feats = torch.cat((other_feats, states[0][...,1:,:], v0_enc), -2)
        
        # a = (v0 - v0_enc[...,-1,:]) / self.timestep
        p1, v1 = self.update_pos_vel(p0, v0, a)
        
        pos_correction = self.compute_correction(p1, v1, feats, box, box_feats, fluid_mask, box_mask)

        # the 1st output channel is correction
        # pos_correction.squeeze(-2))
        p_corrected, v_corrected = self.apply_correction(p0, p1, pos_correction[..., 0, :])
        #print('pos_correction', pos_correction)
        #print('v0', v0[...,:2,:])
        #print('pv corrected', p_corrected[...,:2,:], v_corrected[...,:2,:])

        m_matrix = pos_correction[..., 1:, :]

        # return output channels after the first one
        return p_corrected, v_corrected, m_matrix, (feats[..., other_feats.shape[-2]:,:], None)


