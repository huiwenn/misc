
import sys
sys.path.append('.')
sys.path.append('..')
import os
from collections import namedtuple
import time
import json
import pickle
import argparse
import datetime
from datasets.argoverse_lane_loader import read_pkl_data
from train_utils import *

from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm import tqdm

test_data = np.load('/data/sophiasun/particles_data/loc_test_springs5.npy')
train_data = np.load('/data/sophiasun/particles_data/loc_train_springs5.npy')
valid_data = np.load('/data/sophiasun/particles_data/loc_valid_springs5.npy')

def make_batch(feed, device='cuda', h=30):
    batch = {}
    
    data, output = feed
    # [batch, num_part, timestamps, 2]
    data = data.reshape((data.shape[0], 5, h, 2))
    output = output.reshape((data.shape[0], 5, 19, 2))

    data.to(device)
    output.to(device)

    batch['pos_2s'] = data[...,:h-1,:]
    batch['vel_2s'] = data[...,1:,:] - data[...,:h-1,:]
    batch['pos0'] = data[...,h-1,:]
    batch['vel0'] = output[...,0,:]-data[...,h-1,:]
    #print('pos_2s', batch['pos_2s'].shape, 'vel', batch['vel_2s'].shape)
    
    for i in range(19):
        batch['pos'+str(i+1)] = output[:,:,i,:].to(device)
    
    accel = torch.zeros(data.shape[0], 1, 2).to(device)
    batch['accel'] = accel
    batch['car_mask'] = torch.ones(data.shape[0], 5, 1).to(device)
    return batch

def train_one_batch(model, batch, loss_f, train_window=2):

    batch_size =  batch['pos_2s'].shape[0]

    batch['lane'] = torch.zeros(batch_size, 1, 2, device=device)
    batch['lane_norm'] = torch.zeros(batch_size, 1, 2, device=device)
    batch['lane_mask'] = torch.ones(batch_size, 1, 1, device=device)

    m0 = -5*torch.eye(2, device=device).reshape((1,2,2)).repeat((batch_size, 5, 1, 1))
    sigma0 = calc_sigma_edit(m0)
    U = calc_u(sigma0)
    inputs = ([
        batch['pos_2s'], batch['vel_2s'],
        batch['pos0'], batch['vel0'], 
        batch['accel'], U, 
        batch['lane'], batch['lane_norm'], 
        batch['car_mask'], batch['lane_mask']
    ])

    pr_pos1, pr_vel1, pr_m1, states = model(inputs)
    # pr_m1: batch_size x num_vehicles x 2 x 2
    sigma0 = sigma0 + calc_sigma_edit(pr_m1)
    gt_pos1 = batch['pos1']
    #print('pr', pr_pos1.device, 'gt', gt_pos1.device, 'sigma',sigma0.device, 'mask', batch['car_mask'].device)

    losses = loss_f(pr_pos1, gt_pos1, sigma0, batch['car_mask'].squeeze(-1))
    if torch.isnan(losses).any():
        print('bad news here')
        print('first step')
        print('nan in pr_pos1', torch.isnan(pr_pos1).any())
        print('nan in pr_pos1 all', torch.isnan(pr_pos1).all())

        print('nan in pr_m1', torch.isnan(pr_m1).any())
        print('nan in pr_m1 all', torch.isnan(pr_m1).all())

        print('nan in states', torch.isnan(states[0]).any())
        for r in range(10):
            print('nan in input '+ str(r), torch.isnan(inputs[r]).any())
        
        for a in model.state_dict().items():
            isn = torch.isnan(a[1]).any()
            if isn:
                print(a[0], 'is nan')
        #print('sigma0', sigma0)
        losses = torch.zeros(1).to(device)
        raise RuntimeError

    del gt_pos1
    pos0 = batch['pos0']
    vel0 = batch['vel0']
    for i in range(train_window-1):
        pos_enc = torch.unsqueeze(pos0, 2)
        vel_enc = torch.unsqueeze(vel0, 2)
        U = calc_u(sigma0)
        inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, batch['accel'],
                  U, 
                  batch['lane'], batch['lane_norm'], 
                  batch['car_mask'], batch['lane_mask'])

        pos0, vel0 = pr_pos1, pr_vel1
        # del pos_enc, vel_enc

        pr_pos1, pr_vel1, pr_m1, states = model(inputs, states)
        gt_pos1 = batch['pos'+str(i+2)]

        sigma0 = sigma0 + calc_sigma_edit(pr_m1)
        step_loss = loss_f(pr_pos1, gt_pos1, sigma0, batch['car_mask'].squeeze(-1))
        
        if torch.isnan(step_loss).any():
            print('bad news here')
            print('i', i)
            print('nan in pr_pos1', torch.isnan(pr_pos1).any())
            print('nan in pr_m1', torch.isnan(pr_m1).any())
            print('nan in states', torch.isnan(states[0]).any())
            print('nan in sigma', torch.isnan(sigma0).any())
            #print('sigma0', sigma0)
            continue

        losses +=step_loss

    total_loss = torch.sum(losses, axis=0) / (train_window)
    #print(total_loss)
    return total_loss


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class LSTMDataset(Dataset):
    def __init__(self, data, obs_len=30, shuffle=True):
   
        wholetraj = data
        self.obs_len= obs_len
        # if need preprocessing here
        normalized = wholetraj

        self.input_data = normalized[:, :self.obs_len, :]
        self.output_data = normalized[:, self.obs_len:, :]
        self.data_size = self.input_data.shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx: int
                    ) -> Tuple[torch.FloatTensor, Any, Dict[str, np.ndarray]]:
        return (
            torch.FloatTensor(self.input_data[idx]),
            torch.FloatTensor(
                self.output_data[idx])
        )

        
def train(model):
    device = 'cuda'
    model.to(device)
    model_name = 'pecco_particles'
    log_dir = "runs/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    
    train_dataset = LSTMDataset(train_data)
    val_dataset = LSTMDataset(valid_data)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)


    loss_f = nll_dyna
    
    base_lr=0.001

    model = MyDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), base_lr,betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.93)

    print('loaded datasets, starting training')

  
    epochs = 20
    #batches_per_epoch = args.batches_per_epoch   # batchs_per_epoch.  Dataset is too large to run whole data. 
    data_load_times = []  # Per batch 
    train_losses = []
    valid_losses = []
    valid_metrics_list = []
    min_loss = None
    train_window = 2

    # first eval
    '''
    model.eval()
    with torch.no_grad():
        print('loading validation dataset')
        val_dataset = read_pkl_data(val_path, batch_size=args.val_batch_size, shuffle=False, repeat=True)
        valid_total_loss, _, result = evaluate(model.module, val_dataset, loss_f, train_window=args.val_window,
                                                max_iter=args.val_batches,
                                                device=device, use_lane=args.use_lane,
                                                batch_size=args.val_batch_size)
        num_samples = 0
        writer.add_scalar('MRS', result['mis'], num_samples)
    '''
        
    for i in range(epochs):
        print("training ... epoch " + str(i + 1))#, end='', flush=True)
        epoch_start_time = time.time()

        model.train()
        epoch_train_loss = 0
        sub_idx = 0

        for i_batch, feed_dict in enumerate(tqdm(train_dataloader)):
            batch_tensor = make_batch(feed_dict, device)

            current_loss = train_one_batch(model, batch_tensor, loss_f, train_window=train_window)

            current_loss.backward()
            optimizer.step()
            
            if i_batch%10==0:
                print('loss', current_loss)
            
            del batch_tensor
            epoch_train_loss += float(current_loss)

            del current_loss
            clean_cache(device)


        epoch_train_loss = epoch_train_loss/782.0
        train_losses.append(epoch_train_loss)
        
        # ------ eval ------
        if False:
            model.eval()
            with torch.no_grad():
                print('loading validation dataset')
                val_dataset = read_pkl_data(val_path, batch_size=args.val_batch_size, shuffle=False, repeat=False)
                valid_total_loss, _, result = evaluate(model.module, val_dataset, loss_f, train_window=args.val_window,
                                                        max_iter=args.val_batches,
                                                        device=device, use_lane=args.use_lane,
                                                        batch_size=args.val_batch_size)
                for k,v in result.items():
                    writer.add_scalar(k, v, i)
                
                num_samples = i * batches_per_epoch * args.batch_size
                writer.add_scalar('MRS', result['mis'], num_samples)


            valid_losses.append(float(valid_total_loss))

            if min_loss is None:
                min_loss = valid_losses[-1]

            if valid_losses[-1] <= min_loss:
                print('update weights')
                min_loss = valid_losses[-1]
                best_model = model
                torch.save(model.module, model_name + ".pth")
            
        # ----- end of eval ------

        epoch_end_time = time.time()

        print('epoch: {}, train loss: {}, epoch time: {}, lr: {}, {}'.format(
            i + 1, train_losses[-1],
            round((epoch_end_time - epoch_start_time) / 60, 5),
            format(get_lr(optimizer), "5.2e"), model_name
        ))

        writer.add_scalar("Loss/train", train_losses[-1], i)
        writer.flush()

        scheduler.step()
        #profiler.step()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    
    from models.rho_reg_ECCO_corrected import ECCONetwork

    model = ECCONetwork(radius_scale = 40,
                        layer_channels = [8, 16, 16, 16, 3], #[8, 24, 24, 24, 3], #[16, 32, 32, 32, 3], 
                        encoder_hidden_size=32,
                        correction_scale=36)

    print('made model. loading dataset')
    #with torch.autograd.detect_anomaly():
    train(model)

    