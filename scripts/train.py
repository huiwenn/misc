#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.append('..')
from collections import namedtuple
import time
import pickle
import argparse
import warnings
warnings.simplefilter("ignore")
from evaluate_network import evaluate
from argoverse.map_representation.map_api import ArgoverseMap
from datasets.argoverse_lane_loader import read_pkl_data
from train_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.utils.backcompat.Warning.enabled = False


parser = argparse.ArgumentParser(description="Training setting and hyperparameters")
parser.add_argument('--cuda_visible_devices', default='1,2')
parser.add_argument('--dataset_path', default='/home/leo/particle/argoverse/argoverse_forecasting', 
                    help='path to dataset folder, which contains train and val folders')
parser.add_argument('--train_window', default=10, type=int, help='how many timestamps to iterate in training')
parser.add_argument('--batch_divide', default=1, type=int, 
                    help='divide one batch into several packs, and train them iterativelly.')
parser.add_argument('--epochs', default=70, type=int)
parser.add_argument('--batches_per_epoch', default=600, type=int, 
                    help='determine the number of batches to train in one epoch')
parser.add_argument('--base_lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--model_name', default='multi_mode_p_ecco', type=str)
parser.add_argument('--val_batches', default=50, type=int, 
                    help='the number of batches of data to split as validation set')
parser.add_argument('--val_batch_size', default=16, type=int)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--evaluation', default=False, action='store_true')

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--rho1', dest='representation', action='store_false')
feature_parser.add_argument('--rho-reg', dest='representation', action='store_true')
parser.set_defaults(representation=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

model_name = args.model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_path = os.path.join(args.dataset_path, 'val', 'lane_data')
train_path = os.path.join(args.dataset_path, 'train', 'lane_data')
    
def create_model():
    from models.MultiModePECCO import MultiModePECCO
    model = MultiModePECCO(num_radii = 3, 
                           num_theta = 16, 
                           reg_dim = 8,
                           radius_scale = 40,
                           timestep = 0.1,
                           in_channel = 19,
                           map_hidden = 8, 
                           encoder_channels = [8, 16, 16],
                           decoder_channels = [8, 3], 
                           predict_window = 10, 
                           modes = 6)
    return model

class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def train():
    am = ArgoverseMap()

    val_dataset = read_pkl_data(val_path, batch_size=args.val_batch_size, shuffle=True, repeat=False, max_lane_nodes=700)

    dataset = read_pkl_data(train_path, batch_size=args.batch_size // args.batch_divide, 
                            repeat=True, shuffle=True, max_lane_nodes=650, max_car=30)

    data_iter = iter(dataset)   
    
    model = create_model().to(device)
    # model = torch.load(model_name + '.pth').to(device)
    # model = model_
    model = MyDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), args.base_lr,betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 10, gamma=0.8)
    
    def train_one_batch(model, batch, train_window=2):

        batch_size = args.batch_size
        
        inputs = ([
            batch['pos_2s'], batch['vel_2s'], 
            batch['pos0'], batch['vel0'], 
            batch['lane'], batch['lane_norm'], 
            batch['car_mask'], batch['lane_mask']
        ])

        traj_preds, mode_pred = model(inputs)
        loss = nll_loss_multimodes(traj_preds, batch_tensor, mode_pred)

        return loss
    
    epochs = args.epochs
    batches_per_epoch = args.batches_per_epoch   # batchs_per_epoch.  Dataset is too large to run whole data. 
    data_load_times = []  # Per batch  
    train_losses = []
    valid_losses = []
    valid_metrics_list = []
    min_loss = None

    for i in range(epochs):
        epoch_start_time = time.time()

        model.train()
        model.reset_predict_window(args.train_window)
        epoch_train_loss = 0 
        sub_idx = 0

        print("training ... epoch " + str(i + 1), end='', flush=True)
        for batch_itr in range(batches_per_epoch * args.batch_divide):

            data_fetch_start = time.time()
            batch = next(data_iter)

            if sub_idx == 0:
                optimizer.zero_grad()
                if (batch_itr // args.batch_divide) % 2 == 0:
                    print("... batch " + str((batch_itr // args.batch_divide) + 1), end='', flush=True)
            sub_idx += 1
            
            batch_size = len(batch['pos0'])

            batch_tensor = {}
            convert_keys = (['pos' + str(i) for i in range(args.train_window + 1)] + 
                            ['vel' + str(i) for i in range(args.train_window + 1)] + 
                            ['pos_2s', 'vel_2s', 'lane', 'lane_norm'])

            for k in convert_keys:
                batch_tensor[k] = torch.tensor(np.stack(batch[k]), dtype=torch.float32, device=device)

            for k in ['car_mask', 'lane_mask']:
                batch_tensor[k] = torch.tensor(np.stack(batch[k]), dtype=torch.float32, device=device).unsqueeze(-1)

            for k in ['track_id' + str(i) for i in range(31)] + ['city']:
                batch_tensor[k] = batch[k]

            batch_tensor['car_mask'] = batch_tensor['car_mask'].squeeze(-1)
            del batch

            data_fetch_latency = time.time() - data_fetch_start
            data_load_times.append(data_fetch_latency)
            
            current_loss = train_one_batch(model, batch_tensor, train_window=args.train_window)
            if sub_idx < args.batch_divide:
                current_loss.backward(retain_graph=True)
            else:
                current_loss.backward()
                optimizer.step()
                sub_idx = 0
            del batch_tensor

            epoch_train_loss += float(current_loss)
            del current_loss
            clean_cache(device)

            if batch_itr == batches_per_epoch - 1:
                print("... DONE", flush=True)

        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            model.reset_predict_window(30)
            valid_total_loss = evaluate(model.module, val_dataset,
                                                       train_window=args.train_window, 
                                                       max_iter=args.val_batches, device=device, 
                                                       batch_size=args.val_batch_size)

        valid_losses.append(float(valid_total_loss))

        if min_loss is None:
            min_loss = valid_losses[-1]

        if valid_losses[-1] < min_loss:
            print('update weights')
            min_loss = valid_losses[-1] 
            best_model = model
            torch.save(model.module, model_name + ".pth")

        epoch_end_time = time.time()

        print('epoch: {}, train loss: {}, val loss: {}, epoch time: {}, lr: {}, {}'.format(
            i + 1, train_losses[-1], valid_losses[-1], 
            round((epoch_end_time - epoch_start_time) / 60, 5), 
            format(get_lr(optimizer), "5.2e"), model_name
        ))

        scheduler.step()
        

def evaluation():
    am = ArgoverseMap()
    
    val_dataset = read_pkl_data(val_path, batch_size=args.val_batch_size, shuffle=False, repeat=False)
    
    trained_model = torch.load(model_name + '.pth')
    trained_model.eval()
    
    with torch.no_grad():
        valid_total_loss = evaluate(trained_model, val_dataset, am=am, 
                                                   train_window=args.train_window, max_iter=len(val_dataset), 
                                                   device=device, start_iter=args.val_batches, use_lane=args.use_lane)
        
        
if __name__ == '__main__':
    # with torch.autograd.detect_anomaly():
    if args.train:
        train()
    
    if args.evaluation:
        evaluation()
    
    
    