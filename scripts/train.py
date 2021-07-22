#!/usr/bin/env python3
import os
import sys
sys.path.append('..')
from collections import namedtuple
import time
import pickle
import argparse
import datetime
from evaluate_network import evaluate
from argoverse.map_representation.map_api import ArgoverseMap
from datasets.argoverse_lane_loader import read_pkl_data
from train_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description="Training setting and hyperparameters")
parser.add_argument('--cuda_visible_devices', default='0,1,2,3')
parser.add_argument('--dataset_path', default='/path/to/argoverse_forecasting/', 
                    help='path to dataset folder, which contains train and val folders')
parser.add_argument('--train_window', default=4, type=int, help='how many timestamps to iterate in training')
parser.add_argument('--batch_divide', default=1, type=int, 
                    help='divide one batch into several packs, and train them iterativelly.')
parser.add_argument('--epochs', default=70, type=int)
parser.add_argument('--batches_per_epoch', default=600, type=int, 
                    help='determine the number of batches to train in one epoch')
parser.add_argument('--base_lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--model_name', default='ecco_trained_model', type=str)
parser.add_argument('--use_lane', default=False, action='store_true')
parser.add_argument('--val_batches', default=600, type=int,
                    help='the number of batches of data to split as validation set')
parser.add_argument('--val_batch_size', default=1, type=int)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--evaluation', default=False, action='store_true')
parser.add_argument('--load_model_path', default='', type=str, help='path to model to be loaded')
parser.add_argument('--loss', default='nll', type=str, help='nll or ecco loss')



feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--rho1', dest='representation', action='store_false')
feature_parser.add_argument('--rho-reg', dest='representation', action='store_true')
parser.set_defaults(representation=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

model_name = args.model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)

val_path = os.path.join(args.dataset_path, 'val') #, 'lane_data'
train_path = os.path.join(args.dataset_path, 'train') #, 'lane_data'
    
def create_model():
    if args.representation:
        from models.rho_reg_ECCO import ECCONetwork
        """Returns an instance of the network for training and evaluation"""
        model = ECCONetwork(radius_scale = 40,
                            layer_channels = [8, 16, 8, 8, 3],
                            encoder_hidden_size=23)
    else:
        from models.rho1_ECCO import ECCONetwork
        """Returns an instance of the network for training and evaluation"""
        model = ECCONetwork(radius_scale = 40, encoder_hidden_size=18,
                            layer_channels = [16, 32, 32, 32, 1], 
                            num_radii = 3)
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
    #am = ArgoverseMap()
    log_dir = "runs/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)


    print('loading tain dataset')
    dataset = read_pkl_data(train_path, batch_size=args.batch_size / args.batch_divide,
                            repeat=True, shuffle=True, max_lane_nodes=900)

    data_iter = iter(dataset)

    if args.load_model_path:
        print('loading model from ' + args.load_model_path)
        model_ = torch.load(args.load_model_path + '.pth')
        model = model_
    else:
        model = create_model().to(device)

    model = MyDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), args.base_lr,betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.95)

    print('loaded datasets, starting training')

    def train_one_batch(model, batch, train_window=2):

        batch_size = args.batch_size
        if not args.use_lane:
            batch['lane'] = torch.zeros(batch_size, 1, 2, device=device)
            batch['lane_norm'] = torch.zeros(batch_size, 1, 2, device=device)
            batch['lane_mask'] = torch.ones(batch_size, 1, 1, device=device)
        
        if args.loss == "ecco":
            loss_f = ecco_loss
        else: # args.loss == "nll":
            loss_f = nll

        inputs = ([
            batch['pos_2s'], batch['vel_2s'],
            batch['pos0'], batch['vel0'], 
            batch['accel'], batch['sigmas'], #other feats: 4x2 two M matrices
            batch['lane'], batch['lane_norm'], 
            batch['car_mask'], batch['lane_mask']
        ])

        pr_pos1, pr_vel1, pr_m1, states = model(inputs)
        gt_pos1 = batch['pos1']

        losses = loss_f(pr_pos1, gt_pos1, pr_m1, batch['car_mask'].squeeze(-1))
        del gt_pos1
        pos0 = batch['pos0']
        vel0 = batch['vel0']
        m0 = torch.zeros((batch_size, 60, 2, 2), device=pos0.device)
        for i in range(train_window-1):
            pos_enc = torch.unsqueeze(pos0, 2)
            vel_enc = torch.unsqueeze(vel0, 2)

            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, batch['accel'],
                      torch.cat([m0, pr_m1], dim=-2), 
                      batch['lane'], batch['lane_norm'], 
                      batch['car_mask'], batch['lane_mask'])
            pos0, vel0, m0 = pr_pos1, pr_vel1, pr_m1
            # del pos_enc, vel_enc
            
            pr_pos1, pr_vel1, pr_m1, states = model(inputs, states)
            gt_pos1 = batch['pos'+str(i+2)]
            
            losses += loss_f(pr_pos1, gt_pos1, pr_m1, batch['car_mask'].squeeze(-1))


        total_loss = torch.sum(losses,axis=0) / (batch_size*train_window)

        return total_loss
    
    epochs = args.epochs
    batches_per_epoch = args.batches_per_epoch   # batchs_per_epoch.  Dataset is too large to run whole data. 
    data_load_times = []  # Per batch 
    train_losses = []
    valid_losses = []
    valid_metrics_list = []
    min_loss = None

    for i in range(epochs):
        print("training ... epoch " + str(i + 1), end='', flush=True)
        epoch_start_time = time.time()

        model.train()
        epoch_train_loss = 0 
        sub_idx = 0

        for batch_itr in range(batches_per_epoch * args.batch_divide):

            data_fetch_start = time.time()
            batch = next(data_iter)

            if sub_idx == 0:
                optimizer.zero_grad()
                if (batch_itr // args.batch_divide) % 25 == 0:
                    print("... batch " + str((batch_itr // args.batch_divide) + 1), end='', flush=True)
            sub_idx += 1
            
            batch_tensor = process_batch(batch, device, train_window=args.train_window)
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
            print('loading validation dataset')
            val_dataset = read_pkl_data(val_path, batch_size=args.val_batch_size, shuffle=False, repeat=False)
            valid_total_loss, _ = evaluate(model.module, val_dataset, 
                                                       max_iter=args.val_batches, 
                                                       device=device, use_lane=args.use_lane, 
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

        writer.add_scalar("Loss/train", train_losses[-1], i)
        writer.add_scalar("Loss/validation", valid_losses[-1], i)
        writer.flush()

        scheduler.step()

    writer.close()

        

def evaluation():
    am = ArgoverseMap()
    
    val_dataset = read_pkl_data(val_path, batch_size=args.val_batch_size, shuffle=False, repeat=False)
    
    trained_model = torch.load(model_name + '.pth')
    trained_model.eval()
    
    with torch.no_grad():
        valid_total_loss, valid_metrics = evaluate(trained_model, val_dataset, am=am, 
                                                   train_window=args.train_window, max_iter=len(val_dataset), 
                                                   device=device, start_iter=args.val_batches, use_lane=args.use_lane)
    
    with open('results/{}_predictions.pickle'.format(model_name), 'wb') as f:
        pickle.dump(valid_metrics, f)
        
        
if __name__ == '__main__':
    if args.train:
        # debug 大法好
        # with torch.autograd.detect_anomaly():
        train()
    
    if args.evaluation:
        evaluation()
    
    
    