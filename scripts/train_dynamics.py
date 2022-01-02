#!/usr/bin/env python3
print('importing libraries')
import os
import sys
sys.path.append('..')
from collections import namedtuple
import time
import json
import pickle
import argparse
import datetime
from evaluate_dynamics import evaluate
#from argoverse.map_representation.map_api import ArgoverseMap
from datasets.argoverse_lane_loader import read_pkl_data
from train_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#os.environ["NCCL_DEBUG"] = "INFO"

parser = argparse.ArgumentParser(description="Training setting and hyperparameters")
parser.add_argument('--cuda_visible_devices', default='0,1,2,3,4,5,6,7')
parser.add_argument('--dataset_path', default='/path/to/argoverse_forecasting/', 
                    help='path to dataset folder, which contains train and val folders')
parser.add_argument('--train_window', default=6, type=int, help='how many timestamps to iterate in training')
parser.add_argument('--val_window', default=30, type=int, help='how many timestamps to iterate in validation')
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

def create_model():
    if args.representation:
        from models.rho_reg_ECCO import ECCONetwork
        """Returns an instance of the network for training and evaluation"""
        
        model = ECCONetwork(radius_scale = 40,
                            layer_channels = [8, 16, 16, 16, 3],
                            encoder_hidden_size=21)
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

    print('loading train dataset')
    dataset = read_pkl_data(train_path, batch_size=args.batch_size / args.batch_divide,
                            repeat=True, shuffle=True, max_lane_nodes=900)

    data_iter = iter(dataset)

    if args.load_model_path:
        print('loading model from ' + args.load_model_path)
        model_ = torch.load(args.load_model_path + '.pth')
        model = model_
    else:
        model = create_model().to(device)
    
    if args.loss == "ecco":
        loss_f = ecco_loss
    elif args.loss == "mis": 
        loss_f = mis_loss
    else: # args.loss == "nll":
        loss_f = nll_dyna


    model = MyDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), args.base_lr,betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.95)

    print('loaded datasets, starting training')

    def train_one_batch(model, batch, loss_f, train_window=2):

        batch_size = args.batch_size
        if not args.use_lane:
            batch['lane'] = torch.zeros(batch_size, 1, 2, device=device)
            batch['lane_norm'] = torch.zeros(batch_size, 1, 2, device=device)
            batch['lane_mask'] = torch.ones(batch_size, 1, 1, device=device)

        m0 = -5*torch.eye(2, device=device).reshape((1,2,2)).repeat((batch_size//args.batch_divide, 60, 1, 1))
        sigma0 = calc_sigma(m0)

        inputs = ([
            batch['pos_2s'], batch['vel_2s'],
            batch['pos0'], batch['vel0'], 
            batch['accel'], sigma0, #other feats: 2x2 sigma matrices
            batch['lane'], batch['lane_norm'], 
            batch['car_mask'], batch['lane_mask']
        ])
    
        pr_pos1, pr_vel1, pr_m1, states = model(inputs)

        sigma0 = sigma0 + calc_sigma(pr_m1)
        gt_pos1 = batch['pos1']

        losses = loss_f(pr_pos1, gt_pos1, sigma0, batch['car_mask'].squeeze(-1))
        del gt_pos1
        pos0 = batch['pos0']
        vel0 = batch['vel0']
        for i in range(train_window-1):
            pos_enc = torch.unsqueeze(pos0, 2)
            vel_enc = torch.unsqueeze(vel0, 2)
            
            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, batch['accel'],
                      sigma0, 
                      batch['lane'], batch['lane_norm'], 
                      batch['car_mask'], batch['lane_mask'])

            pos0, vel0 = pr_pos1, pr_vel1
            # del pos_enc, vel_enc
            
            pr_pos1, pr_vel1, pr_m1, states = model(inputs, states)
            gt_pos1 = batch['pos'+str(i+2)]
            
            sigma0 = sigma0 + calc_sigma(pr_m1)

            losses += loss_f(pr_pos1, gt_pos1, sigma0, batch['car_mask'].squeeze(-1))

        total_loss = torch.sum(losses, axis=0) / (train_window)
        return total_loss
    
    epochs = args.epochs
    batches_per_epoch = args.batches_per_epoch   # batchs_per_epoch.  Dataset is too large to run whole data. 
    data_load_times = []  # Per batch 
    train_losses = []
    valid_losses = []
    valid_metrics_list = []
    min_loss = None

    '''
    trace = torch.profiler.tensorboard_trace_handler("./profile")
    with torch.profiler.profile(schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        on_trace_ready=trace,
        with_stack=True
    ) as profiler:
    '''
    #---
    for i in range(epochs):
        print("training ... epoch " + str(i + 1))#, end='', flush=True)
        epoch_start_time = time.time()

        model.train()
        epoch_train_loss = 0
        sub_idx = 0

        for batch_itr in range(batches_per_epoch * args.batch_divide):

            data_fetch_start = time.time()
            batch = next(data_iter)

            if sub_idx == 0:
                optimizer.zero_grad()
                if (batch_itr // args.batch_divide) % 10 == 0:
                    print("... batch " + str((batch_itr // args.batch_divide) + 1), end='', flush=True)
            sub_idx += 1

            batch_tensor = process_batch(batch, device, train_window=args.train_window)
            del batch

            data_fetch_latency = time.time() - data_fetch_start
            data_load_times.append(data_fetch_latency)

            current_loss = train_one_batch(model, batch_tensor, loss_f, train_window=args.train_window)

            if sub_idx < args.batch_divide:
                current_loss.backward(retain_graph=True)
            else:
                current_loss.backward()
                optimizer.step()
                sub_idx = 0
            del batch_tensor

            epoch_train_loss += float(current_loss)

            # test todo
            # print('current loss', float(current_loss))

            del current_loss
            clean_cache(device)

            if batch_itr // args.batch_divide == batches_per_epoch - 1:
                print("... DONE")#, flush=True)

        epoch_train_loss = epoch_train_loss/(batches_per_epoch * args.batch_divide)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            print('loading validation dataset')
            val_dataset = read_pkl_data(val_path, batch_size=args.val_batch_size, shuffle=False, repeat=False)
            valid_total_loss, _, _ = evaluate(model.module, val_dataset, loss_f, train_window=args.val_window,
                                                    max_iter=args.val_batches,
                                                    device=device, use_lane=args.use_lane,
                                                    batch_size=args.val_batch_size)


        valid_losses.append(float(valid_total_loss))

        if min_loss is None:
            min_loss = valid_losses[-1]

        if valid_losses[-1] <= min_loss:
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
        #profiler.step()

    #---
    writer.close()

        

def evaluation():
    #am = ArgoverseMap()
    if args.loss == "ecco":
        loss_f = ecco_loss
    elif args.loss == "mis": 
        loss_f = mis_loss
    else: # args.loss == "nll":
        loss_f = nll_dyna

    val_dataset = read_pkl_data(val_path, batch_size=args.val_batch_size, shuffle=False, repeat=False)
    #dataset = read_pkl_data(train_path, batch_size=args.batch_size / args.batch_divide, repeat=True, shuffle=True, max_lane_nodes=900)


    trained_model = torch.load(model_name + '.pth')
    trained_model.eval()
    
    with torch.no_grad():
        # change back to val_dataset
        valid_total_loss, prediction_gt, valid_metrics = evaluate(trained_model, val_dataset, loss_f, train_window=args.val_window,
                                                       max_iter=args.val_batches, 
                                                       device=device, use_lane=args.use_lane, 
                                                       batch_size=args.val_batch_size)
    
    with open('results/{}_predictions.pickle'.format(model_name), 'wb') as f:
        pickle.dump(prediction_gt, f)

    for k,v in valid_metrics.items():
        valid_metrics[k] = v.tolist()
    with open('results/{}_metrics.json'.format(model_name), 'w') as f:
        json.dump(valid_metrics, f)
        
        
if __name__ == '__main__':

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    model_name = args.model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    val_path = os.path.join(args.dataset_path, 'val') #, 'lane_data'
    train_path = os.path.join(args.dataset_path, 'train') #, 'lane_data'

    if args.train:
        # debug 大法好
        # with torch.autograd.detect_anomaly():
        train()
    
    if args.evaluation:
        evaluation()
    
    
    
