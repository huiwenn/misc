#!/usr/bin/env python3
import os
import numpy as np
import sys
sys.path.append('..')
sys.path.append('.')
from collections import namedtuple
from glob import glob
import time
import gc
import pickle
import argparse

from train_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.pedestrian_pkl_loader import read_pkl_data


parser = argparse.ArgumentParser(description="Training setting and hyperparameters")
parser.add_argument('--cuda_visible_devices', default='0,1,2,3')
parser.add_argument('--dataset_path', default='/path/to/argoverse_forecasting/', 
                    help='path to dataset folder, which contains train and val folders')
parser.add_argument('--train_window', default=6, type=int, help='how many timestamps to iterate in training')
parser.add_argument('--batch_divide', default=4, type=int, 
                    help='divide one batch into several packs, and train them iterativelly.')
parser.add_argument('--epochs', default=70, type=int)
parser.add_argument('--batches_per_epoch', default=300, type=int, 
                    help='determine the number of batches to train in one epoch')
parser.add_argument('--base_lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--model_name', default='ecco_ped', type=str)
parser.add_argument('--val_batches', default=60, type=int, 
                    help='the number of batches of data to split as validation set')
parser.add_argument('--val_batch_size', default=4, type=int)
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

val_path = os.path.join(args.dataset_path, 'val')#, 'lane_data')
train_path = os.path.join(args.dataset_path, 'train') #, 'lane_data')
    

def create_model():
    from models.pedestrian_reg_equi_det import ParticlesNetwork
    """Returns an instance of the network for training and evaluation"""
    model = ParticlesNetwork(radius_scale = 6)
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

def main():

    val_dataset = read_pkl_data(val_path, batch_size=4, shuffle=False, repeat=False)

    dataset = read_pkl_data(train_path, batch_size=args.batch_size, repeat=True, 
                            shuffle=True)

    data_iter = iter(dataset)   
    
    model_ = create_model().to(device)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # model_ = torch.load('weights/' + model_name + ".pth")  
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # model = model_.to(device)
    model = MyDataParallel(model_)
    optimizer = torch.optim.Adam(model.parameters(), args.base_lr,betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.97) #0.9968
    
    def train_one_batch(model, batch, train_window=2):

        batch_size = args.batch_size / args.batch_divide

        inputs = ([
            batch['pos_enc'], batch['vel_enc'], 
            batch['pos0'], batch['vel0'], 
            batch['accel'], None, 
            batch['man_mask']
        ])

        # print_inputs_shape(inputs)
        # print(batch['pos0'])
        pr_pos1, pr_vel1, states = model(inputs)
        gt_pos1 = batch['pos1']
        # print(pr_pos1)

        losses = 0.5 * loss_fn(pr_pos1, gt_pos1, torch.sum(batch['man_mask'], dim = -2) - 1, batch['man_mask'].squeeze(-1))
        del gt_pos1

        # pos_2s = batch['pos_2s']
        # vel_2s = batch['vel_2s']
        pos0 = batch['pos0']
        vel0 = batch['vel0']
        for i in range(train_window-1):
            pos_enc = torch.unsqueeze(pos0, 2)
            # pos_2s = torch.cat([pos_2s[:,:,1:,:], pos_enc], axis=2)
            vel_enc = torch.unsqueeze(vel0, 2)
            # vel_2s = torch.cat([vel_2s[:,:,1:,:], vel_enc], axis=2)
            # del pos_enc, vel_enc
            accel = pr_vel1 - vel_enc[...,-1,:]
            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, accel, None,
                      batch['man_mask'])
            pos0, vel0 = pr_pos1, pr_vel1
            del pos_enc, vel_enc
            
            pr_pos1, pr_vel1, states = model(inputs, states)

            gt_pos1 = batch['pos'+str(i+2)]
            clean_cache(device)

            losses += 0.5 * loss_fn(pr_pos1, gt_pos1,
                               torch.sum(batch['man_mask'], dim = -2) - 1, batch['man_mask'].squeeze(-1))


        total_loss = 128 * torch.sum(losses,axis=0) / batch_size
        return total_loss
    
    epochs = args.epochs
    batches_per_epoch = args.batches_per_epoch   # batchs_per_epoch.  Dataset is too large to run whole data. 
    batch_divide = args.batch_divide
    model_name = args.model_name
    data_load_times = []  #Per batch 
    train_losses = []
    valid_losses = []
    valid_metrics_list = []
    min_loss = None

    for i in range(epochs):
        epoch_start_time = time.time()

        model = model.train()
        epoch_train_loss = 0 
        sub_idx = 0

        print("training ... epoch " + str(i + 1), end='')
        for batch_itr in range(batches_per_epoch*batch_divide):

            data_fetch_start = time.time()
            batch = next(data_iter)
            
            if sub_idx == 0:
                optimizer.zero_grad()
                if (batch_itr // batch_divide) % 25 == 0:
                    print("... batch " + str((batch_itr // batch_divide) + 1), end='', flush=True)
            sub_idx += 1

            batch_size = len(batch['pos0'])

            batch_tensor = process_batch_ped_2d(batch, device, train_window=args.train_window)
            del batch

            data_fetch_latency = time.time() - data_fetch_start
            data_load_times.append(data_fetch_latency)

            current_loss = train_one_batch(model, batch_tensor, train_window=args.train_window)
            if sub_idx < batch_divide:
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

        model = model.eval()
        with torch.no_grad():
            valid_total_loss, valid_metrics = evaluate(model.module, val_dataset, 
                                                       train_window=12, max_iter=args.val_batches, 
                                                       device=device, batch_size=args.val_batch_size)

        valid_losses.append(float(valid_total_loss))
        valid_metrics_list.append(valid_metrics)
        
        # torch.save(model.module, os.path.join(checkpoint_path, model_name + '_' + str(i) + ".pth"))

        if min_loss is None:
            min_loss = valid_losses[-1]

        if valid_losses[-1] < min_loss:
            min_loss = valid_losses[-1] 
            best_model = model
            torch.save(model.module, model_name + ".pth")

            #Add evaluation Metrics

        epoch_end_time = time.time()

        print('epoch: {}, train loss: {}, val loss: {}, epoch time: {}, lr: {}, {}'.format(
            i + 1, train_losses[-1], valid_losses[-1], 
            round((epoch_end_time - epoch_start_time) / 60, 5), 
            format(get_lr(optimizer), "5.2e"), model_name
        ))

        scheduler.step()
    
        with open('results/{}_val_metrics.pickle'.format(model_name), 'wb') as f:
            pickle.dump(valid_metrics_list, f)
        

def final_evaluation():
    
    test_dataset = read_pkl_data(val_path, batch_size=4, shuffle=False, repeat=False)
    
    trained_model = torch.load('weights/' + model_name + '.pth')
    trained_model.eval()
    
    with torch.no_grad():
        valid_total_loss, valid_metrics = evaluate(trained_model, test_dataset, 
                                                   train_window=train_window, max_iter=len(test_dataset), 
                                                   device=device, start_iter=240)
    
    with open('results/{}_predictions.pickle'.format(model_name), 'wb') as f:
        pickle.dump(valid_metrics, f)


def evaluate(model, val_dataset, use_lane=False,
             train_window=3, max_iter=2500, device='cpu', start_iter=0, 
             batch_size=32, use_normalize_input=False, normalize_scale=3):
    
    print('evaluating.. ', end='', flush=True)

        
    count = 0
    prediction_gt = {}
    losses = []
    val_iter = iter(val_dataset)
    
    for i, sample in enumerate(val_dataset):
        
        if i >= max_iter:
            break
        
        if i < start_iter:
            continue
        
        pred = []
        gt = []

        if count % 10 == 0:
            print('{}'.format(count + 1), end=' ', flush=True)
        
        count += 1
        
        data = {}
        convert_keys = (['pos' + str(i) for i in range(13)] + 
                        ['vel' + str(i) for i in range(13)] + 
                        ['pos_enc', 'vel_enc'])

        for k in convert_keys:
            data[k] = torch.tensor(np.stack(sample[k])[...,:2], dtype=torch.float32, device=device)
            
        for k in ['man_mask']:
            data[k] = torch.tensor(np.stack(sample[k]), dtype=torch.float32, device=device).unsqueeze(-1)
            
        data['scene_idx'] = np.stack(sample['scene_idx'])
        scenes = data['scene_idx'].tolist()
        
        data['man_mask'] = data['man_mask']
        
        accel = data['vel0'] - data['vel_enc'][...,-1,:]
        # accel = torch.zeros(1, 1, 2).to(device)
        data['accel'] = accel
        
        inputs = ([
            data['pos_enc'], data['vel_enc'], 
            data['pos0'], data['vel0'], 
            data['accel'], None, 
            data['man_mask']
        ])

        pr_pos1, pr_vel1, states = model(inputs)
        gt_pos1 = data['pos1']

        # l = 0.5 * loss_fn(pr_pos1, gt_pos1, model.num_fluid_neighbors.unsqueeze(-1), data['car_mask'])
        l = 0.5 * loss_fn(pr_pos1, gt_pos1, torch.sum(data['man_mask'], dim = -2) - 1, data['man_mask'].squeeze(-1))

        pr_agent, gt_agent = pr_pos1[:,0], gt_pos1[:,0]
        # print(pr_agent, gt_agent)

   
        pred.append(pr_agent.unsqueeze(1).detach().cpu())
        gt.append(gt_agent.unsqueeze(1).detach().cpu())
        del pr_agent, gt_agent
        clean_cache(device)

        pos_2s = data['pos_enc']
        vel_2s = data['vel_enc']
        pos0 = data['pos0']
        vel0 = data['vel0']
        for i in range(11):
            pos_enc = torch.unsqueeze(pos0, 2)
            # pos_2s = torch.cat([pos_2s[:,:,1:,:], pos_enc], axis=2)
            vel_enc = torch.unsqueeze(vel0, 2)
            # vel_2s = torch.cat([vel_2s[:,:,1:,:], vel_enc], axis=2)
            accel = pr_vel1 - vel_enc[...,-1,:]
            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, accel, None, 
                      data['man_mask'])
            pos0, vel0 = pr_pos1, pr_vel1
            pr_pos1, pr_vel1, states = model(inputs, states)
            clean_cache(device)
            
            if i < train_window - 1:
                gt_pos1 = data['pos'+str(i+2)]
                l += 0.5 * loss_fn(pr_pos1, gt_pos1, torch.sum(data['man_mask'], dim = -2) - 1, data['man_mask'].squeeze(-1))

            pr_agent, gt_agent = pr_pos1[:,0], data['pos'+str(i+2)][:,0]
            # print(pr_agent, gt_agent)

            pred.append(pr_agent.unsqueeze(1).detach().cpu())
            gt.append(gt_agent.unsqueeze(1).detach().cpu())
            losses.append(l)
            clean_cache(device)

        predict_result = (torch.cat(pred, axis=1), torch.cat(gt, axis=1))
        for idx, scene_id in enumerate(scenes):
            prediction_gt[scene_id] = (predict_result[0][idx], predict_result[1][idx])


    total_loss = 128 * torch.sum(torch.stack(losses),axis=0) / max_iter
    
    result = {}
    de = {}
    for k, v in prediction_gt.items():
        de[k] = torch.sqrt((v[0][:,0] - v[1][:,0])**2 + 
                        (v[0][:,1] - v[1][:,1])**2)
        
    ade = []
    fde = []
    for k, v in de.items():
        ade.append(np.mean(v.numpy()))
        fde.append(v.numpy()[-1])
    
    result['ADE'] = np.mean(ade)
    result['ADE_std'] = np.std(ade)
    result['fde'] = np.mean(fde)
    result['fde_std'] = np.std(fde)

    print(result)
    print('done')

    return total_loss, prediction_gt

        
if __name__ == '__main__':
    main()
    
    #final_evaluation()
    
    
    