#!/usr/bin/env python3
import os
import numpy as np
import sys
sys.path.append('..')
from datasets.helper import get_lane_direction
from collections import namedtuple
from glob import glob
import time
import gc
import pickle
from utils.deeplearningutilities.tf import Trainer, MyCheckpointManager
from evaluate_network_equivariant import evaluate
from argoverse.map_representation.map_api import ArgoverseMap
from train_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


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
args = parser.parse_args()



use_lane = args.use_lane
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))


val_path = os.path.join(args.dataset_path, 'val') #, 'lane_data'
train_path = os.path.join(args.dataset_path, 'train') #, 'lane_data'
    

# dummy variable
use_normalize_input = False
normalize_scale = 3



def create_model():
    from models.reg_equivariant_model import ParticlesNetwork
    """Returns an instance of the network for training and evaluation"""
    model = ParticlesNetwork(radius_scale = 40, layer_channels = [8, 16, 8, 8, 1], encoder_hidden_size=18)
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
    am = ArgoverseMap()

    val_dataset = read_pkl_data(val_path, batch_size=8, shuffle=False, repeat=False)

    dataset = read_pkl_data(train_path, batch_size=args.batch_size, repeat=True, shuffle=True)

    data_iter = iter(dataset)   
    
    model_ = create_model()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # model_ = torch.load(model_name + ".pth") 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # model = model_
    model = MyDataParallel(model_).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.base_lr,betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9968)
    
    def train_one_batch(model, batch, train_window=2):

        batch_size = args.batch_size

        inputs = ([
            batch['pos_2s'], batch['vel_2s'], 
            batch['pos0'], batch['vel0'], 
            batch['accel'], None,
            batch['lane'], batch['lane_norm'], 
            batch['car_mask'], batch['lane_mask']
        ])

        # print_inputs_shape(inputs)
        # print(batch['pos0'])
        pr_pos1, pr_vel1, states = model(inputs)
        gt_pos1 = batch['pos1']
        # print(pr_pos1)

        # losses = 0.5 * loss_fn(pr_pos1, gt_pos1, model.num_fluid_neighbors.unsqueeze(-1), batch['car_mask'])
        losses = 0.5 * loss_fn(pr_pos1, gt_pos1, torch.sum(batch['car_mask'], dim = -2) - 1, batch['car_mask'].squeeze(-1))
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
            if teacher_forcing:
                inputs = (pos_enc, vel_enc, 
                          batch['pos'+str(i+1)], batch['vel'+str(i+1)], 
                          batch['accel'], None, batch['lane'],
                          batch['lane_norm'],batch['car_mask'], batch['lane_mask'])
                pos0, vel0 = batch['pos'+str(i+1)], batch['vel'+str(i+1)]
            else:
                inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, batch['accel'], None,
                          batch['lane'],
                          batch['lane_norm'],batch['car_mask'], batch['lane_mask'])
                pos0, vel0 = pr_pos1, pr_vel1
            # del pos_enc, vel_enc
            
            pr_pos1, pr_vel1, states = model(inputs, states)

            gt_pos1 = batch['pos'+str(i+2)]
            # clean_cache(device)

            # losses += 0.5 * loss_fn(pr_pos1, gt_pos1,
            #                    model.num_fluid_neighbors.unsqueeze(-1), batch['car_mask'])

            losses += 0.5 * loss_fn(pr_pos1, gt_pos1,
                               torch.sum(batch['car_mask'], dim = -2) - 1, batch['car_mask'].squeeze(-1))


            # pr_pos1, pr_vel1 = pr_pos2, pr_vel2
            # print(pr_pos1)


        total_loss = 128 * torch.sum(losses,axis=0) / batch_size


        return total_loss
    
    epochs = args.epochs
    batches_per_epoch = args.batches_per_epoch   # batchs_per_epoch.  Dataset is too large to run whole data. 
    data_load_times = []  #Per batch 
    train_losses = []
    valid_losses = []
    valid_metrics_list = []
    min_loss = None

    for i in range(epochs):
        epoch_start_time = time.time()

        model.train()
        epoch_train_loss = 0 
        sub_idx = 0

        print("training ... epoch " + str(i + 1), end='')
        for batch_itr in range(batches_per_epoch):
            
            data_fetch_start = time.time()
            batch = next(data_iter)
            
            if sub_idx == 0:
                optimizer.zero_grad()
                if (batch_itr // args.batch_divide) % 10 == 0:
                    print("... batch " + str((batch_itr // args.batch_divide) + 1), end='', flush=True)
            sub_idx += 1

            batch_size = len(batch['pos0'])

            if use_lane:
                pass
            else:
                batch['lane_mask'] = [np.array([0])] * batch_size

            batch_tensor = {}
            convert_keys = (['pos' + str(i) for i in range(train_window + 1)] + 
                            ['vel' + str(i) for i in range(train_window + 1)] + 
                            ['pos_2s', 'vel_2s', 'lane', 'lane_norm'])

            for k in convert_keys:
                batch_tensor[k] = torch.tensor(np.stack(batch[k])[...,:2], dtype=torch.float32, device=device)
                
            if use_normalize_input:
                batch_tensor, max_pos = normalize_input(batch_tensor, normalize_scale, train_window)

            for k in ['car_mask', 'lane_mask']:
                batch_tensor[k] = torch.tensor(np.stack(batch[k]), dtype=torch.float32, device=device).unsqueeze(-1)

            for k in ['track_id' + str(i) for i in range(30)] + ['city']:
                batch_tensor[k] = batch[k]

            batch_tensor['car_mask'] = batch_tensor['car_mask'].squeeze(-1)
            accel = torch.zeros(batch_size, 1, 2).to(device)
            batch_tensor['accel'] = accel
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
            valid_total_loss, valid_metrics = evaluate(model.module, val_dataset, am=am, 
                                                       train_window=args.train_window, max_iter=50, 
                                                       device=device, use_lane=use_lane,
                                                       use_normalize_input=use_normalize_input, 
                                                       normalize_scale=normalize_scale, 
                                                       batch_size=val_dataset.batch_size)

        valid_losses.append(float(valid_total_loss))
        valid_metrics_list.append(valid_metrics)
        
        # torch.save(model.module, os.path.join(checkpoint_path, model_name + '_' + str(i) + ".pth"))

        if min_loss is None:
            min_loss = valid_losses[-1]

        if valid_losses[-1] < min_loss:
            min_loss = valid_losses[-1] 
            best_model = model
            torch.save(model.module, args.model_name + ".pth")

            #Add evaluation Metrics

        epoch_end_time = time.time()

        print('epoch: {}, train loss: {}, val loss: {}, epoch time: {}, lr: {}, {}'.format(
            i + 1, train_losses[-1], valid_losses[-1], 
            round((epoch_end_time - epoch_start_time) / 60, 5), 
            format(get_lr(optimizer), "5.2e"), args.model_name
        ))

        scheduler.step()
        
        with open('results/{}_val_metrics.pickle'.format(args.model_name), 'wb') as f:
            pickle.dump(valid_metrics_list, f)
        



def evaluate(model, val_dataset, fluid_errors=None, am=None, use_lane=False,
             train_window=3, max_iter=2500, device='cpu', start_iter=0, 
             batch_size=32, use_normalize_input=False, normalize_scale=3):
    
    print('evaluating.. ', end='', flush=True)

    if fluid_errors is None:
        fluid_errors = TrafficErrors()
    
    if am is None:
        from argoverse.map_representation.map_api import ArgoverseMap
        am = ArgoverseMap()
        
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

        if count % 1 == 0:
            print('{}'.format(count + 1), end=' ', flush=True)
        
        count += 1
        
        if use_lane:
            pass
        else:
            sample['lane_mask'] = [np.array([0])] * batch_size
        
        data = {}
        convert_keys = (['pos' + str(i) for i in range(31)] + 
                        ['vel' + str(i) for i in range(31)] + 
                        ['pos_2s', 'vel_2s', 'lane', 'lane_norm'])

        for k in convert_keys:
            data[k] = torch.tensor(np.stack(sample[k])[...,:2], dtype=torch.float32, device=device)
        
        if use_normalize_input:
            data, max_pos = normalize_input(data, normalize_scale, 29)

        for k in ['track_id' + str(i) for i in range(31)] + ['city', 'agent_id', 'scene_idx']:
            data[k] = np.stack(sample[k])
        
        for k in ['car_mask', 'lane_mask']:
            data[k] = torch.tensor(np.stack(sample[k]), dtype=torch.float32, device=device).unsqueeze(-1)
            
        scenes = data['scene_idx'].tolist()
            
        data['agent_id'] = data['agent_id'][:,np.newaxis]
        
        data['car_mask'] = data['car_mask'].squeeze(-1)
        accel = torch.zeros(1, 1, 2).to(device)
        data['accel'] = accel

        lane = data['lane']
        lane_normals = data['lane_norm']
        agent_id = data['agent_id']
        city = data['city']
        
        inputs = ([
            data['pos_2s'], data['vel_2s'], 
            data['pos0'], data['vel0'], 
            data['accel'], None,
            data['lane'], data['lane_norm'], 
            data['car_mask'], data['lane_mask']
        ])

        pr_pos1, pr_vel1, states = model(inputs)
        gt_pos1 = data['pos1']

        # l = 0.5 * loss_fn(pr_pos1, gt_pos1, model.num_fluid_neighbors.unsqueeze(-1), data['car_mask'])
        l = 0.5 * loss_fn(pr_pos1, gt_pos1, 
                          torch.sum(data['car_mask'], dim = -2) - 1, data['car_mask'].squeeze(-1))

        pr_agent, gt_agent = get_agent(pr_pos1, data['pos1'],
                                       data['track_id0'], 
                                       data['track_id1'], 
                                       agent_id, device)

        # fluid_errors.add_errors(scene_id, data['frame_id0'][0], 
        #                         data['frame_id1'][0], pr_agent, 
        #                         gt_agent)
        if use_normalize_input:
            pred.append(pr_agent.unsqueeze(1).detach().cpu() *  normalize_scale)
            gt.append(gt_agent.unsqueeze(1).detach().cpu() *  normalize_scale)
        else:
            pred.append(pr_agent.unsqueeze(1).detach().cpu())
            gt.append(gt_agent.unsqueeze(1).detach().cpu())
        del pr_agent, gt_agent
        clean_cache(device)

        # pr_direction = get_lane_direction(
        #     pr_pos1, batch['city'][batch_i], am
        # )
        # pos_2s = data['pos_2s']
        # vel_2s = data['vel_2s']
        pos0 = data['pos0']
        vel0 = data['vel0']
        for i in range(29):
            pos_enc = torch.unsqueeze(pos0, 2)
            # pos_2s = torch.cat([pos_2s[:,:,1:,:], pos_enc], axis=2)
            vel_enc = torch.unsqueeze(vel0, 2)
            # vel_2s = torch.cat([vel_2s[:,:,1:,:], vel_enc], axis=2)
            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, data['accel'], None, 
                      data['lane'], data['lane_norm'], data['car_mask'], data['lane_mask'])
            pos0, vel0 = pr_pos1, pr_vel1
            pr_pos1, pr_vel1, states = model(inputs, states)
            clean_cache(device)
            
            if i < train_window - 1:
                gt_pos1 = data['pos'+str(i+2)]
                # l += 0.5 * loss_fn(pr_pos1, gt_pos1,
                #                    model.num_fluid_neighbors.unsqueeze(-1), data['car_mask'])
                l += 0.5 * loss_fn(pr_pos1, gt_pos1,
                                   torch.sum(data['car_mask'], dim = -2) - 1, data['car_mask'].squeeze(-1))

            pr_agent, gt_agent = get_agent(pr_pos1, data['pos'+str(i+2)],
                                           data['track_id0'], 
                                           data['track_id'+str(i+2)], 
                                           agent_id, device)

            # fluid_errors.add_errors(scene_id, data['frame_id'+str(i+1)][0], 
            #                         data['frame_id'+str(i+2)][0], pr_agent, 
            #                         gt_agent)
            if use_normalize_input:
                pred.append(pr_agent.unsqueeze(1).detach().cpu() *  normalize_scale)
                gt.append(gt_agent.unsqueeze(1).detach().cpu() *  normalize_scale)
            else:
                pred.append(pr_agent.unsqueeze(1).detach().cpu())
                gt.append(gt_agent.unsqueeze(1).detach().cpu())
            
            clean_cache(device)

        # print(pr_pos1[0], gt_pos1[0], pr_agent[0], gt_agent[0])
        
        losses.append(l)

        predict_result = (torch.cat(pred, axis=1), torch.cat(gt, axis=1))
        for idx, scene_id in enumerate(scenes):
            prediction_gt[scene_id] = (predict_result[0][idx], predict_result[1][idx])


    # with open('prediction_20t_nomap_5k.pickle', 'wb') as f:
    #     pickle.dump(predictions, f)
    
    total_loss = 128 * torch.sum(torch.stack(losses),axis=0) / max_iter
    
    result = {}
    de = {}
    # return total_loss, prediction_gt
    
    for k, v in prediction_gt.items():
        de[k] = torch.sqrt((v[0][:,0] - v[1][:,0])**2 + 
                        (v[0][:,1] - v[1][:,1])**2)
        
    ade = []
    de1s = []
    de2s = []
    de3s = []
    for k, v in de.items():
        ade.append(np.mean(v.numpy()))
        de1s.append(v.numpy()[9])
        de2s.append(v.numpy()[19])
        de3s.append(v.numpy()[-1])
    
    result['ADE'] = np.mean(ade)
    result['ADE_std'] = np.std(ade)
    result['DE@1s'] = np.mean(de1s)
    result['DE@1s_std'] = np.std(de1s)
    result['DE@2s'] = np.mean(de2s)
    result['DE@2s_std'] = np.std(de2s)
    result['DE@3s'] = np.mean(de3s)
    result['DE@3s_std'] = np.std(de3s)

    print(result)
    print('done')

    return total_loss, prediction_gt


def final_evaluation():
    am = ArgoverseMap()
    
    val_dataset = read_pkl_data(val_path, batch_size=8, shuffle=False, repeat=False)
    
    trained_model = torch.load(args.model_name + '.pth')
    trained_model.eval()
    
    with torch.no_grad():
        valid_total_loss, valid_metrics = evaluate(trained_model, val_dataset, am=am, 
                                                   train_window=args.train_window, max_iter=len(val_dataset), 
                                                   device=device, start_iter=200, use_lane=use_lane,
                                                   use_normalize_input=use_normalize_input, 
                                                   normalize_scale=normalize_scale)
    
    with open('results/{}_predictions.pickle'.format(args.model_name), 'wb') as f:
        pickle.dump(valid_metrics, f)
        
        
if __name__ == '__main__':
    main()
    
    # final_evaluation()
    
    
    