#!/usr/bin/env python3
import os
import sys
sys.path.append('..')
from collections import namedtuple
import time
import pickle
import argparse
import datetime
from datasets.argoverse_lane_loader import read_pkl_data
from train_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#os.environ["NCCL_DEBUG"] = "INFO"

parser = argparse.ArgumentParser(description="Training setting and hyperparameters")
parser.add_argument('--cuda_visible_devices', default='0,1,2,3')
parser.add_argument('--dataset_path', default='/path/to/argoverse_forecasting/', 
                    help='path to dataset folder, which contains train and val folders')
parser.add_argument('--train_window', default=4, type=int, help='how many timestamps to iterate in training')
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

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

model_name = args.model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)

val_path = os.path.join(args.dataset_path, 'val') #, 'lane_data'
train_path = os.path.join(args.dataset_path, 'train') #, 'lane_data'
    
def create_model():
    from models.cstcov import ParticlesNetwork
    """Returns an instance of the network for training and evaluation"""
    model = ParticlesNetwork()

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

    if args.loss == "ecco":
        loss_f = ecco_loss
    elif args.loss == "mis": 
        loss_f = mis_loss
    else: # args.loss == "nll":
        loss_f = nll

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

    def train_one_batch(model, batch, loss_f, train_window=2):

        batch_size = int(args.batch_size / args.batch_divide)
        if not args.use_lane:
            batch['lane'] = torch.zeros(batch_size, 1, 3, device=device)
            batch['lane_norm'] = torch.zeros(batch_size, 1, 3, device=device)
            batch['lane_mask'] = torch.ones(batch_size, 1, 1, device=device)

        pos_zero = torch.unsqueeze(torch.zeros(batch['pos0'].shape[:-1], device=device),-1)
        batch['pos0'] = torch.cat([batch['pos0'], pos_zero], dim = -1)
        batch['vel0'] = torch.cat([batch['vel0'], pos_zero], dim = -1)

        lane_zero = torch.zeros(batch_size, 900, 1, device=device)
        batch['lane'] = torch.cat([batch['lane'], lane_zero], dim = -1)
        batch['lane_norm'] = torch.cat([batch['lane_norm'], lane_zero], dim = -1)

        batch['accel'] = torch.cat([batch['accel'], torch.zeros(batch['accel'].shape[:-1],device=device).unsqueeze(-1)], dim = -1)
        zero_2s = torch.unsqueeze(torch.zeros(batch['vel_2s'].shape[:-1], device=device),-1)
        batch['vel_2s'] = torch.cat([batch['vel_2s'], zero_2s], dim = -1)

        inputs = ([
            batch['pos_2s'], batch['vel_2s'],
            batch['pos0'], batch['vel0'], 
            batch['accel'], batch['sigmas'], #other feats: 4x2 two M matrices
            batch['lane'], batch['lane_norm'], 
            batch['car_mask'], batch['lane_mask']
        ])
    
        pr_pos1, pr_vel1, pr_m1, states = model(inputs)

        # test todo
        # pr_m1 = torch.zeros((batch_size, 60, 2, 2), device=device) 

        gt_pos1 = batch['pos1']

        losses = loss_f(pr_pos1, gt_pos1, pr_m1, batch['car_mask'].squeeze(-1))
        del gt_pos1
        pos0 = batch['pos0']
        vel0 = batch['vel0']
        m0 = torch.zeros((batch_size, 60, 2, 2), device=device)

        for i in range(train_window-1):
            pos_enc = torch.unsqueeze(pos0, 2)
            vel_enc = torch.unsqueeze(vel0, 2)
            
            # test todo 
            #pr_m1 = torch.zeros((batch_size, 60, 2, 2), device=pos0.device) 
            
            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, batch['accel'],
                      torch.cat([m0, pr_m1], dim=-2), 
                      batch['lane'], batch['lane_norm'], 
                      batch['car_mask'], batch['lane_mask'])

            pos0, vel0, m0 = pr_pos1, pr_vel1, pr_m1
            # del pos_enc, vel_enc
            
            pr_pos1, pr_vel1, pr_m1, states = model(inputs, states)
            gt_pos1 = batch['pos'+str(i+2)]
            
            losses += loss_f(pr_pos1, gt_pos1, pr_m1, batch['car_mask'].squeeze(-1))

        total_loss = torch.sum(losses,axis=0) / (train_window)
        return total_loss
    
    epochs = args.epochs
    batches_per_epoch = args.batches_per_epoch   # batchs_per_epoch.  Dataset is too large to run whole data. 
    data_load_times = []  # Per batch 
    train_losses = []
    valid_losses = []
    valid_metrics_list = []
    min_loss = None
    '''
    with torch.profiler.profile(schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        with_stack=True
    ) as profiler:
    '''
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

            if batch_itr == batches_per_epoch - 1:
                print("... DONE", flush=True)

        epoch_train_loss = epoch_train_loss/(batches_per_epoch * args.batch_divide)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            print('loading validation dataset')
            val_dataset = read_pkl_data(val_path, batch_size=args.val_batch_size, shuffle=False, repeat=False)
            valid_total_loss, _ = evaluate(model.module, val_dataset, loss_f, train_window=args.val_window,
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
    #am = ArgoverseMap()
    if args.loss == "ecco":
        loss_f = ecco_loss
    elif args.loss == "mis": 
        loss_f = mis_loss
    else: # args.loss == "nll":
        loss_f = nll

    val_dataset = read_pkl_data(val_path, batch_size=args.val_batch_size, shuffle=False, repeat=False)
    
    trained_model = torch.load(model_name + '.pth')
    trained_model.eval()
    
    with torch.no_grad():
        valid_total_loss, valid_metrics = evaluate(trained_model, val_dataset, loss_f, train_window=args.val_window,
                                                       max_iter=args.val_batches, 
                                                       device=device, use_lane=args.use_lane, 
                                                       batch_size=args.val_batch_size)
    
    with open('results/{}_predictions.pickle'.format(model_name), 'wb') as f:
        pickle.dump(valid_metrics, f)
        




def evaluate(model, val_dataset, loss_f, use_lane=False,
             train_window=3, max_iter=2500, device='cpu', start_iter=0, 
             batch_size=32):
    
    print('evaluating.. ', end='', flush=True)
        
    count = 0
    prediction_gt = {}
    losses = 0

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

        data = process_batch(sample, device)

        if not use_lane:
            data['lane'] = torch.zeros(batch_size, 1, 3, device=device)
            data['lane_norm'] = torch.zeros(batch_size, 1, 3, device=device)
            data['lane_mask'] = torch.ones(batch_size, 1, 1, device=device)

        pos_zero = torch.unsqueeze(torch.zeros(data['pos0'].shape[:-1], device=device),-1)
        data['pos0'] = torch.cat([data['pos0'], pos_zero], dim = -1)
        data['vel0'] = torch.cat([data['vel0'], pos_zero], dim = -1)

        lane_zero = torch.zeros(batch_size, 650, 1, device=device)
        data['lane'] = torch.cat([data['lane'], lane_zero], dim = -1)
        data['lane_norm'] = torch.cat([data['lane_norm'], lane_zero], dim = -1)

        data['accel'] = torch.cat([data['accel'], torch.zeros(data['accel'].shape[:-1],device=device).unsqueeze(-1)], dim = -1)
        zero_2s = torch.unsqueeze(torch.zeros(data['vel_2s'].shape[:-1], device=device),-1)
        data['vel_2s'] = torch.cat([data['vel_2s'], zero_2s], dim = -1)

        

        lane = data['lane']
        lane_normals = data['lane_norm']
        agent_id = data['agent_id']
        city = data['city']
        scenes = data['scene_idx']

        inputs = ([
            data['pos_2s'], data['vel_2s'], 
            data['pos0'], data['vel0'], 
            data['accel'], data['sigmas'],
            data['lane'], data['lane_norm'], 
            data['car_mask'], data['lane_mask']
        ])

        pr_pos1, pr_vel1, pr_m1, states = model(inputs)

        #test todo 
        # pr_m1 = torch.zeros((batch_size, 60, 2, 2), device=device)

        gt_pos1 = data['pos1']
        
        losses = loss_f(pr_pos1, gt_pos1, pr_m1, data['car_mask'].squeeze(-1))

        pr_agent, gt_agent = get_agent(pr_pos1, data['pos1'],
                                       data['track_id0'], 
                                       data['track_id1'], 
                                       agent_id, device, pr_m1=pr_m1)
        
        pred.append(pr_agent.unsqueeze(1).detach().cpu())
        gt.append(gt_agent.unsqueeze(1).detach().cpu())
        del pr_agent, gt_agent
        clean_cache(device)

        pos0 = data['pos0']
        vel0 = data['vel0']
        m0 = torch.zeros((batch_size, 60, 2, 2), device=device)
        for j in range(train_window-1):
            pos_enc = torch.unsqueeze(pos0, 2)
            vel_enc = torch.unsqueeze(vel0, 2)
            
            # test todo 
            # pr_m1 = torch.zeros((batch_size, 60, 2, 2), device=device)

            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, data['accel'],
                      torch.cat([m0, pr_m1], dim=-2), 
                      data['lane'],
                      data['lane_norm'], data['car_mask'], data['lane_mask'])

            pos0, vel0, m0 = pr_pos1, pr_vel1, pr_m1

            pr_pos1, pr_vel1, pr_m1, states = model(inputs, states)
            clean_cache(device)
            

            gt_pos1 = data['pos'+str(j+2)]
            losses += loss_f(pr_pos1, gt_pos1, pr_m1, data['car_mask'].squeeze(-1))

            pr_agent, gt_agent = get_agent(pr_pos1, data['pos'+str(j+2)],
                                           data['track_id0'],
                                           data['track_id'+str(j+1)],
                                           agent_id, device, pr_m1=pr_m1)

            pred.append(pr_agent.unsqueeze(1).detach().cpu())
            gt.append(gt_agent.unsqueeze(1).detach().cpu())
            
            clean_cache(device)

        predict_result = (torch.cat(pred, axis=1), torch.cat(gt, axis=1))
        for idx, scene_id in enumerate(scenes):
            prediction_gt[scene_id] = (predict_result[0][idx], predict_result[1][idx])
    
    total_loss = torch.sum(losses,axis=0) / (train_window) 
    
    result = {}
    de = {}
    coverage = {}
    mis = {}

    for k, v in prediction_gt.items():
        de[k] = torch.sqrt((v[0][:,0] - v[1][:,0])**2 + 
                        (v[0][:,1] - v[1][:,1])**2)
        coverage[k] = get_coverage(v[0][:,:2], v[1], v[0][:,3:].reshape(train_window,2,2)) #pr_pos, gt_pos, pred_m, car_mask)
        mis[k] = mis_loss(v[0][:,:2], v[1],v[0][:,3:].reshape(train_window,2,2))


    ade = []
    for k, v in de.items():
        ade.append(np.mean(v.numpy()))
    
    acoverage = []
    for k, v in coverage.items():
        acoverage.append(np.mean(v.numpy()))


    amis = []
    for k, v in mis.items():
        amis.append(np.mean(v.numpy()))

    result['loss'] = total_loss.detach().cpu().numpy()
    result['ADE'] = np.mean(ade)
    result['ADE_std'] = np.std(ade)
    result['coverage'] = np.mean(acoverage)
    result['mis'] = np.mean(amis)

    if train_window >= 29:
        de1s = []
        de2s = []
        de3s = []
        cov1s = []
        cov2s = []
        cov3s = []
        for k, v in de.items():
            de1s.append(v.numpy()[10])
            de2s.append(v.numpy()[20])
            de3s.append(v.numpy()[-1])
        for k,v in coverage.items():
            cov1s.append(np.mean(v[:10].numpy()))
            cov2s.append(np.mean(v[10:20].numpy()))
            cov3s.append(np.mean(v[20:30].numpy()))

        result['DE@1s'] = np.mean(de1s)
        result['DE@1s_std'] = np.std(de1s)
        result['DE@2s'] = np.mean(de2s)
        result['DE@2s_std'] = np.std(de2s)
        result['DE@3s'] = np.mean(de3s)
        result['DE@3s_std'] = np.std(de3s)
        result['cov@1s'] = np.mean(cov1s)
        result['cov@2s'] = np.mean(cov2s)
        result['cov@3s'] = np.mean(cov3s)


    print(result)
    print('done')

    return total_loss, prediction_gt



        
if __name__ == '__main__':
    if args.train:
        # debug 大法好
        # with torch.autograd.detect_anomaly():
        train()
    
    if args.evaluation:
        evaluation()
    
    
    
