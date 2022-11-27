#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
from collections import namedtuple
import time
import pickle
import argparse
import datetime
from train_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm import tqdm


def train_one_batch(model, batch, loss_f, train_window=2):
    device = 'cuda'

    batch_size = batch['pos_2s'].shape[0]

    inputs = ([
        batch['pos_2s'], batch['vel_2s'],
        batch['pos0'], batch['vel0'], 
        batch['accel'], batch['sigmas'], #other feats: 4x2 two M matrices
        batch['car_mask']
    ])

    pr_pos1, pr_vel1, pr_m1, states = model(inputs)

    gt_pos1 = batch['pos1']
    losses = loss_f(pr_pos1, gt_pos1, pr_m1, batch['car_mask'].squeeze(-1))
    del gt_pos1
    pos0 = batch['pos0']
    vel0 = batch['vel0']

    for i in range(train_window-1):
        pos_enc = torch.unsqueeze(pos0, 2)
        vel_enc = torch.unsqueeze(vel0, 2)
        
        inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, batch['accel'],
                    pr_m1, 
                    batch['car_mask'])

        pos0, vel0 = pr_pos1, pr_vel1

        pr_pos1, pr_vel1, pr_m1, states = model(inputs, states)
        gt_pos1 = batch['pos'+str(i+2)]
        
        losses += loss_f(pr_pos1, gt_pos1, pr_m1, batch['car_mask'].squeeze(-1))

    total_loss = torch.sum(losses,axis=0) / (train_window)
    return total_loss

def train(model, model_name='cstconv_particles'):
    log_dir = "exp/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    sample_count = 0

    train_dataset = LSTMDataset(train_data)
    val_dataset = LSTMDataset(valid_data)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    loss_f = nll #nll_dyna
    
    base_lr=0.001

    model = MyDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), base_lr,betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.93)

    print('loaded datasets, starting training')

    '''
    if args.load_model_path:
        print('loading model from ' + args.load_model_path)
        model_ = torch.load(args.load_model_path + '.pth')
        model = model_
    else:
        model = create_model().to(device)
    '''
    
    epochs = 50 #args.epochs
    #batches_per_epoch = args.batches_per_epoch   # batchs_per_epoch.  Dataset is too large to run whole data. 
    data_load_times = []  # Per batch 
    train_losses = []
    valid_losses = []
    valid_metrics_list = []
    min_loss = None
    train_window = 19
    val_window = 19
    
    # first eval
    model.eval()
    with torch.no_grad():
        print('loading validation dataset')
        valid_total_loss, _, result = evaluate(model.module, val_dataloader, loss_f, 
                                                train_window=val_window)

    for i in range(epochs):
        print("training ... epoch " + str(i + 1), end='', flush=True)
        epoch_start_time = time.time()

        model.train()
        epoch_train_loss = 0 
        sub_idx = 0

        for i_batch, feed_dict in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            batch_tensor = make_batch_ped(feed_dict, device)
            current_loss = train_one_batch(model, batch_tensor, loss_f, train_window=train_window)

            current_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            
            if i_batch%50==0:
                print('loss', current_loss)
            
            del batch_tensor
            epoch_train_loss += float(current_loss)
        
            del current_loss
            clean_cache(device)


        epoch_train_loss = epoch_train_loss/782.0
        train_losses.append(epoch_train_loss)

        # ------ eval ------

        model.eval()
        with torch.no_grad():
            print('loading validation dataset')
            valid_total_loss, _, result = evaluate(model.module, val_dataloader, loss_f, 
                                                train_window=val_window)

            for k,v in result.items():
                writer.add_scalar(k, v, i)

        valid_losses.append(float(valid_total_loss))

        if min_loss is None:
            min_loss = valid_losses[-1]

        if valid_losses[-1] < min_loss:
            print('update weights')
            min_loss = valid_losses[-1] 
            best_model = model
            torch.save(model.module, model_name + ".pth")

        epoch_end_time = time.time()

        print('epoch: {}, train loss: {},  epoch time: {}, lr: {}, {}'.format(
            i + 1, train_losses[-1], 
            round((epoch_end_time - epoch_start_time) / 60, 5), 
            format(get_lr(optimizer), "5.2e"), model_name
        ))

        writer.add_scalar("Loss/train", train_losses[-1], i)
        #writer.add_scalar("Loss/validation", valid_losses[-1], i)
        writer.flush()

        scheduler.step()

    writer.close()

        

def evaluation(model_name):
    loss_f = nll
    val_dataset = LSTMDataset(valid_data)

    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    trained_model = torch.load(model_name + '.pth')
    trained_model.eval()

    with torch.no_grad():
        valid_total_loss, _, result = evaluate(model.module, val_dataloader, loss_f, 
                                                train_window=19)
    with open('results/{}_predictions.pickle'.format(model_name), 'wb') as f:
        pickle.dump(valid_metrics, f)


def evaluate(model, val_dataloader, loss_f, max_iter=10, train_window=3):
    
    print('evaluating.. ', end='', flush=True)
    device = 'cuda'
    count = 0
    prediction_gt = {}
    losses = 0
    model.eval()

    for i_batch, feed_dict in enumerate(tqdm(val_dataloader)):
        if i_batch > max_iter:
            break
        pred = []
        gt = []
        m1 = []

        if count % 1 == 0:
            print('{}'.format(count + 1), end=' ', flush=True)
        
        count += 1

        data = make_batch_ped(feed_dict, device)

        device = 'cuda'

        batch_size = data['pos_2s'].shape[0]

        inputs = ([
            data['pos_2s'], data['vel_2s'],
            data['pos0'], data['vel0'], 
            data['accel'], data['sigmas'], #other feats: 4x2 two M matrices
            data['car_mask']
        ])

        pr_pos1, pr_vel1, pr_m1, states = model(inputs)

        gt_pos1 = data['pos1']
        losses = loss_f(pr_pos1, gt_pos1, pr_m1, data['car_mask'].squeeze(-1))

        pred.append(pr_pos1.unsqueeze(1).detach().cpu())
        gt.append(gt_pos1.unsqueeze(1).detach().cpu())
        m1.append(pr_m1.unsqueeze(1).detach().cpu())
        clean_cache(device)

        pos0 = data['pos0']
        vel0 = data['vel0']

        for i in range(train_window-1):
            pos_enc = torch.unsqueeze(pos0, 2)
            vel_enc = torch.unsqueeze(vel0, 2)
            
            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, data['accel'],
                        pr_m1, 
                        data['car_mask'])

            pos0, vel0 = pr_pos1, pr_vel1

            pr_pos1, pr_vel1, pr_m1, states = model(inputs, states)
            gt_pos1 = data['pos'+str(i+2)]
            
            losses += loss_f(pr_pos1, gt_pos1, pr_m1, data['car_mask'].squeeze(-1))

            pred.append(pr_pos1.unsqueeze(1).detach().cpu())
            gt.append(gt_pos1.unsqueeze(1).detach().cpu())
            m1.append(pr_m1.unsqueeze(1).detach().cpu())

            
        total_loss = torch.sum(losses,axis=0) / (train_window)
        
        predict_result = (torch.cat(pred, axis=1), torch.cat(gt, axis=1), torch.cat(m1, axis=1))
        #print(predict_result[0].shape, predict_result[1].shape, predict_result[2].shape)
        for idx in range(predict_result[0].shape[0]):
            scene_id = idx
            prediction_gt[scene_id] = (predict_result[0][idx], predict_result[1][idx],predict_result[2][idx])
    
    result = {}
    de = {}
    coverage = {}
    mis = {}
    nlls = {}
    minde = {}
    es = {}

    for k, v in prediction_gt.items():
        de[k] = torch.sqrt((v[0][...,0] - v[1][...,0])**2 + 
                        (v[0][...,1] - v[1][...,1])**2)
                        
        cov = calc_sigma(v[2])
        coverage[k] = get_coverage(v[0][...,:2], v[1], cov) #pr_pos, gt_pos, pred_m, car_mask)
        #mis[k] = mis_loss(v[0][:,:2], v[1],v[0][:,3:].reshape(train_window,2,2))
        nlls[k] = nll(v[0][...,:2], v[1], v[2])
        es[k] = torch.norm(v[0][...,:2] - v[1], dim=-1) 
        es[k] -= torch.tensor([ [torch.trace(c_) for c_ in c ] for c in cov])

    ade = []
    for k, v in de.items():
        ade.append(np.mean(v.numpy()))
    
    fde = []
    for k, v in de.items():
        fde.append(np.mean(v[-1].numpy()))
    
    acoverage = []
    for k, v in coverage.items():
        acoverage.append(np.mean(v.numpy()))

    # amis = []
    # for k, v in mis.items():
    #     amis.append(np.mean(v.numpy()))
    aes = []
    for k, v in es.items():
        aes.append(np.mean(v.numpy()))
    
    anll = []
    for k, v in nlls.items():
        anll.append(np.mean(v.numpy()))

    result['loss'] = total_loss.detach().cpu().numpy()
    result['ADE'] = np.mean(ade)
    result['FDE'] = np.mean(fde)

    result['ADE_std'] = np.std(ade)
    result['coverage'] = np.mean(acoverage)
    #result['mis'] = np.mean(amis)
    result['nll'] = np.mean(anll)
    result['es'] = np.mean(aes)

    #result['minADE'] = np.mean(list(minde.values()))
    #result['minFDE'] = np.mean(np.array(list(minde.values()))[:,-1])


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

    return total_loss, prediction_gt, result

    
        
if __name__ == '__main__':

    test_data = np.load('/data/sophiasun/particles_data/loc_test_springs5.npy')
    train_data = np.load('/data/sophiasun/particles_data/loc_train_springs5.npy')
    valid_data = np.load('/data/sophiasun/particles_data/loc_valid_springs5.npy')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("using device", device)
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))

    from models.cstcov_ped import ParticlesNetwork
    model = ParticlesNetwork(encoder_hidden_size=95)
    model.to(device)
    print('made model. loading dataset')

    train(model, model_name='ctsconv_particles_es')
    
    #evaluation('ctsconv_particles')
    
    
 