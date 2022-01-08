#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import pickle


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_utils import *



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

        if use_lane:
            pass
        else:
            sample['lane_mask'] = [np.array([0])] * batch_size
            data['lane'], data['lane_norm'] = torch.zeros(batch_size, 1, 2, device=device), torch.zeros(batch_size, 1, 2, device=device)
            data['lane_mask'] = torch.ones(batch_size, 1, 1, device=device)
        

        lane = data['lane']
        lane_normals = data['lane_norm']
        agent_id = data['agent_id']
        city = data['city']
        scenes = data['scene_idx'].tolist()

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
                                       data['track_id0'].squeeze(-1), 
                                       data['track_id1'].squeeze(-1), 
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
                                           data['track_id0'].squeeze(-1),
                                           data['track_id'+str(j+1)].squeeze(-1),
                                           agent_id, device, pr_m1=pr_m1)

            pred.append(pr_agent.unsqueeze(1).detach().cpu())
            gt.append(gt_agent.unsqueeze(1).detach().cpu())
            
            #clean_cache(device)

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
        coverage[k] = get_coverage(v[0][:,:2], v[1], v[0][:,2:].reshape(train_window,2,2)) #pr_pos, gt_pos, pred_m, car_mask)
        mis[k] = mis_loss(v[0][:,:2], v[1],v[0][:,2:].reshape(train_window,2,2))


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

    return total_loss, prediction_gt, result





