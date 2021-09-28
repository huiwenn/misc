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
        
        sigmas = []
        pred = []
        gt = []

        if count % 1 == 0:
            print('{}'.format(count + 1), end=' ', flush=True)
        
        count += 1
        
        batch = process_batch_ped(sample, device)


        box_zeros = torch.zeros(batch_size, 1, 2, device=device)
        boxnorm_zeros = torch.zeros(batch_size, 1, 2, device=device)
        box_mask = torch.ones(batch_size, 1, 1, device=device)

        m0 = torch.zeros((batch['pos_enc'].shape[0], 60, 2, 2), device=device)
        sigma0 = calc_sigma(m0)

        inputs = ([
            batch['pos_enc'], batch['vel_enc'], 
            batch['pos0'], batch['vel0'], 
            batch['accel'], sigma0, 
            box_zeros, boxnorm_zeros,
            batch['man_mask'], box_mask
        ])

        pr_pos1, pr_vel1, pr_m1, states = model(inputs)
        gt_pos1 = batch['pos1']
        
        sigma0 = sigma0 + calc_sigma(pr_m1)
        losses = loss_f(pr_pos1, gt_pos1, sigma0, batch['man_mask'].squeeze(-1))
        
        pr_agent, gt_agent, sigma_agent = pr_pos1[:,0], gt_pos1[:,0], sigma0[:,0]
        
        sigmas.append(sigma_agent.unsqueeze(1).detach().cpu())
        pred.append(pr_agent.unsqueeze(1).detach().cpu())
        gt.append(gt_agent.unsqueeze(1).detach().cpu())
        del pr_agent, gt_agent
        clean_cache(device)

        pos0 = batch['pos0']
        vel0 = batch['vel0']
        m0 = torch.zeros((batch_size, 60, 2, 2), device=device)
        for j in range(train_window-1):
            pos_enc = torch.unsqueeze(pos0, 2)
            vel_enc = torch.unsqueeze(vel0, 2)
            accel = pr_vel1 - vel_enc[...,-1,:]

            # test todo 
            # pr_m1 = torch.zeros((batch_size, 60, 2, 2), device=device)

            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, accel,
                      sigma0, 
                      box_zeros, boxnorm_zeros,
                      batch['man_mask'], box_mask)

            pos0, vel0, m0 = pr_pos1, pr_vel1, pr_m1

            pr_pos1, pr_vel1, pr_m1, states = model(inputs, states)
            clean_cache(device)
            
            gt_pos1 = batch['pos'+str(j+2)]

            sigma0 = sigma0 + calc_sigma(pr_m1)
            losses += loss_f(pr_pos1, gt_pos1, sigma0, batch['man_mask'].squeeze(-1))

            pr_agent, gt_agent, sigma_agent = pr_pos1[:,0], gt_pos1[:,0], sigma0[:,0]
            
            sigmas.append(sigma_agent.unsqueeze(1).detach().cpu())
            pred.append(pr_agent.unsqueeze(1).detach().cpu())
            gt.append(gt_agent.unsqueeze(1).detach().cpu())
            
            #clean_cache(device)

        predict_result = (torch.cat(pred, axis=1), torch.cat(gt, axis=1), torch.cat(sigmas,axis=1))

        scenes = batch['scene_idx'].tolist()

        for idx, scene_id in enumerate(scenes):
            prediction_gt[scene_id] = (predict_result[0][idx], predict_result[1][idx], predict_result[2][idx])
    
    #print('predgt', prediction_gt)
    total_loss = torch.sum(losses,axis=0) / (train_window) 
    
    result = {}
    de = {}
    coverage = {}

    for k, v in prediction_gt.items():
        #print('outputs', v[0], v[1])
        #M = v[0][:,2:].reshape(v[0].shape[0],2,2)
        #sig = calc_sigma(M)
        #print('sigma',sig)
        de[k] = torch.sqrt((v[0][:,0] - v[1][:,0])**2 + 
                        (v[0][:,1] - v[1][:,1])**2)
        coverage[k] = get_coverage_dyna(v[0][:,:2], v[1], v[2].reshape(train_window,2,2)) 
        
    ade = []
    for k, v in de.items():
        ade.append(np.mean(v.numpy()))
    
    acoverage = []
    for k, v in coverage.items():
        acoverage.append(np.mean(v.numpy()))


    result['loss'] = total_loss.detach().cpu().numpy()
    result['ADE'] = np.mean(ade)
    result['ADE_std'] = np.std(ade)
    result['coverage'] = np.mean(acoverage)

    fdes = []
    for k, v in de.items():
        fdes.append(v.numpy()[-1])
    result['FDE'] = np.mean(fdes)
    
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





