#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
import importlib
import torch
import pickle


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_utils import *


def evaluate(model, val_dataset, use_lane=False,
             train_window=3, max_iter=2500, device='cpu', start_iter=0, 
             batch_size=32):
    
    print('evaluating.. ', end='', flush=True)
        
    count = 0
    des = {}
    losses = 0
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
        
        data = {}
        convert_keys = (['pos' + str(i) for i in range(31)] + 
                        ['vel' + str(i) for i in range(31)] + 
                        ['pos_2s', 'vel_2s', 'lane', 'lane_norm'])

        for k in convert_keys:
            data[k] = torch.tensor(np.stack(sample[k])[...,:2], dtype=torch.float32, device=device)


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
            data['lane'], data['lane_norm'], 
            data['car_mask'], data['lane_mask']
        ])

        traj_preds, mode_pred = model(inputs)
        
        loss = nll_loss_multimodes(traj_preds, data, mode_pred)
        
        losses += loss

        predict_result = get_de_multi_modes(traj_preds, data)
        for idx, scene_id in enumerate(scenes):
            des[scene_id] = predict_result[idx].cpu()
    
    total_loss = losses / max_iter
    
    result = {}
        
    ade = []
    de1s = []
    de2s = []
    de3s = []
    for k, v in des.items():
        ade.append(np.min(np.mean(v.numpy(),0)))
        de1s.append(np.min(v.numpy()[9]))
        de2s.append(np.min(v.numpy()[19]))
        de3s.append(np.min(v.numpy()[-1]))
    
    result['minADE'] = np.mean(ade)
    result['minADE_std'] = np.std(ade)
    result['minDE@1s'] = np.mean(de1s)
    result['minDE@1s_std'] = np.std(de1s)
    result['minDE@2s'] = np.mean(de2s)
    result['minDE@2s_std'] = np.std(de2s)
    result['minDE@3s'] = np.mean(de3s)
    result['minDE@3s_std'] = np.std(de3s)

    print(result)
    print('done')

    return total_loss





