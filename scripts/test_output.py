import os
import sys
import time
import joblib
import tensorflow as tf
from termcolor import cprint
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as f
from tensorpack import dataflow
import pandas as pd
import gc
import pickle
from glob import glob
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from train_dynamics import *
from train_utils import *
sys.path.append('.')
sys.path.append('..')



def process_batch_test(batch, device): 
    '''processing script for new dataset'''
    
    batch_size = len(batch['city'])

    batch['lane_mask'] = [np.array([0])] * batch_size

    batch_tensor = {}

    for k in ['lane', 'lane_norm']:
        batch_tensor[k] = torch.tensor(np.stack(batch[k]), dtype=torch.float32, device=device)

    batch_size = len(batch['pos0'])

    batch_tensor = {}
    convert_keys = (['pos0', 'vel0','pos_2s', 'vel_2s', 'lane', 'lane_norm'])

    for k in convert_keys:
        batch_tensor[k] = torch.tensor(np.stack(batch[k])[...,:2], dtype=torch.float32, device=device)
        
    '''
    if use_normalize_input:
        batch_tensor, max_pos = normalize_input(batch_tensor, normalize_scale, train_window)
    ''' 
    for k in ['car_mask', 'lane_mask']:
        batch_tensor[k] = torch.tensor(np.stack(batch[k]), dtype=torch.float32, device=device).unsqueeze(-1)

    for k in ['track_id0'] + ['agent_id']:
        batch_tensor[k] = np.array(batch[k])
    
    batch_tensor['car_mask'] = batch_tensor['car_mask'].squeeze(-1)
    accel = torch.zeros(batch_size, 1, 2).to(device)
    batch_tensor['accel'] = accel


    # batch sigmas: starting with two zero 2x2 matrices
    batch_tensor['scene_idx'] = batch['scene_idx']
    batch_tensor['city'] = batch['city']
    batch_tensor['sigmas'] = torch.zeros(batch_size, 60, 4, 2).to(device) # for pecco change this back to 60, 2, 2
    #batch_tensor
    return batch_tensor


def get_agent_test(pr: object, pr_id: object, agent_id: object, device: object = 'cpu', pr_m1: object = None) -> object: # only works for batch size 1
    agent_id = np.expand_dims(agent_id, 1)

    pr_agent = pr[pr_id == agent_id, :]

    if pr_m1 is not None:
        pr_m_agent = torch.flatten(pr_m1[pr_id == agent_id, :], start_dim=-2, end_dim=-1)
    else:
        pr_m_agent = torch.zeros(pr_agent.shape[0], 0) #placeholder

    return torch.cat([pr_agent, pr_m_agent], dim=-1)


def predict_test(sample, model, device):

    data = process_batch_test(sample, device)
    del sample
    sample_k=6
    with torch.no_grad():
        lane = data['lane']
        lane_normals = data['lane_norm']
        agent_id = data['agent_id']
        city = data['city']
        scenes = data['scene_idx']

        sigmas = []
        pred = []
        samples = []

        m0 = -10*torch.eye(2, device=device).reshape((1,2,2)).repeat((data['pos0'].shape[0], 60, 1, 1))
        sigma0 = calc_sigma_edit(m0)

        inputs = ([
            data['pos_2s'], data['vel_2s'], 
            data['pos0'], data['vel0'], 
            data['accel'], sigma0,
            data['lane'], data['lane_norm'], 
            data['car_mask'], data['lane_mask']
        ])
        pr_pos1, pr_vel1, pr_m1, states = model(inputs)

        sigma0 = sigma0 + calc_sigma_edit(pr_m1)

        pr_agent = get_agent_test(pr_pos1, 
                                       data['track_id0'], 
                                       agent_id, device, pr_m1=sigma0)

        sigmas.append(sigma0.unsqueeze(1).detach().cpu())
        pred.append(pr_agent.unsqueeze(1).detach().cpu())
        p = torch.distributions.MultivariateNormal(pr_agent[:, :2], pr_agent[:, 2:].reshape(pr_agent.shape[0],2,2))
        sample = p.sample(sample_shape=(sample_k,))
        samples.append(sample)  
        
        del pr_agent
        clean_cache(device)

        pos0 = data['pos0']
        vel0 = data['vel0']
        for j in range(29):
            pos_enc = torch.unsqueeze(pos0, 2)
            vel_enc = torch.unsqueeze(vel0, 2)
            U = calc_sigma_edit(sigma0)
            inputs = (pos_enc, vel_enc, pr_pos1, pr_vel1, data['accel'],
                      U, 
                      data['lane'],
                      data['lane_norm'], data['car_mask'], data['lane_mask'])

            pos0, vel0 = pr_pos1, pr_vel1

            pr_pos1, pr_vel1, pr_m1, states = model(inputs, states)
            clean_cache(device)

            sigma0 = sigma0 + calc_sigma_edit(pr_m1)

            pr_agent = get_agent_test(pr_pos1, 
                                   data['track_id0'],
                                   agent_id, device, pr_m1=sigma0)

            sigmas.append(sigma0.unsqueeze(1).detach().cpu())
            pred.append(pr_agent.unsqueeze(1).detach().cpu())

            p = torch.distributions.MultivariateNormal(pr_agent[:, :2], pr_agent[:, 2:].reshape(pr_agent.shape[0],2,2))
            sample = p.sample(sample_shape=(sample_k,))
            samples.append(sample)  

            del inputs, pos_enc, vel_enc, pr_m1
            torch.cuda.empty_cache()
            clean_cache(device)

        predict_result = (torch.cat(pred, axis=1), torch.cat(sigmas,axis=1))

        return predict_result, torch.cat(samples,axis=1)

    
class ArgoverseTest(object):

    def __init__(self, file_path: str, shuffle: bool = True, random_rotation: bool = False,
                 max_car_num: int = 50, freq: int = 10, use_interpolate: bool = False, 
                 use_lane: bool = False, use_mask: bool = True):
        if not os.path.exists(file_path):
            raise Exception("Path does not exist.")

        self.afl = ArgoverseForecastingLoader(file_path)
        self.shuffle = shuffle
        self.random_rotation = random_rotation
        self.max_car_num = max_car_num
        self.freq = freq
        self.use_interpolate = use_interpolate
        self.am = ArgoverseMap()
        self.use_mask = use_mask
        self.file_path = file_path
        

    def get_feat(self, scene):

        data, city = self.afl[scene].seq_df, self.afl[scene].city

        lane = np.array([[0., 0.]], dtype=np.float32)
        lane_drct = np.array([[0., 0.]], dtype=np.float32)


        tstmps = data.TIMESTAMP.unique()
        tstmps.sort()

        data = self._filter_imcomplete_data(data, tstmps, 20)

        data = self._calc_vel(data, self.freq)

        agent = data[data['OBJECT_TYPE'] == 'AGENT']['TRACK_ID'].values[0]

        car_mask = np.zeros((self.max_car_num, 1), dtype=np.float32)
        car_mask[:len(data.TRACK_ID.unique())] = 1.0

        feat_dict = {'city': city, 
                     'lane': lane, 
                     'lane_norm': lane_drct, 
                     'scene_idx': scene,  
                     'agent_id': agent, 
                     'car_mask': car_mask}

        pos_enc = [subdf[['X', 'Y']].values[np.newaxis,:] 
                   for _, subdf in data[data['TIMESTAMP'].isin(tstmps[:19])].groupby('TRACK_ID')]
        pos_enc = np.concatenate(pos_enc, axis=0)
        # pos_enc = self._expand_dim(pos_enc)
        feat_dict['pos_2s'] = self._expand_particle(pos_enc, self.max_car_num, 0)

        vel_enc = [subdf[['vel_x', 'vel_y']].values[np.newaxis,:] 
                   for _, subdf in data[data['TIMESTAMP'].isin(tstmps[:19])].groupby('TRACK_ID')]
        vel_enc = np.concatenate(vel_enc, axis=0)
        # vel_enc = self._expand_dim(vel_enc)
        feat_dict['vel_2s'] = self._expand_particle(vel_enc, self.max_car_num, 0)

        pos = data[data['TIMESTAMP'] == tstmps[19]][['X', 'Y']].values
        pos = self._expand_dim(pos)
        feat_dict['pos0'] = self._expand_particle(pos, self.max_car_num, 0)
        vel = data[data['TIMESTAMP'] == tstmps[19]][['vel_x', 'vel_y']].values
        vel = self._expand_dim(vel)
        feat_dict['vel0'] = self._expand_particle(vel, self.max_car_num, 0)
        track_id =  data[data['TIMESTAMP'] == tstmps[19]]['TRACK_ID'].values
        feat_dict['track_id0'] = self._expand_particle(track_id, self.max_car_num, 0, 'str')
        feat_dict['frame_id0'] = 0
    
        '''
        for t in range(31):
            pos = data[data['TIMESTAMP'] == tstmps[19 + t]][['X', 'Y']].values
            pos = self._expand_dim(pos)
            feat_dict['pos' + str(t)] = self._expand_particle(pos, self.max_car_num, 0)
            vel = data[data['TIMESTAMP'] == tstmps[19 + t]][['vel_x', 'vel_y']].values
            vel = self._expand_dim(vel)
            feat_dict['vel' + str(t)] = self._expand_particle(vel, self.max_car_num, 0)
            track_id =  data[data['TIMESTAMP'] == tstmps[19 + t]]['TRACK_ID'].values
            feat_dict['track_id' + str(t)] = self._expand_particle(track_id, self.max_car_num, 0, 'str')
            feat_dict['frame_id' + str(t)] = t
        '''
        return feat_dict
    
    def __len__(self):
        return len(glob(os.path.join(self.file_path, '*')))

    @classmethod
    def _expand_df(cls, data, city_name):
        timestps = data['TIMESTAMP'].unique().tolist()
        ids = data['TRACK_ID'].unique().tolist()
        df = pd.DataFrame({'TIMESTAMP': timestps * len(ids)}).sort_values('TIMESTAMP')
        df['TRACK_ID'] = ids * len(timestps)
        df['CITY_NAME'] = city_name
        return pd.merge(data, df, on=['TIMESTAMP', 'TRACK_ID'], how='right')


    @classmethod
    def __calc_vel_generator(cls, df, freq=10):
        for idx, subdf in df.groupby('TRACK_ID'):
            sub_df = subdf.copy().sort_values('TIMESTAMP')
            sub_df[['vel_x', 'vel_y']] = sub_df[['X', 'Y']].diff() * freq
            yield sub_df.iloc[1:, :]

    @classmethod
    def _calc_vel(cls, df, freq=10):
        return pd.concat(cls.__calc_vel_generator(df, freq=freq), axis=0)
    
    @classmethod
    def _expand_dim(cls, ndarr, dtype=np.float32):
        return np.insert(ndarr, 2, values=0, axis=-1).astype(dtype)
    
    @classmethod
    def _linear_interpolate_generator(cls, data, col=['X', 'Y']):
        for idx, df in data.groupby('TRACK_ID'):
            sub_df = df.copy().sort_values('TIMESTAMP')
            sub_df[col] = sub_df[col].interpolate(limit_direction='both')
            yield sub_df.ffill().bfill()
    
    @classmethod
    def _linear_interpolate(cls, data, col=['X', 'Y']):
        return pd.concat(cls._linear_interpolate_generator(data, col), axis=0)
    
    @classmethod
    def _filter_imcomplete_data(cls, data, tstmps, window=20):
        complete_id = list()
        for idx, subdf in data[data['TIMESTAMP'].isin(tstmps[:window])].groupby('TRACK_ID'):
            if len(subdf) == window:
                complete_id.append(idx)
        return data[data['TRACK_ID'].isin(complete_id)]
    
    @classmethod
    def _expand_particle(cls, arr, max_num, axis, value_type='int'):
        dummy_shape = list(arr.shape)
        dummy_shape[axis] = max_num - arr.shape[axis]
        dummy = np.zeros(dummy_shape)
        if value_type == 'str':
            dummy = np.array(['dummy' + str(i) for i in range(np.product(dummy_shape))]).reshape(dummy_shape)
        return np.concatenate([arr, dummy], axis=axis)

class process_utils(object):
            
    @classmethod
    def expand_dim(cls, ndarr, dtype=np.float32):
        return np.insert(ndarr, 2, values=0, axis=-1).astype(dtype)
    
    @classmethod
    def expand_particle(cls, arr, max_num, axis, value_type='int'):
        dummy_shape = list(arr.shape)
        dummy_shape[axis] = max_num - arr.shape[axis]
        dummy = np.zeros(dummy_shape)
        if value_type == 'str':
            dummy = np.array(['dummy' + str(i) for i in range(np.product(dummy_shape))]).reshape(dummy_shape)
        return np.concatenate([arr, dummy], axis=axis)
    

def get_max_min(datas):
    mask = datas['car_mask']
    slicer = mask[0].astype(bool).flatten()
    pos_keys = ['pos0'] + ['pos_2s']
    max_x = np.concatenate([np.max(np.stack(datas[pk])[0,slicer,...,0]
                                   .reshape(np.stack(datas[pk]).shape[0], -1), 
                                   axis=-1)[...,np.newaxis]
                            for pk in pos_keys], axis=-1)
    min_x = np.concatenate([np.min(np.stack(datas[pk])[0,slicer,...,0]
                                   .reshape(np.stack(datas[pk]).shape[0], -1), 
                                   axis=-1)[...,np.newaxis]
                            for pk in pos_keys], axis=-1)
    max_y = np.concatenate([np.max(np.stack(datas[pk])[0,slicer,...,1]
                                   .reshape(np.stack(datas[pk]).shape[0], -1), 
                                   axis=-1)[...,np.newaxis]
                            for pk in pos_keys], axis=-1)
    min_y = np.concatenate([np.min(np.stack(datas[pk])[0,slicer,...,1]
                                   .reshape(np.stack(datas[pk]).shape[0], -1), 
                                   axis=-1)[...,np.newaxis]
                            for pk in pos_keys], axis=-1)
    max_x = np.max(max_x, axis=-1) + 10
    max_y = np.max(max_y, axis=-1) + 10
    min_x = np.max(min_x, axis=-1) - 10
    min_y = np.max(min_y, axis=-1) - 10
    return min_x, max_x, min_y, max_y


def process_func(putil, datas, am):
    
    city = datas['city'][0]
    x_min, x_max, y_min, y_max = get_max_min(datas)

    seq_lane_props = am.city_lane_centerlines_dict[city]

    lane_centerlines = []
    lane_directions = []

    # Get lane centerlines which lie within the range of trajectories
    for lane_id, lane_props in seq_lane_props.items():

        lane_cl = lane_props.centerline

        if (
            np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min
        ):
            lane_centerlines.append(lane_cl[1:])
            lane_drct = np.diff(lane_cl, axis=0)
            lane_directions.append(lane_drct)
    if len(lane_centerlines) > 0:
        lane = np.concatenate(lane_centerlines, axis=0)
        # lane = putil.expand_dim(lane)
        lane_drct = np.concatenate(lane_directions, axis=0)
        # lane_drct = putil.expand_dim(lane_drct)[...,:3]
        datas['lane'] = [lane]
        datas['lane_norm'] = [lane_drct]
        return datas
    else:
        return datas

if __name__ == '__main__':

    device = 'cuda:0'
    print('loading model')
    torch.cuda.set_device(2) 
    model = torch.load( '../pecco_dyna_argo_edit_april.pth')
    model.to(device)

    am = ArgoverseMap()
    putil = process_utils()

    dataset_path = '/data/sophiasun/argoverse'

    print('loading dataset')
    test_path = os.path.join(dataset_path, 'test_obs', 'data')

    afl_test = ArgoverseForecastingLoader(test_path)
    at_test = ArgoverseTest(test_path, max_car_num=60)
    
    print('loaded dataset')

    train_num = len(afl_test)
    batch_start = time.time()
    #os.mkdir(os.path.join('.', 'test_results'))
    #os.mkdir(os.path.join('.', 'test_results_temp'))
    batch_start = time.time()
    output_all = {}
    counter = 0
    with open(os.path.join('.', 'test_results_temp','0.pkl'), 'rb') as f:
            output_all = pickle.load(f)
    print('loaded stuff. starting the rest')
    for i, data in enumerate(afl_test):
        print('\r'+str(i)+'/'+str(len(afl_test)),end="")
        
        if i % 1000 == 0:
            batch_end = time.time()
            print("SAVED ============= {} / {} ....... {}".format(i, counter, batch_end - batch_start))
            batch_start = time.time()
            '''
            with open(os.path.join('.', 'test_results_temp',str(i)+'.pkl'), 'wb') as f:
                pickle.dump(output_all, f)
            '''
        if i<78000:
            continue
        seq_id = int(data.current_seq.name[:-4])
        
        data = {k:[v] for k, v in at_test.get_feat(i).items()}
        datas = process_func(putil, data, am)
        pred, samples = predict_test(datas, model, device)
        
        output_all[seq_id] = samples.cpu()
    
    print('converting to numpy')
    for k,v in output_all.items():
        output_all[k] = v.cpu().numpy()

    print('saving everything')
    with open(os.path.join('.', 'test_results_temp','final.pkl'), 'wb') as f:
        pickle.dump(output_all, f)

    print('generating h5')
    from argoverse.evaluation.competition_util import generate_forecasting_h5
    
    output_path = './competition_files/'
    os.mkdir(os.path.join('.', 'competition_files'))

    generate_forecasting_h5(output_all, output_path) #this might take awhile