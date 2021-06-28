import torch
import gc
import numpy as np


def euclidean_distance(a, b, epsilon=1e-9):
    return torch.sqrt(torch.sum((a - b)**2, axis=-1) + epsilon)


def loss_fn(pr_pos, gt_pos, num_fluid_neighbors, car_mask):
    gamma = 0.5
    neighbor_scale = 1 / 40
    importance = torch.exp(-neighbor_scale * num_fluid_neighbors)
    dist = euclidean_distance(pr_pos, gt_pos)**gamma
    mask_dist = dist * car_mask
    batch_losses = torch.mean(importance * mask_dist, axis=-1)
    # print(batch_losses)
    return torch.sum(batch_losses)


def quadratic_func(x, M):
    part1 = torch.einsum('...x,...xy->...y', x, M)
    return torch.einsum('...x,...x->...', part1, x)


def calc_sigma(M):
    M1 = torch.tanh(M)
    sigma = torch.einsum('...xy,...xz->...yz', M1, M1)
    return torch.matrix_exp(sigma)


def nll(pr_pos, gt_pos, pred_m, car_mask):
    print('pred_m', pred_m.shape)

    sigma = calc_sigma(pred_m)
    
    loss = quadratic_func(gt_pos - pr_pos, sigma.inverse()) \
           + torch.log(2 * 3.1416 * torch.pow(sigma.det(), -0.5))

    print('loss', loss.shape)
    return torch.sum(loss * car_mask)


def clean_cache(device):
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()
    if device == torch.device('cpu'):
        # gc.collect()
        pass
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def unsqueeze_n(tensor, n):
    for i in range(n):
        tensor = tensor.unsqueeze(-1)
    return tensor
        
    
def normalize_input(tensor_dict, scale, train_window):
    pos_keys = (['pos' + str(i) for i in range(train_window + 1)] + 
                ['pos_2s', 'lane', 'lane_norm'])
    vel_keys = (['vel' + str(i) for i in range(train_window + 1)] + 
                ['vel_2s'])
    max_pos = torch.cat([torch.max(tensor_dict[pk].reshape(tensor_dict[pk].shape[0], -1), axis=-1)
                         .values.unsqueeze(-1) 
                         for pk in pos_keys], axis=-1)
    max_pos = torch.max(max_pos, axis=-1).values
    
    for pk in pos_keys:
        tensor_dict[pk][...,:2] = (tensor_dict[pk][...,:2] - unsqueeze_n(max_pos, len(tensor_dict[pk].shape)-1)) / scale
    
    for vk in vel_keys:
        tensor_dict[vk] = tensor_dict[vk] / scale
        
    return tensor_dict, max_pos

def process_batch(batch, device, train_window = 30): 
    '''processing script for new dataset'''
    batch_size = len(batch['city'])

    batch['lane_mask'] = [np.array([0])] * batch_size

    batch_tensor = {}

    for k in ['lane', 'lane_norm']:
        batch_tensor[k] = torch.tensor(np.stack(batch[k]), dtype=torch.float32, device=device)

    pos_2s = torch.tensor(batch['p_in'], dtype=torch.float32, device=device)
    vel_2s = torch.tensor(batch['v_in'], dtype=torch.float32, device=device)
    batch_tensor['pos_2s'] = pos_2s[...,:-1,:]
    batch_tensor['vel_2s'] = vel_2s[...,:-1,:]
    batch_tensor['pos0'] = pos_2s[..., -1,:]
    batch_tensor['vel0'] = vel_2s[..., -1,:]

    p_out = np.stack(batch['p_out'])
    v_out = np.stack(batch['v_out'])
    for k in range(train_window):
        batch_tensor['pos' + str(k+1)] = torch.tensor(p_out[:, :, k, :], dtype=torch.float32, device=device)
        batch_tensor['vel' + str(k+1)] = torch.tensor(v_out[:, :, k, :], dtype=torch.float32, device=device)

    for k in ['car_mask', 'lane_mask']:
        batch_tensor[k] = torch.tensor(np.stack(batch[k]), dtype=torch.float32, device=device).unsqueeze(-1)

    track_id = np.stack(batch['track_id'])
    for k in range(30):
        batch_tensor['track_id' + str(k)] = track_id[..., k, :]

    for k in ['city', 'agent_id', 'scene_idx']:
        batch_tensor[k] = np.stack(batch[k])
    
    batch_tensor['agent_id'] = batch_tensor['agent_id'][:,np.newaxis]

    batch_tensor['car_mask'] = batch_tensor['car_mask'].squeeze(-1)
    accel = torch.zeros(batch_size, 1, 2).to(device)
    batch_tensor['accel'] = accel



    # batch sigmas: starting with two zero 2x2 matrices
    batch_tensor['sigmas'] = torch.zeros(batch_size, 60, 4, 2).to(device)

    return batch_tensor