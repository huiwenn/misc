import torch
import gc

def get_agent(pr, gt, pr_id, gt_id, agent_id, device='cpu'):
        
    pr_agent = pr[pr_id == agent_id,:]
    gt_agent = gt[gt_id == agent_id,:]
    
    return pr_agent, gt_agent

def get_de(traj_pred_one_mode, batch_tensor):
    des = []
    for t, pred in enumerate(traj_pred_one_mode):
        pred_agent, gt_agent = get_agent(pred, batch_tensor['pos'+str(t+1)], 
                               batch_tensor['track_id0'], batch_tensor['track_id'+str(t+1)], 
                               batch_tensor['agent_id'])
        des.append(torch.norm(pred_agent[...,0,:] - gt_agent, dim=-1))
    return torch.stack(des, -1)

def get_de_multi_modes(traj_pred, batch_tensor):
    des = []
    for m, preds in enumerate(traj_pred):
        des.append(get_de(preds, batch_tensor))
    return torch.stack(des, -1)

def quadratic_func(x, M):
    part1 = torch.einsum('...x,...xy->...y', x, M)
    return torch.einsum('...x,...x->...', part1, x)

def calc_sigma(M):
    M1 = torch.tanh(M)
    sigma = torch.einsum('...xy,...xz->...yz', M1, M1)
    return torch.matrix_exp(sigma)

def nll_loss(pred, gt, mask):
    mu = pred[...,0,:]
    # sigma = torch.einsum('...xy,...xz->...yz', pred[mask>0][...,1:,:], pred[mask>0][...,1:,:])
    sigma = calc_sigma(pred[...,1:,:])
    nll = quadratic_func(gt - mu, sigma.inverse()) + torch.log(sigma.det())
    return nll * mask

def nll_loss_per_sample(preds, data):
    loss = 0
    for i, pred in enumerate(preds):
        loss = loss + nll_loss(pred, data['pos'+str(i+1)], data['car_mask'][...,0])
    return loss / (i + 1)

def nll_loss_multimodes(preds, data, modes_pred, noise=0.0):
    """NLL loss multimodes for training.
    Args:
        pred is a list (with N modes) of predictions
        data is ground truth    
        noise is optional
    """
    modes = len(preds)
    log_lik = []   
    with torch.no_grad():
        for pred in preds:
            nll = nll_loss_per_sample(pred, data)
            log_lik.append(-nll.unsqueeze(-1))
        log_lik = torch.cat(log_lik, -1)
  
    priors = modes_pred.detach().unsqueeze(1)
      
    log_posterior_unnorm = log_lik + torch.log(priors)
    log_posterior_unnorm += torch.randn(*log_posterior_unnorm.shape).to(log_lik.device) * noise
    log_posterior = log_posterior_unnorm - torch.logsumexp(log_posterior_unnorm, axis=-1).unsqueeze(-1)
    post_pr = torch.exp(log_posterior)

    loss = 0.0
    for m, pred in enumerate(preds):
        nll_k = nll_loss_per_sample(pred, data) * post_pr[...,m] 
        nll_k = nll_k.sum(-1) / data['car_mask'][...,0].sum(-1)
        loss += nll_k.sum()

    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    loss += kl_loss(torch.log(modes_pred.unsqueeze(1)), post_pr) 
    return loss 


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

    