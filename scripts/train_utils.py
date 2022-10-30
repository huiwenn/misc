import torch
import gc
import numpy as np

def get_agent(pr: object, gt: object, pr_id: object, gt_id: object, agent_id: object, device: object = 'cpu', pr_m1: object = None) -> object: # only works for batch size 1
    agent_id = np.expand_dims(agent_id, 1)

    pr_agent = pr[pr_id == agent_id, :]
    gt_agent = gt[gt_id == agent_id, :]

    if pr_m1 is not None:
        pr_m_agent = torch.flatten(pr_m1[pr_id == agent_id, :], start_dim=-2, end_dim=-1)
    else:
        pr_m_agent = torch.zeros(pr_agent.shape[0], 0) #placeholder

    return torch.cat([pr_agent, pr_m_agent], dim=-1), gt_agent



def get_agent_multi(pr, gt, pr_id, gt_id, agent_id,  pr_m1, p, device = 'cpu',) -> object: # only works for batch size 1
    agent_id = np.expand_dims(agent_id, 1)

    pr_agent = pr[pr_id == agent_id, :]
    gt_agent = gt[gt_id == agent_id, :]

    pr_m_agent = torch.flatten(pr_m1[pr_id == agent_id, :], start_dim=-2, end_dim=-1)
    #print('pagent', p.shape)
    p_agent = p[gt_id == agent_id,:]

    return torch.cat([pr_agent, pr_m_agent], dim=-1), gt_agent, p_agent


def euclidean_distance(a, b, epsilon=1e-9, mask=1):
    return torch.sqrt(torch.sum((a - b)**2, axis=-1)*mask + epsilon)


def loss_fn(pr_pos, gt_pos, num_fluid_neighbors, car_mask):
    gamma = 0.5
    neighbor_scale = 1 / 40
    importance = torch.exp(-neighbor_scale * num_fluid_neighbors)
    dist = euclidean_distance(pr_pos[...,:2], gt_pos)**gamma
    mask_dist = dist * car_mask
    mask_dist = mask_dist[mask_dist.nonzero(as_tuple=True)] #rid zero values
    batch_losses = torch.mean(importance * mask_dist, axis=-1)
    return torch.sum(batch_losses)

def ecco_loss(pr_pos, gt_pos, pred_m, car_mask):
    l = loss_fn(pr_pos, gt_pos, torch.sum(car_mask) - 1, car_mask)
    return l

def mis_loss(pr_pos, gt_pos, pred_m, car_mask=None, rho = 0.9, scale=1 ,sigma_ready=False):
    if sigma_ready:
        sigma = pred_m
    else:
        sigma = calc_sigma(pred_m)
    c_alpha = - 2 * torch.log(torch.tensor(1.0)-rho)
    det = torch.det(sigma)
    c_ =  quadratic_func(gt_pos[...,:2] - pr_pos[...,:2], sigma.inverse()) / det #c prime

    c_delta = c_ - c_alpha
    c_delta = torch.where(c_delta > torch.tensor(0, device=c_delta.device), c_delta, torch.zeros(c_delta.shape, device=c_delta.device))

    mrs = torch.sqrt(det) * (c_alpha + scale*c_delta/rho)

    return torch.mean(mrs)    

def quadratic_func(x, M):
    part1 = torch.einsum('...x,...xy->...y', x, M)
    return torch.einsum('...x,...x->...', part1, x)

def calc_sigma_old(M):
    M = torch.tanh(M)
    sigma = torch.einsum('...xy,...xz->...yz', M, M)
    # scalling
    return 0.1*torch.matrix_exp(sigma)

def calc_sigma_multi(Ms):   
    modes = Ms.shape[-2]//2
    sigmas = []
    for i in range(modes):
        M = Ms[:,:,i*2:i*2+2,:]
        sig = calc_sigma_edit(M)
        sigmas.apend(sig)
    return sigmas

def calc_sigma(M):
    M = torch.tanh(M)
    expM = torch.matrix_exp(M)
    expMT = torch.matrix_exp(torch.transpose(M,-2,-1))
    sigma = torch.einsum('...xy,...yz->...xz', expM, expMT)
    # here sigma[0,0] ranges from 0.13 to 27.7
    sigma_scaled = 0.5*sigma 
    return sigma_scaled

def calc_sigma_edit(M, scale = 0.1):
    M = torch.tanh(M)
    expM = torch.matrix_exp(M)
    expMT = torch.matrix_exp(torch.transpose(M,-2,-1))
    sigma = torch.einsum('...xy,...yz->...xz', expM, expMT)
    return scale*sigma #for argoverse 0.1, for ped 1.5

def calc_u(sigma):
    
    if torch.isnan(sigma).any():
        print('input sigma nan already')

    device = sigma.device
    # This part tis to add a small scalar to the sigma such that linalg computs 
    eps = 1e-5
    # r1 = torch.empty(*sigma.shape[:-2],device=device).random_(2)
    # rsum = torch.ones_like(r1,device=device)
    # r0 = rsum - r1
    # rz = torch.zeros_like(r1.unsqueeze(-1),device=device)
    # mask = torch.cat([r1.unsqueeze(-1),rz,rz,r0.unsqueeze(-1)], axis=-1).reshape(sigma.shape[0],sigma.shape[1],2,2)
    mask = torch.eye(2, device=device).reshape((1,2,2)).repeat((sigma.shape[0], sigma.shape[1], 1, 1))
    sigma = sigma + eps*mask

    #print('sig', sigma)
    L, V = torch.linalg.eigh(sigma)
    #print('V', V)
    #print('L', L)
    U = V @ torch.diag_embed(L.pow(0.5))

    if torch.isnan(U).any():
        print('output U nan')

    return U

def calc_u_multi(sigmas):   
    modes = sigmas.shape[-2]//2
    Us = []
    for i in range(modes):
        M = sigmas[:,:,i*2:i*2+2,:]
        U = calc_u(M)
        Us.apend(U)
    return U

def nll(pr_pos, gt_pos, pred_m, car_mask=1):
    sigma = calc_sigma(pred_m)
    eps = 1e-6
    sigma = sigma + eps * torch.ones_like(sigma, device = sigma.device)
    loss = 0.5 * quadratic_func(gt_pos - pr_pos[...,:2], sigma.inverse()) \
        + torch.log(2 * 3.1416 * torch.pow(sigma.det(), 0.5))
    return torch.mean(loss * car_mask)

def nll_dyna(pr_pos, gt_pos, sigma, car_mask=1, prob=1):
    #print('det', torch.pow(sigma.det(), 0.5))

    loss = 0.5 * quadratic_func(gt_pos - pr_pos[...,:2], sigma.inverse()) \
        + torch.log(2 * 3.1416 * torch.pow(sigma.det(), 0.5))
    return torch.mean(loss * car_mask * prob)

def wta_loss(pr_pos, gt_pos, sigma, car_mask=1): 
    #winner take all loss
    all_nll_dyna = []
    for m in range(len(pr_pos)):
        all_nll_dyna.append(nll_dyna(pr_pos[m], gt_pos, sigma[m], car_mask))
    return  torch.min(torch.stack(all_nll_dyna))

    
class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def nll_multimodal_dyna(output, gt_pos, car_mask=1):
    print(output.shape) #batch, car, output, 2
    modes = output.shape[2] // 4
    weighted_loss=0
    for m in mode:
        pr_pos, sigma, weight = output[:,:, m,:] 
        weighted_loss += weight[0] * nll_dyna(pr_pos, gt_pos, sigma, car_mask=1)
    return weighted_loss

def get_coverage(pr_pos, gt_pos, pred_m, rho = 0.9, sigma_ready=False):
    if sigma_ready:
        sigma=pred_m
    else:
        sigma = calc_sigma(pred_m)
    det = torch.det(sigma)
    dist = quadratic_func(gt_pos - pr_pos, sigma.inverse()) / det
    contour = - 2 * torch.log(torch.tensor(1.0, device=dist.device)-rho)
    cover = torch.where(dist < contour, torch.ones(dist.shape, device=dist.device), torch.zeros(dist.shape, device=dist.device))
    return cover    

def sample(pos, sigma, k):
    """
    Samples from the model.
    args: 
        n:          int
    returns:
        x:          torch.Tensor (n, d)
        y:          torch.Tensor (n)
    """
    n = len(pos)
    counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.pi.squeeze()).sample()
    x = torch.empty(0, device=counts.device)
    y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

    # Only iterate over components with non-zero counts
    for k in np.arange(self.n_components)[counts > 0]: 
        d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
        x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

        x = torch.cat((x, x_k), dim=0)

    return x, y

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

    batch_size = len(batch['pos0'])

    batch_tensor = {}
    convert_keys = (['pos' + str(i) for i in range(train_window + 1)] + 
                    ['vel' + str(i) for i in range(train_window + 1)] + 
                    ['pos_2s', 'vel_2s', 'lane', 'lane_norm'])

    for k in convert_keys:
        batch_tensor[k] = torch.tensor(np.stack(batch[k])[...,:2], dtype=torch.float32, device=device)
        
    '''
    if use_normalize_input:
        batch_tensor, max_pos = normalize_input(batch_tensor, normalize_scale, train_window)
    ''' 
    for k in ['car_mask', 'lane_mask']:
        batch_tensor[k] = torch.tensor(np.stack(batch[k]), dtype=torch.float32, device=device).unsqueeze(-1)

    for k in ['track_id' + str(i) for i in range(30)] + ['agent_id']:
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


def process_batch_ped(batch, device, train_window = 12, train_particle_num=40):
    batch_tensor = {}

    batch_tensor['man_mask'] = torch.tensor(np.stack(batch['man_mask'])[:,:train_particle_num],
                                    dtype=torch.float32, device=device).unsqueeze(-1)

    convert_keys = (['pos' + str(i) for i in range(train_window + 1)] + 
                    ['vel' + str(i) for i in range(train_window + 1)] + 
                    ['pos_enc', 'vel_enc'])
    batch_tensor['scene_idx'] = np.stack(batch['scene_idx'])
        
    for k in convert_keys:
        batch_tensor[k] = torch.tensor(np.stack(batch[k])[:,:train_particle_num],#[...,:2],
                                        dtype=torch.float32, device=device)

    pos_zero = torch.unsqueeze(torch.zeros(batch_tensor['pos0'].shape[:-1], device=device),-1)
    batch_tensor['pos0'] = torch.cat([batch_tensor['pos0'], pos_zero], dim = -1)
    batch_tensor['vel0'] = torch.cat([batch_tensor['vel0'], pos_zero], dim = -1)

    zero_2s = torch.unsqueeze(torch.zeros(batch_tensor['vel_enc'].shape[:-1], device=device),-1)
    batch_tensor['vel_enc'] = torch.cat([batch_tensor['vel_enc'], zero_2s], dim = -1)
    batch_tensor['pos_enc'] = torch.cat([batch_tensor['pos_enc'], zero_2s], dim = -1)

    accel = batch_tensor['vel0'] - batch_tensor['vel_enc'][...,-1,:]
    # accel = torch.zeros(batch_size, 1, 2).to(device)
    batch_tensor['accel'] = accel
    batch_size = batch_tensor['pos0'].shape[0]
    #batch_tensor['sigmas'] = -5*torch.eye(2, device=device).reshape((1,2,2)).repeat((batch_tensor['pos0'].shape[0], batch_tensor['pos0'].shape[1], 1, 1)) 
    batch_tensor['sigmas'] = torch.zeros(batch_size, train_particle_num, 2, 2).to(device)

    return batch_tensor


def process_batch_ped_2d(batch, device, train_window = 12, train_particle_num=40):
    batch_tensor = {}

    batch_tensor['man_mask'] = torch.tensor(np.stack(batch['man_mask'])[:,:train_particle_num],
                                    dtype=torch.float32, device=device).unsqueeze(-1)

    convert_keys = (['pos' + str(i) for i in range(train_window + 1)] + 
                    ['vel' + str(i) for i in range(train_window + 1)] + 
                    ['pos_enc', 'vel_enc'])
    batch_tensor['scene_idx'] = np.stack(batch['scene_idx'])
        
    for k in convert_keys:
        batch_tensor[k] = torch.tensor(np.stack(batch[k])[:,:train_particle_num][...,:2],
                                        dtype=torch.float32, device=device)

    pos_zero = torch.unsqueeze(torch.zeros(batch_tensor['pos0'].shape[:-1], device=device),-1)

    zero_2s = torch.unsqueeze(torch.zeros(batch_tensor['vel_enc'].shape[:-1], device=device),-1)

    accel = batch_tensor['vel0'] - batch_tensor['vel_enc'][...,-1,:]
    # accel = torch.zeros(batch_size, 1, 2).to(device)
    batch_tensor['accel'] = accel
    return batch_tensor

