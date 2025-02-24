import argparse

import json
import time
import math
import os 
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchdiffeq import odeint_adjoint, odeint

from models import U, V, GnnV5, U_HJB
from NO3 import NO3, NO3_1, NO3_2
from loss_func import inference, loss_heat, loss_hjb, loss_pino_heat, loss_pino_hjb
from utils import redirect_log_file, set_seed, set_gpu_max_mem

class EqnConfig():
    def __init__(self, eqn_type, v_type, dim=None, total_time=None, num_time_interval=None):
        with open(f"configs/{eqn_type}_{v_type}_d{dim}.json", 'r') as f:
            config = json.load(f)
        self.eqn_type = config['eqn_type']
        self.v_type = config['v_type']
        self.dim = config['dim']
        self.u_width = config['u_width']
        self.total_time = config['total_time']
        if eqn_type == 'heat':
            self.a = config['a']
            self.b = config['b']
        if eqn_type == 'hjb':
            self.eps = config['eps']
        
        self.num_time_interval = config['num_time_interval']
        self.seed = config['seed']
        self.num_sample = config['num_sample']

        self.fit_n_iter = config['fit_n_iter']
        self.fit_bs = config['fit_bs']
        self.fit_thereshold = config['fit_thereshold']
        
        self.train_dropout = config['train_dropout']
        self.train_hiddens = config['train_hiddens']
        self.gnn_skip_connection = config['gnn_skip_connection']
        self.train_bs = config['train_bs']
        self.train_epoch = config['train_epoch']
        self.train_lr = config['train_lr']

class Heat():
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim  # PDE 的维度
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.a = eqn_config.a # 左边界
        self.b = eqn_config.b  # 右边界

        self.x_init = 0.0  # x 的初始条件
        self.sigma = self.dim + 0.0  # 波动率项

    def sample(self, num_sample, device, generator=None):
        x_sample = self.a + (self.b-self.a)*torch.rand((num_sample, self.dim), device=device)
        return x_sample

    def solution(self, x, u0, n_x, bs=1):
        # output u(x, t) = \int u0(y) N(y − x, 2t*I) dy, while N is density of gaussian
        # Inputs:
        #     x: (bs, N, d) or (N, d)
        #     t: float
        # Outputs: 
        #     u: (bs, N,)
        t = self.total_time
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)
        
        x_shape = x.shape
        x = x.view(-1, 1, x_shape[-1])   # (bs*N, 1, d)
        y = torch.randn((1, n_x, x_shape[-1]), device=x.device)   #(1, n_x, d)
        y = y * math.sqrt(2*t) + x   #scale and shift, (bs*N, n_x, d)  #NOTE: the sign should be positive, not negative!
        out =  u0(y).mean(dim=1) # ->(bs*N, n_x) -> (bs*N, )
        
        return out.view(x_shape[:-1])   # (bs, N)

class HJB():
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim  # PDE 的维度
        self.width = eqn_config.u_width
        self.total_time = eqn_config.total_time
        self.eps = eqn_config.eps
        
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval

        self.x_init = 0.0  # x 的初始条件

    def sample(self, theta_0, n_x):
        dim, width = self.dim, self.width
        bs = theta_0.shape[0]
        c_size = dim
        a_size = dim * width
        b_size = dim * width
        b = theta_0[:, c_size:c_size+b_size].reshape(bs, dim, width)
        a = theta_0[:, c_size+b_size: c_size+b_size+a_size].reshape(bs, dim, width)
        N_x = math.ceil(n_x / width) * width
        m = N_x // width
        
        x = torch.randn(bs, dim, N_x, device=theta_0.device) / a.repeat(1,1,m).abs() + b.repeat(1,1,m)
        x = x[:, :, :n_x] # (bs, dim, n_x)
        
        x = x.unsqueeze(2)   # (bs, dim, 1, n_x)
        mu = b.unsqueeze(-1)  #(bs, dim, width, 1)
        sigma = 1/a.abs().unsqueeze(-1) #(bs, dim, width, 1)
        # ref: https://cs229.stanford.edu/section/gaussians.pdf, Section3: The diagonal covariance matrix case
        p_x = (torch.exp(-(x - mu)**2/(2 * sigma**2) ) / (sigma * math.sqrt(2*math.pi))).prod(1).sum(1) / width  # (bs, dim, width, n_x) --> (bs, n_x)
        x = x.squeeze(2).permute(0, 2, 1)  # (bs, n_x, dim)
        return x, p_x  # (bs, n_x, dim), # (bs, n_x)

    def solution(self, x, g, n_y=20000, bs=1, t=0):
        # Inputs:
        #     x: (bs, N, d) or (N, d)
        #     g: terminal function (..., d) --> (...,)
        #     n_y: int, number of quadrature points
        #     bs: int, batch size
        #     t: float, time index
        # Outputs: 
        #     u: (bs, N,)
        T = self.total_time
        eps = self.eps
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)
        x_shape = x.shape
        x = x.reshape(-1, 1, x_shape[-1])   # (bs*N, 1, d)

        # generate quadrature points y form N(x, 2*eps*(T-t))
        # batched computation for saving memeory, tune n_y_bs to fit the GPU.
        n_y_bs = 100 #original=1000
        int_ = 0
        for _ in range(n_y//n_y_bs):
            y = torch.randn((x.shape[0], n_y_bs, x.shape[-1]), device=x.device)   #(bs*N, n_y_bs, d)
            y = y * math.sqrt(2*eps*(T-t)) + x   #scale and shift, (bs*N, n_y_bs, d)  #paper version
            exp_ = torch.exp(-g(y)/(2*eps)) # (bs*N, n_y_bs)  #textbook version
            int_ += exp_.mean(dim=-1)  # (bs*N)
        int_ /= (n_y//n_y_bs)
        out = -2*eps*torch.log(int_)  # (bs*N)  
        return out.view(x_shape[:-1])   # (bs, N)
    
def fit_theta(theta0, u, eqn, n_x_outer, n_x_inner, theta_init=None, n_iter=2000, thereshold=1e-3):
    '''
    Inputs:
        theta0: (bs, n_params)
        u: instance of models.U
        n_x_outer: number of samples points used in outer optimizer.
        n_x_inner: number of samples points used in inner sampling of true sol.
        theta_init:  tensor, initial value of theta
        n_iter: number of iterations. 2000 for 10dim, 3000 for 15dim
    Outputs:
        theta: (bs, n_params)
    '''
    def u0(y):
        theta0_rep = theta0.unsqueeze(1).repeat(1, n_x_outer, 1).reshape(-1, u.n_params) 
        return u.forward_2(theta0_rep, y)
    
    bs = theta0.shape[0]
    device = theta0.device
    if not theta_init is None:
        theta = nn.Parameter(theta_init.clone())
    else:
        theta = nn.Parameter(theta0.clone())
    
    # minimize ||f(x) - u(x)||^2 by Adam
    optimizer = torch.optim.Adam([theta], lr=0.001)
    scaler = torch.amp.GradScaler('cuda')  #NOTE: use mixed precision training, save memory.

    for i in range(0,n_iter+1): #original:2001 for dim-10, 3001 for dim-15
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if i % 50 == 0: #update x every 50 iterations, save computation.
                x = eqn.sample(n_x_outer, device)
                true_sol = eqn.solution(x, u0, n_x_inner, bs)  # (bs, n_x_outer)
            
            pred_sol = u.forward_2(theta, x)
            relative_error = ((true_sol - pred_sol)**2).mean(dim=1) / (true_sol**2).mean(dim=1)
            loss = relative_error.mean()
            if loss < thereshold:
                break
            if i % 500 == 0:
               print(f"iter {i}, loss={loss.item()}", flush=1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return theta.detach(), relative_error.detach()   # (bs, n_params), (bs, )

## generate samples uniformly from a ball using the Muller method
## see http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
def d_ball_uniform(num_samples,d,scale_max,**kwargs):
    scale_min = kwargs.get("scale_min",0)
    u = torch.randn((num_samples,d))
    norms = (torch.sum(u**2,1).unsqueeze(-1)) ** 0.5
    r = (scale_max-scale_min)*torch.rand((num_samples,1))+scale_min

    final_samples = r*u/norms
    return final_samples

def gen_train_dataset(n_params, n_1 = 100000, n_2= 50000):
    '''training dataset, only initial values.'''
    theta_0_2 = torch.randn(n_2, n_params)*math.sqrt(0.5)
    if n_1 > 0:
        theta_0_1 = d_ball_uniform(n_1, n_params, n_params/50) #for dim=5, n_params=1000, n_params/50=20, i.e. |theta_0_1| < 20 
        theta_0 = torch.cat((theta_0_1, theta_0_2), dim=0)
    else:
        theta_0 = theta_0_2
    
    dataset = torch.utils.data.TensorDataset(theta_0)  #（n_1+n_2, n_params）
    return dataset

def gen_hjb_train_dataset(n, dim=8, width=50, is_train=True):
    w_low, w_high = -1, 0
    b_low, b_high = -2, 2
    c_low, c_high = -0.01, 0.01
    b = (b_high-b_low) * torch.rand(n, dim*width) + b_low
    w = (w_high-w_low) * torch.rand(n, width) + w_low
    c = (c_high-c_low) * torch.rand(n, dim) + c_low

    if is_train:
        a_low, a_high = 0.1, 2
        a = (a_high-a_low) * torch.rand(n, dim*width) + a_low
    if not is_train:
        a_low = math.sqrt(0.1)
        a_high = 2
        a = (a_high-a_low) * torch.rand(n, 1, width) + a_low
        a = a.repeat(1, dim, 1).reshape(n, dim*width)
    
    theta = torch.cat([c, b, a, w], dim=-1)  # (n, n_params)
    dataset = torch.utils.data.TensorDataset(theta)
    return dataset

def gen_heat_test_dataset(u, eqn, eqn_config):
    '''generate test dataset. initial values and true solutions.
    Inputs:
        u: instance of U
        seed: int
        num_samples: int, required size of dataset
        l2re_threshold: float, threshold of valid sample. 
        prev_dataset: str, path to dataset file
    Outputs:
        None. Dataset is stored in file
    '''
    set_seed(eqn_config.seed, cudnn_benchmark=True)
    n_params = u.n_params
    # Set bs, n_x_outer, n_x_inner according to GPU memory size.
    bs = 2  # 3 for 20GB, 11 for 80GB
    n_x_outer = 500  # 600 for 5-dim, 500 for 10-dim, 350 for 15-dim, 300 for 20-dim
    n_x_inner = 2500  #original: 2500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    theta_x_dataset = gen_train_dataset(n_params, n_1=10000, n_2=5000)  # n_1=10000, n_2=5000 -->（15000, n_params）
    theta_x_dataloader = torch.utils.data.DataLoader(theta_x_dataset, batch_size=bs, shuffle=True, drop_last=True)


    # --- phase 1: generate initial values ---
    print("start generate theta_T_init.",flush=1)
    start_time = time.time()
    theta_T_init = None
    theta_T_list = []
    for i, theta_0 in enumerate(theta_x_dataloader):
        theta_0 = theta_0[0].to(device)
        theta_T, L2RE = fit_theta(theta_0, u, eqn, n_x_outer, n_x_inner, theta_init=theta_T_init, n_iter=eqn_config.fit_n_iter, thereshold=eqn_config.fit_thereshold)
        theta_T_init = theta_T  # use the optimized theta_T as the initial value for the next iter.
        mask = L2RE < eqn_config.fit_thereshold
        if mask.sum() > 0:
            theta_T_list.append(theta_T[mask])
        if len(theta_T_list) >= bs:
            theta_T_init = torch.cat(theta_T_list, dim=0)[:bs]
            break
    assert not theta_T_init is None
    print(f"theta_T_init.shape={theta_T_init.shape}, time={time.time()-start_time}")

    # --- phase 2: generate dataset ---
    print("start generate dataset...", flush=1)
    os.makedirs("checkpoints", exist_ok=True)
    out_file = f'checkpoints/{eqn_config.eqn_type}_dataset_dim{u.dim}_params{u.n_params}_seed{eqn_config.seed}.pt'
    
    theta_0_list = list()
    theta_T_list = list()    
    cnt = 0
    start_time = time.time()
    for i, theta_0 in enumerate(theta_x_dataloader):
        theta_0 = theta_0[0].to(device)
        theta_T, L2RE = fit_theta(theta_0, u, eqn, n_x_outer, n_x_inner, theta_init=theta_T_init, n_iter=eqn_config.fit_n_iter, thereshold=eqn_config.fit_thereshold)
        msk = L2RE < eqn_config.fit_thereshold
        if msk.sum() > 0:
            theta_0_list.append(theta_0[msk])
            theta_T_list.append(theta_T[msk])
        cnt += msk.sum()
        if  (i+1)%100==0 or cnt >= eqn_config.num_sample:
            theta_0 = torch.cat(theta_0_list, dim = 0)
            theta_T = torch.cat(theta_T_list, dim = 0)
            dataset= torch.utils.data.TensorDataset(theta_0, theta_T)
            print(f'batch {i}, accept ratio = {cnt/((i+1)*bs)}, time = {time.time()-start_time}', flush=True)
            torch.save(dataset, out_file)
            if cnt >= eqn_config.num_sample:
                break
    print(f"generate dataset done, saved at {out_file}, num_sample {len(dataset)} time={time.time()-start_time}", flush=1)

def gen_hjb_test_dataset(u, eqn, eqn_config):
    '''
    Input:
        n: num of required samples
        u: instance of u hjb
        T: time horizon.
        eps: diffusion term.

    Output:
        a dataset with at least n data samples, Each sample contains:
        theta_T: tensor (n_params)
        x: tensor (n_x, dim)
        p: tensor (n_x, ) or (1, )
        u_0: tensor (n_x,),  
    '''
    seed = eqn_config.seed
    set_seed(seed)
    bs = 2
    n_x = 1000
    
    n = eqn_config.num_sample
    theta_T = gen_hjb_train_dataset(n=n+bs, dim=u.dim, width=u.width, is_train=False).tensors[0]
    n = len(theta_T)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = [[], [], [], []]
    for i in range(n//bs):
        theta_T_i = theta_T[i*bs:(i+1)*bs].to(device)
        x, p_x = eqn.sample(theta_T_i, n_x)  # (bs, n_x, dim)
        def g(y): #batched u.
            theta_rep = theta_T_i.unsqueeze(1).repeat(1, n_x, 1).reshape(-1, u.n_params)
            return u.forward_2(theta_rep, y)
        u_0 = eqn.solution(x, g, n_y=20000, bs=bs) #(bs, n_x)

        test_dataset[0].append(theta_T_i)
        test_dataset[1].append(x)
        test_dataset[2].append(p_x)
        test_dataset[3].append(u_0) 
    test_dataset = [torch.cat(test_dataset[i], dim=0) for i in range(4)]
    test_dataset = torch.utils.data.TensorDataset(*test_dataset)
    filename = f"checkpoints/{eqn_config.eqn_type}_dataset_dim{u.dim}_params{u.n_params}_seed{seed}.pt"
    torch.save(test_dataset, filename)
    print("dataset generated, saved to ", filename)
    return test_dataset

def train(u, v, eqn_config, eqn, ckp_path=None, u_label=None, is_para=False, is_pino=False):
    '''main function.'''
    if is_pino:
        model_path = f"checkpoints/{eqn_config.eqn_type}_{u.__class__.__name__}_{v.__class__.__name__}_dim{u.dim}.pth"
    else:
        model_path = f"checkpoints/{eqn_config.eqn_type}_{u.__class__.__name__}_{v.__class__.__name__}_dim{u.dim}_vdrop{v.dropout_p:.3f}.pth"
    print("model save at", model_path)
    print(f"num of trainable params={sum(p.numel() for p in v.parameters() if p.requires_grad)}, num of total params={sum(p.numel() for p in v.parameters())}")
    
    bs = eqn_config.train_bs
    n_epoch = eqn_config.train_epoch
    lr = eqn_config.train_lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #---define model---
    u_label = u if u_label is None else u_label
    u_label.to(device)
    u.to(device)
    v.to(device)
    if is_para:
        v = nn.DataParallel(v)  #NOTE: multi-GPU
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, v.parameters()), lr=lr)
    
    #---load data---
    eqn_type = eqn_config.eqn_type
    if eqn_type == 'heat':
        dataset_train = gen_train_dataset(u.n_params, n_1=100000, n_2=50000)
    elif eqn_type == 'hjb':
        dataset_train = gen_hjb_train_dataset(n=10000, dim=u.dim, width=u.width, is_train=True)
    dataload_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True, drop_last=False)
    
    dataset_name = f"checkpoints/{eqn_config.eqn_type}_dataset_dim{eqn_config.dim}_params{u.n_params}_seed{eqn_config.seed}.pt"
    print(f"---load dataset from {dataset_name}---")
    dataset_test = torch.load(dataset_name, map_location=device)  
    n = len(dataset_test)
    n_test = (n // 10) * 6
    n_valid = n  - n_test
    dataset_test, dataset_valid = torch.utils.data.random_split(dataset_test, [n_test, n_valid])
    dataload_test = torch.utils.data.DataLoader(dataset_test, batch_size=n_test, shuffle=False, drop_last=False)
    dataload_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=n_valid, shuffle=False, drop_last=False)
    
    
    os.makedirs("checkpoints/", exist_ok=True) 
    
    #---load checkpoint---
    if ckp_path:
        print(f"---load check point from {ckp_path}---")
        checkpoint = torch.load(ckp_path, map_location=device)
        v.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups: #Need reset lr
            param_group['lr'] = lr
        
    start_time = time.time()
    best_ckp = {'model_state_dict':None, 'optimizer_state_dict':None, 'valid_l2re':1e10}
    
    for i in range(n_epoch):
        v.train()
        for j, data in enumerate(dataload_train):
            theta_0 = data[0].to(device)
            try:
                if eqn_type == 'heat' and is_pino:
                    r_T = loss_pino_heat(theta_0, u, v, eqn)
                elif eqn_type == 'heat' and not is_pino:
                    r_T, theta_T_norm = loss_heat(theta_0, u, v, eqn)
                elif eqn_type == 'hjb' and is_pino:
                    r_T = loss_pino_hjb(theta_0, u, v, eqn)
                elif eqn_type == 'hjb' and not is_pino:
                    r_T, theta_T_norm = loss_hjb(theta_0, u, v, eqn)
                assert not r_T.isnan(), "r_T is nan"
                assert not r_T.isinf(), "r_T is inf"
                optimizer.zero_grad()
                loss = r_T
                loss.backward()
                optimizer.step()
                    
            except AssertionError as e:  #restart from last checkpoint if error
                print(f"epoch {i}, batch {j}, err:{e}", flush=True)
                print(f"r_T ={r_T.item()}")
                v.load_state_dict(best_ckp['model_state_dict'])
                optimizer.load_state_dict(best_ckp['optimizer_state_dict'])
                continue
            
            
            if j % 2 == 0:
                print(f"---- epoch {i}, batch {j} ({j/len(dataload_train):.4f}) ----")
                print(f"r_T: {r_T.item()}")
                T = eqn_config.total_time
                if eqn_type == 'hjb':
                    valid_l2re = eval_hjb(u, v, eqn, dataload_valid, T, device, is_pino=is_pino)
                    test_l2re = eval_hjb(u, v, eqn, dataload_test, T, device, is_pino=is_pino)
                else:
                    valid_l2re = eval_heat(u, v, eqn, dataload_valid, T, device, is_pino=is_pino, u_label=u_label)
                    test_l2re = eval_heat(u, v, eqn, dataload_test, T, device, is_pino=is_pino, u_label=u_label)
                print("Valid L2RE_l_p=",  valid_l2re, "Test L2RE_l_p=",  test_l2re)
                print(f"wall time(hours)={(time.time()-start_time)/3600:.4f}" , flush=True)

                if valid_l2re < best_ckp['valid_l2re'] :
                    best_ckp['valid_l2re'] = valid_l2re
                    best_ckp['model_state_dict'] = copy.deepcopy(v.state_dict())  #deepcopy for recovery, in case loss=nan.
                    best_ckp['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict()) 
                    print(f"best ckp updated", flush=True)
                    torch.save(best_ckp, model_path)

def test_heat(u, eqn, thetaT_label, thetaT_pred, normed=True, bs=-1, nx=5000, u_label=None):
    '''function distance between label and prediciton'''
    n = len(thetaT_pred)
    bs = min(5, n) if bs == -1 else bs
    u_label = u if u_label is None else u_label
    
    a = eqn.a
    b = eqn.b
    x = eqn.sample(nx, device=thetaT_pred.device)
    err_list = []
    for i in range(n//bs):
        thetaT_label_i = thetaT_label[i*bs:(i+1)*bs]
        thetaT_pred_i = thetaT_pred[i*bs:(i+1)*bs]
        pred_sol = u.forward_2(thetaT_pred_i, x)
        label_sol = u_label.forward_2(thetaT_label_i, x)
        if normed:
            err = (((label_sol - pred_sol)**2).mean(dim=1) / (label_sol**2).mean(dim=1)).mean()
        else:
            err = ((label_sol - pred_sol)**2).mean() * (b-a)**u.dim
        err_list.append(err)
    
    return sum(err_list)/len(err_list)

def test_hjb(u, data, pred_theta_0, normed=True):
    '''function distance between label and samples of true sol.'''
    device = pred_theta_0.device
    x = data[1].to(device)
    p_x = data[2].to(device)
    label_u_0 = data[3].to(device)
    pred_u_0 = u.forward_2(pred_theta_0, x)
    if normed:
        err = (((label_u_0 - pred_u_0)**2 / p_x).mean(dim=1) / (label_u_0**2 / p_x).mean(dim=1)).mean()
    else:
        err = ((label_u_0 - pred_u_0)**2 / p_x).mean() 
    return err

def eval_heat(u, v, eqn, dataloader, T, device, is_pino, verbose=False, u_label=None):
    '''evaluate the model on the test dataset.'''
    v.eval()
    if u_label is None:
        u_label = u
        
    with torch.no_grad():
        L2RE_list = []
        for theta0_label, thetaT_label in dataloader:
            theta0_label = theta0_label.to(device)
            thetaT_label = thetaT_label.to(device)
            if is_pino:
                t = torch.ones(theta0_label.shape[0], device=device) * T
                thetaT_pred = v(theta0_label, t)
            else:
                thetaT_pred = inference(theta0_label, v, T) #NOTE: original: use simple inference
            
            L2RE = test_heat(u, eqn, thetaT_label, thetaT_pred, u_label=u_label)
            L2RE_list.append(L2RE)

        if verbose: #print at last batch
            print(f"l2RE_list={torch.tensor(L2RE_list)}")
        return sum(L2RE_list)/len(L2RE_list)

def eval_hjb(u, v, eqn, dataloader, T, device, is_pino=False, verbose=False):
    '''eval v based on pre-generated test_dataset.'''
    v.eval()
    with torch.no_grad():
        L2RE_list = []
        for data in dataloader:
            theta_T = data[0].to(device).float()
            if is_pino:
                t = torch.ones(theta_T.shape[0], device=device) * T
                pred_theta_0 = v(theta_T, t)
            else:
                pred_theta_0 = inference(theta_T, v, T)
            
            L2RE = test_hjb(u, data, pred_theta_0)
            L2RE_list.append(L2RE)

        if verbose: #print last batch
            print(f"l2RE_list={torch.tensor(L2RE_list)}")
        return sum(L2RE_list)/len(L2RE_list)


# ---- unit test codes -----
def test_train(eqn_config, v_type, drop_out):
    eqn_type = eqn_config.eqn_type
    if eqn_type == 'heat':
        u = U(dim=eqn_config.dim, width=eqn_config.u_width)
    elif eqn_type == 'hjb':
        u = U_HJB(dim=eqn_config.dim, width=eqn_config.u_width)
    else:
        u = None
    print(f"u.type={u.__class__.__name__}, u.width={u.width}")
    
    if eqn_type == "heat":
        eqn = Heat(eqn_config)
    elif eqn_type == "hjb":
        eqn = HJB(eqn_config)
    else:
        eqn = None
        
    v_width = 1000
    if drop_out:
        dropout_p = eqn_config.train_dropout
    else:
        dropout_p = 0.0
    
    is_pino = False
    if v_type == 'gnn':
        n_hiddens = eqn_config.train_hiddens
        if eqn_config.gnn_skip_connection:
            skip_flag = True
        else:
            skip_flag = False
        v = GnnV5(u, dropout_p=dropout_p, n_hiddens=n_hiddens, skip_connection=skip_flag)
    elif v_type == 'no':
        is_pino = True
        T = eqn_config.total_time
        n_hiddens = eqn_config.train_hiddens
        v = NO3_1(dim=u.n_params, width=n_hiddens, depth=2, T=T)
    else: # MLP V
        v = V(dim=u.n_params, dropout_p=dropout_p, width=v_width)

    ckp_path = None
    u_label = U(dim=eqn_config.dim, width=eqn_config.u_width)
    train(u, v, eqn_config, eqn, ckp_path=ckp_path, u_label=u_label, is_pino=is_pino)

def test_gen_dataset(eqn_config):
    if eqn_type == "heat":
        eqn = Heat(eqn_config)
        u = U(eqn_config.dim, width=eqn_config.u_width)
        gen_heat_test_dataset(u=u, eqn=eqn, eqn_config=eqn_config)
    elif eqn_type == "hjb":
        eqn = HJB(eqn_config)
        u = U_HJB(eqn_config.dim, width=eqn_config.u_width)
        gen_hjb_test_dataset(u=u, eqn=eqn, eqn_config=eqn_config)
    else:
        eqn = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="方程类型，维度以及是否dropout")
    parser.add_argument('--eqn_type', type=str, default='hjb', help='方程类型')
    parser.add_argument('--cur_dim', type=int, default=10, help='数据维度')
    parser.add_argument('--v_type', type=str, default='no', help='模型类型')
    parser.add_argument('--drop_out', type=int, default=1, help='是否dropout')
    parser.add_argument('--train_mode', type=int, default=1, help='训练模式')

    args = parser.parse_args()
    eqn_type = args.eqn_type
    cur_dim = args.cur_dim
    v_type = args.v_type
    drop_out = args.drop_out
    train_mode = args.train_mode
    eqn_config = EqnConfig(eqn_type=eqn_type, v_type=v_type, dim=cur_dim)
    
    set_seed(eqn_config.seed)
    redirect_log_file(exp_name=f"{eqn_type}_d{cur_dim}_drop{drop_out}")
    set_gpu_max_mem(default_device=0, force=False)
    
    if not train_mode:
        test_gen_dataset(eqn_config)
    else:
        test_train(eqn_config, v_type, drop_out)
    