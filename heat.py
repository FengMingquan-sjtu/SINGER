from contextlib import contextmanager
import signal
import time
import math
import os 
import random
import copy
from functools import partial
import os
import tempfile
from pathlib import Path


import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint, odeint
import matplotlib.pyplot as plt

#--- ray tune--- # pip install -U "ray[data,train,tune,serve]"
#from ray import tune
#from ray.train import Checkpoint, get_checkpoint, report
#from ray.tune.schedulers import ASHAScheduler
#import ray.cloudpickle as pickle
# --- --- 

from models import U, V, GnnV5, U6, U3, U5
from utils import redirect_log_file, timing, save_load, set_seed, set_gpu_max_mem



#@timing
def fit_theta(theta0, u, n_x_outer, n_x_inner, theta_init=None):
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
    
    a=-1 #left boundary
    b=1 #right boundary
    

    def u0(y):
        theta0_rep = theta0.unsqueeze(1).repeat(1, n_x_outer, 1).reshape(-1, u.n_params) 
        return u.forward_2(theta0_rep, y)
    
    bs = theta0.shape[0]
    if not theta_init is None:
        theta = nn.Parameter(theta_init.clone())
    else:
        theta = nn.Parameter(theta0.clone())
    
    # minimize ||f(x) - u(x)||^2 by Adam
    optimizer = torch.optim.Adam([theta], lr=0.001)
    scaler = torch.cuda.amp.GradScaler()  #NOTE: use mixed precision training, save memory.

    if u.dim == 10 or u.dim==5:
        n_iter = 2000
    elif u.dim == 15:
        n_iter = 3000
    elif u.dim == 20:
        n_iter = 4000
    else:
        n_iter=3000

    for i in range(0,n_iter+1): #original:2001 for dim-10, 3001 for dim-15
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if i % 50 == 0: #update x every 50 iterations, save computation.
                x = a + (b-a)*torch.rand((n_x_outer, u.dim), device=theta0.device)
                true_sol = heat_solution_2(x, u0, n_x_inner, bs)  # (bs, n_x_outer)
            
            pred_sol = u.forward_2(theta, x)
            relative_error = ((true_sol - pred_sol)**2).mean(dim=1) / (true_sol**2).mean(dim=1)
            loss = relative_error.mean()
            #if i % 500 == 0:
            #    print(f"iter {i}, loss={loss.item()}", flush=1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return theta.detach(), relative_error.detach()   # (bs, n_params), (bs, )


def gen_test_dataset(u, seed=0, num_samples=250, l2re_threshold=1e-3, prev_dataset=None):
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
    set_seed(seed, cudnn_benchmark=True)
    n_params = u.n_params
    # Set bs, n_x_outer, n_x_inner according to GPU memory size.
    bs = 3  # 3 for 20GB, 11 for 80GB
    n_x_outer = 350  # 600 for 5-dim, 500 for 10-dim, 350 for 15-dim, 300 for 20-dim
    n_x_inner = 2500  #original: 2500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    theta_x_dataset = gen_train_dataset(n_params, n_1=0, n_2=50000)  # n_1=10000, n_2=5000 -->（15000, n_params）
    theta_x_dataloader = torch.utils.data.DataLoader(theta_x_dataset, batch_size=bs, shuffle=True, drop_last=True)


    # --- phase 1: generate initial values ---
    if not prev_dataset is None:
        dataset = torch.load(prev_dataset, map_location=device)
        theta_T = dataset.tensors[1]
        theta_T_init = theta_T[:bs]  #load the first batch as init.
    else:
        print("start generate theta_T_init.",flush=1)
        start_time = time.time()
        theta_T_init = None
        theta_T_list = []
        for i, theta_0 in enumerate(theta_x_dataloader):
            theta_0 = theta_0[0].to(device)
            theta_T, L2RE = fit_theta(theta_0, u, n_x_outer, n_x_inner, theta_init=theta_T_init)
            theta_T_init = theta_T  # use the optimized theta_T as the initial value for the next iter.
            mask = L2RE < l2re_threshold
            if mask.sum() > 0:
                theta_T_list.append(theta_T[mask])
            if len(theta_T_list) >= bs:
                theta_T_init = torch.cat(theta_T_list, dim=0)[:bs]
                break
        assert not theta_T_init is None
        print(f"theta_T_init.shape={theta_T_init.shape}, time={time.time()-start_time}")



    # --- phase 2: generate dataset ---
    print("start generate dataset...", flush=1)
    out_file = f'checkpoints/heat_dataset_dim{u.dim}_params{theta_0.shape[1]}_seed{seed}.pt'
    
    theta_0_list = list()
    theta_T_list = list()    
    cnt = 0
    start_time = time.time()
    for i, theta_0 in enumerate(theta_x_dataloader):
        theta_0 = theta_0[0].to(device)
        theta_T, L2RE = fit_theta(theta_0, u, n_x_outer, n_x_inner, theta_init=theta_T_init)
        msk = L2RE < l2re_threshold
        if msk.sum() > 0:
            theta_0_list.append(theta_0[msk])
            theta_T_list.append(theta_T[msk])
        cnt += msk.sum()
        if  (i+1)%100==0 or cnt >= num_samples:
            theta_0 = torch.cat(theta_0_list, dim = 0)
            theta_T = torch.cat(theta_T_list, dim = 0)
            dataset= torch.utils.data.TensorDataset(theta_0, theta_T)
            print(f'batch {i}, accept ratio = {cnt/((i+1)*bs)}, time = {time.time()-start_time}', flush=True)
            torch.save(dataset, out_file)
            if cnt >= num_samples:
                break
    print(f"generate dataset done, saved at {out_file}, num_sample {len(dataset)} time={time.time()-start_time}", flush=1)

def heat_solution_2(x, u0, n_x, bs=1, t=0.1):
    '''output u(x, t) = \int u0(y) N(y − x, 2t*I) dy, while N is density of gaussian
    Inputs:
        x: (bs, N, d) or (N, d)
        t: float
    Outputs: 
        u: (bs, N,)
    '''

    if x.dim() == 2:
        x = x.unsqueeze(0).repeat(bs, 1, 1)
    
    x_shape = x.shape
    x = x.view(-1, 1, x_shape[-1])   # (bs*N, 1, d)
    y = torch.randn((1, n_x, x_shape[-1]), device=x.device)   #(1, n_x, d)
    y = y * math.sqrt(2*t) + x   #scale and shift, (bs*N, n_x, d)  #NOTE: the sign should be positive, not negative!
    out =  u0(y).mean(dim=1) # ->(bs*N, n_x) -> (bs*N, )
    
    return out.view(x_shape[:-1])   # (bs, N)



def laplacian(u, theta_rep, x):
    '''div \cdot grad u.
    Inputs:
        u: instance of U.
        theta_rep: torch.Tensor, shape (bs, n_x, n_param)
        x: torch.Tensor, shape (bs, n_x, d)
    Outputs:
        div_u: torch.Tensor, shape (bs, n_x)
    '''
    with torch.enable_grad():
        x.requires_grad_()
        du_dx = torch.autograd.grad(u.forward_2(theta_rep,x).sum(), x, create_graph=True)[0] # shape=x.shape= (bs, n_x, d)
        du_dx = du_dx.sum(dim=(0,1)) #shape=(d,)
    

        div_u = 0
        for i in range(x.shape[-1]):
            du_dxi_dxi =  torch.autograd.grad(du_dx[i], x, create_graph=True)[0][...,i].detach()
            div_u += du_dxi_dxi ##shape=(bs, n_x)
    return div_u

def du_dtheta_(u, theta_rep, x):
    '''du/dtheta.
    Inputs:
        u: instance of U.
        theta_rep: torch.Tensor, shape (bs, n_x, n_param)
        x: torch.Tensor, shape (bs, n_x, d)
    Outputs:
        du/dtheta: torch.Tensor, shape (bs, n_x, n_param)
    '''
    with torch.enable_grad():
        theta_rep.requires_grad_()
        u_output = u.forward_2(theta_rep,x).sum()        
        du_dtheta = torch.autograd.grad(u_output, theta_rep)[0].detach()  #shape=(bs, n_x, n_param)
        return du_dtheta





def loss_(theta_0, u, v, i=0, j=0):
    '''loss (Eq.6)'''
    bs = theta_0.shape[0] 
    n_x = 1000 #author mails that n_x=1000
    d=u.dim
    a=-1 #left boundary
    b=1 #right boundary
    T=0.1 #time horizon 
    x = a + (b-a)*torch.rand((bs, n_x, d), device=theta_0.device)# quadrature points
    
    base_seed = torch.randint(0, 100000, (1,)).item()

    def ode_func(t, gamma):
        '''function used in odeint.
        Inputs:
            t: torch.Tensor, shape (1,) or (bs,)
            gamma: tuple of two torch.Tensor, theta and r, shape (bs, n_param) and (1,)
        '''
        
        theta, r = gamma
        
        g = torch.Generator(device=theta.device).manual_seed(base_seed + int(t.item()*10000))
        
        dtheta_dt = v(theta, generator=g)


        #--- calculate dr_dt (Eq.5) via Monte Carlo Integ---
        theta_rep = theta.unsqueeze(1).repeat(1, n_x,1)  #shape=(bs, n_x, n_param)  
        du_dtheta = du_dtheta_(u, theta_rep, x)  #shape=(bs, n_x, n_param)
        du_dt = du_dtheta * dtheta_dt.unsqueeze(1)  #shape=(bs, n_x, n_param) 
        du_dt = du_dt.sum(-1)  #shape=(bs, n_x)
        lhs = du_dt
        
        #heat equation u_t = Delta u (Eq.23)
        rhs = laplacian(u, theta_rep, x)  #shape=(bs, n_x)
        dr_dt = (lhs - rhs).pow(2).mean() * (b-a)**d    #  MC Integration result needs *= (b-a)**dim
        
        return (dtheta_dt, dr_dt)
    

    r_0 = torch.zeros(1).to(theta_0) # initial condition for r
    t = torch.tensor([0, T]).to(theta_0)  #time grid
    
    traj = odeint_adjoint(ode_func, y0=(theta_0, r_0), t=t, method='rk4', options={'step_size':1e-2}, adjoint_params=v.parameters())
    
     
    theta_traj, r_traj = traj
    theta_T = theta_traj[-1] #shape = (bs, n_param)
    theta_T_norm = torch.linalg.vector_norm(theta_T, ord=2, dim=-1).mean()#shape =  (bs, n_param)-->(bs,)-->(1,)
    r_T = r_traj[-1] #final condition for r
    return r_T, theta_T_norm #shape = (1,),   (1,)


## [copied from original codes]
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

def train(u, v, ckp_path=None, is_rand_u = False, u_label=None, is_para=False):
    '''main function.'''
    model_path = f"checkpoints/heat_{u.__class__.__name__}_{v.__class__.__name__}_dim{u.dim}_vdrop{v.dropout_p:.3f}.pth"
    print("model save at", model_path)
    print(f"num of trainable params={sum(p.numel() for p in v.parameters() if p.requires_grad)}, num of total params={sum(p.numel() for p in v.parameters())}")
    bs = 64   #old= 64
    start_epoch = 0
    n_epoch = 1000
    lr = 5e-4  #old= 5e-4
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
    dataset_train = gen_train_dataset(u.n_params, n_1=100000, n_2=50000)
    dataload_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True, drop_last=False)
    
    if u.dim == 10:
        #dataset_name = 'key_datasets/heat_dataset_dim10_seed10.pt' # NOTE: original: use this filetered dataset
        dataset_name ='key_datasets/heat_dataset_dim10_params970_seed10.pt'
    elif u.dim == 15:
        dataset_name = 'checkpoints/heat_dataset_dim15_params1375_seed10_12.pt'
    elif u.dim == 20:
        dataset_name = 'checkpoints/heat_dataset_dim20_params1780_seed10_11.pt'
    elif u.dim == 5:
        dataset_name = 'checkpoints/heat_dataset_dim5_params565_seed10_12.pt'
    print(f"---load dataset from {dataset_name}---")
    dataset_test = torch.load(dataset_name)  
    n = len(dataset_test)
    n_test = (n // 10) * 7
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
    
    for i in range(start_epoch+1, n_epoch+1):
        for j, data in enumerate(dataload_train):
            v.train()
            theta_0 = data[0].to(device)
            try:
                #old_u_witdh = u.width
                #u.width = torch.randint(60, 80, (1,)).item()  #set random width, 80 exclusive
                #old_u_dim = u.dim
                #u.dim = torch.randint(5, 10, (1,)).item()  #set random dim, 10 exclusive
                r_T, theta_T_norm  = loss_(theta_0, u, v, i, j)
                assert not r_T.isnan(), "r_T is nan"
                assert not r_T.isinf(), "r_T is inf"
                optimizer.zero_grad()
                loss = r_T
                loss.backward()
                optimizer.step()
                #u.width = old_u_witdh #reset width
                #u.dim = old_u_dim # reset dim
                    
            except AssertionError as e:  #restart from last checkpoint if error
                print(f"epoch {i}, batch {j}, err:{e}", flush=True)
                print(f"r_T, theta_T_norm={r_T.item(), theta_T_norm.item()}")
                v.load_state_dict(best_ckp['model_state_dict'])
                optimizer.load_state_dict(best_ckp['optimizer_state_dict'])
                
                continue
            
            
            if j % 20 == 0:
                print(f"---- epoch {i}, batch {j} ({j/len(dataload_train):.4f}) ----")
                print(f"r_T: {r_T.item()}, theta_T_norm: {theta_T_norm.item()},")
                valid_l2re = eval(u, v, dataload_valid,  device, u_label=u_label)
                test_l2re = eval(u, v, dataload_test,  device, u_label=u_label)
                print("Valid L2RE_l_p=",  valid_l2re, "Test L2RE_l_p=",  test_l2re)
                print(f"wall time(hours)={(time.time()-start_time)/3600:.4f}" , flush=True)

                if valid_l2re < best_ckp['valid_l2re'] :
                    best_ckp['valid_l2re'] = valid_l2re
                    best_ckp['model_state_dict'] = copy.deepcopy(v.state_dict())  #deepcopy for recovery, in case loss=nan.
                    best_ckp['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict()) 
                    print(f"best ckp updated", flush=True)
                    torch.save(best_ckp, model_path)





def test_2(u, thetaT_label, thetaT_pred, normed=True, bs=-1, nx=5000, u_label=None):
    '''function distance between label and prediciton'''
    n = len(thetaT_pred)
    bs = min(5, n) if bs == -1 else bs
    u_label = u if u_label is None else u_label
    a=-1 #left boundary
    b=1 #right boundary

    x = a + (b-a)*torch.rand((nx, u.dim), device=thetaT_pred.device)
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


def test_3(u, theta0, thetaT_pred, normed=True, T=0.1):
    '''function distance between true_sol and prediciton at time t.'''
    n = len(thetaT_pred)
    a=-1 #left boundary
    b=1 #right boundary
    bs = 2  # 3 for 20GB, 11 for 80GB
    n_x_outer = 500  # 600 for 5-dim, 500 for 10-dim, 350 for 15-dim, 300 for 20-dim
    n_x_inner = 2500  #original: 2500
    
    x = a + (b-a)*torch.rand((n_x_outer, u.dim), device=thetaT_pred.device)
    err_list = []
    for i in range(n//bs):
        theta0_i = theta0[i*bs:(i+1)*bs]
        thetaT_pred_i = thetaT_pred[i*bs:(i+1)*bs]

        def u0(y):
            theta0_rep = theta0_i.unsqueeze(1).repeat(1, n_x_outer, 1).reshape(-1, u.n_params) 
            return u.forward_2(theta0_rep, y)
        label_sol = heat_solution_2(x, u0, n_x_inner, bs, t=T)  # (bs, n_x_outer)
        pred_sol = u.forward_2(thetaT_pred_i, x)

        if normed:
            err = (((label_sol - pred_sol)**2).mean(dim=1) / (label_sol**2).mean(dim=1)).mean()
        else:
            err = ((label_sol - pred_sol)**2).mean() * (b-a)**u.dim
        err_list.append(err)
    
    return sum(err_list)/len(err_list)

def eval(u, v, dataloader, device, verbose=False, u_label=None):
    '''evaluate the model on the test dataset.'''
    v.eval()
    is_diff_u = False
    if u_label is None:
        u_label = u
    elif u_label.__class__.__name__ != u.__class__.__name__ or u_label.n_params != u.n_params or u_label.width != u.width:
        is_diff_u = True
        
    with torch.no_grad():
        L2RE_list = []
        for theta0_label, thetaT_label in dataloader:
            theta0_label = theta0_label.to(device)
            thetaT_label = thetaT_label.to(device)
            if is_diff_u:
                thetaT_pred = inference_2(theta0_label, u_label, u, v)
            else:
                thetaT_pred = inference(theta0_label, v) #NOTE: original: use simple inference
            
            L2RE = test_2(u, thetaT_label, thetaT_pred, u_label=u_label)
            L2RE_list.append(L2RE)

        if verbose: #print at last batch
            print(f"l2RE_list={torch.tensor(L2RE_list)}")
        return sum(L2RE_list)/len(L2RE_list)
    
def _get_theta_0(theta_0_label, u_label, u):
    '''convert theta_0_label (within u_label) to theta_0(within u)'''
    for theta_0_label_cache, theta_0_cache in _get_theta_0.cache.items():
        if torch.equal(theta_0_label, theta_0_label_cache):
            return theta_0_cache
    
    print("initial fit of theta_0 (inference_2)")
    with torch.enable_grad():
        bs = theta_0_label.shape[0]
        theta_0 = torch.randn(bs, u.n_params).to(theta_0_label) *math.sqrt(0.5) #   *math.sqrt(0.5) for NODE.
        theta_0.requires_grad_(True)
        optimizer_name = 'LBFGS'

        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam([theta_0], lr=1e-1, weight_decay=1e-5)  #lr = 1e-1, n_iter= 400 for NODE.
            for i in range(1,1001):
                optimizer.zero_grad()
                loss = test_2(u, theta_0_label, theta_0, normed=True, u_label=u_label)
                loss.backward()
                optimizer.step()
                if i % 50 == 0:
                    print(f"init loss={loss.item()}", flush=1)
        elif optimizer_name == 'LBFGS':
            optimizer = torch.optim.LBFGS([theta_0], lr=1)
            for i in range(1, 51):
                def closure():
                    optimizer.zero_grad()
                    loss = test_2(u, theta_0_label, theta_0, normed=True, u_label=u_label)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                if i % 10 == 0:
                    print(f"init loss={closure().item()}", flush=1)
    
    theta_0 = theta_0.detach()
    _get_theta_0.cache[theta_0_label] = theta_0
    #print(theta_0[0].tolist())
    return theta_0
_get_theta_0.cache = {}

def inference_2(theta_0_label, u_label, u, v):
    ''' inference when u_label and u are different.
    Need convert theta_0_label (within u_label) to theta_0(within u)
    '''
    theta_0 = _get_theta_0(theta_0_label, u_label, u)
    return inference(theta_0, v)


def inference(theta_0, v, T=0.1):
    '''Simple ODE forward.'''
    
    
    def ode_func(t, theta):
        return v(theta) ##dtheta_dt, shape=(bs, n_param)
        
    
    t = torch.tensor([0, T]).to(theta_0)  #time grid
    #traj = odeint(ode_func, y0=theta_0, t=t, method='dopri5', rtol=1e-2, atol=1e-7)
    traj = odeint(ode_func, y0=theta_0, t=t, method='rk4', options={'step_size':T/10})
    thetaT_pred = traj[-1] 

    return thetaT_pred


# ---- unit test codes -----

def test_train():
    dim = 5 #original:10
    is_gnn_v_5 = True
    u = U(dim=dim, width=80) 
    #u = U6(dim=dim, width=40, width_1=40)
    #u = U3(dim=dim, width=80)
    #u = U5(dim=dim, width=80)
    #u = U(dim=dim, width=90) 
    print(f"u.type={u.__class__.__name__}, {u.width=}")
    v_width = 1000
        
    dropout_p=0.1  #original: 0.1
    print(f"{dropout_p=}")
        
    if is_gnn_v_5:
        n_hiddens = 150 if dim == 20 else 100
        v = GnnV5(u, dropout_p=dropout_p, n_hiddens=n_hiddens)
    else: # MLP V
        v = V(dim=u.n_params, dropout_p=dropout_p, width=v_width)

    #ckp_path = "checkpoints/heat_NODE_dim10.pth"
    ckp_path = None
    u_label = U(dim=dim)
    train(u, v, ckp_path=ckp_path, u_label=u_label)

def test_gen_dataset():
    u = U(dim=10, width=105)
    seed = 10
    l2re_threshold=1e-3  #1e-3 for 5, 10,15 dims, 2e-3 for 20 dims.
    gen_test_dataset(u, seed=seed, num_samples=20, l2re_threshold=l2re_threshold)




   





    

def concat_datasets():
    '''concatenate datasets of different seeds.'''
    dataset_list = []
    #for seed in [10,11,12]:
    #    dataset_name = f'checkpoints/heat_dataset_dim15_params1375_seed{seed}.pt'
    #for seed in [10,11]:
    #    dataset_name = f'checkpoints/heat_dataset_dim20_params1780_seed{seed}.pt'
    for seed in [10, 11, 12]:
        dataset_name = f'checkpoints/heat_dataset_dim5_params565_seed{seed}.pt'
        dataset = torch.load(dataset_name, map_location='cpu')
        dataset_list.append(dataset)
    dataset = torch.utils.data.ConcatDataset(dataset_list)
    print(f"num of samples={len(dataset)}")
    torch.save(dataset, 'checkpoints/heat_dataset_dim5_params565_seed10_12.pt')

def test_ckp():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dropout_p=0.1
    u = U(dim=10)
    n_hiddens = 150 if u.dim == 20 else 100
    #v = GnnV5(u, dropout_p=dropout_p, n_hiddens=n_hiddens).to(device)
    #ckp_path = f"checkpoints/heat_U_GnnV5_dim{u.dim}_vdrop0.100.pth"
    v = V(dim=u.n_params).to(device)
    ckp_path = "checkpoints/heat_U_V_dim10_drop0.100.pth"
    model_state = torch.load(ckp_path, map_location=device)['model_state_dict']
    v.load_state_dict(model_state)
    dataset_name ='key_datasets/heat_dataset_dim10_params970_seed10.pt'
    #dataset_name = 'checkpoints/heat_dataset_dim15_params1375_seed10_12.pt'
    dataset_test = torch.load(dataset_name)  
    n = len(dataset_test)
    n_test = (n // 10) * 7
    n_valid = n  - n_test
    dataset_test, dataset_valid = torch.utils.data.random_split(dataset_test, [n_test, n_valid])
    dataload_test = torch.utils.data.DataLoader(dataset_test, batch_size=n_test, shuffle=False, drop_last=False)
    dataload_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=n_valid, shuffle=False, drop_last=False)

    valid_l2re = eval(u, v, dataload_valid,  device)
    test_l2re = eval(u, v, dataload_test,  device)
    print("Valid L2RE_l_p=",  valid_l2re, "Test L2RE_l_p=",  test_l2re)



if __name__ == '__main__':
    
    redirect_log_file()
    set_seed(1)
    set_gpu_max_mem(1, force=False)
    #concat_datasets()
    test_train()
    #test_ckp()
    #test_perm_invariant()
    #test_gen_dataset()
    