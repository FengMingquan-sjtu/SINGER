'''Physics-Informed Neural Operator (PINO) for solving Heat Equations'''
import time
import os
import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint, odeint

import models
from utils import redirect_log_file, timing, set_seed, set_gpu_max_mem
import heat
import NO3
import PINO
import hjb

def loss_(theta_0, u, no, eps = 1, T=1):
    '''PINN loss'''
    # ---hyperparameters---
    n_x = 5000  #NOTE: original:10000
    x, p_x = hjb.impt_sample(u, theta_0, n_x)  #shape=(bs, n_x, d), (bs, n_x)

    # ---residual loss---
    def ode_func(t, r):
        du_dt, theta_t = PINO.du_dt_(u, no, theta_0, x, t, dt=T/100)
        lhs = du_dt
        rhs = heat.laplacian(u, theta_t, x)  #shape=(bs, n_x)
        #HJB equation u_t = eps * Delta u - 1/2 * norm(grad u)**2  (Eq.27) #NOTE: eps > 0 
        rhs = eps * heat.laplacian(u, theta_t, x)  #shape=(bs, n_x)
        rhs -= 0.5 * hjb.du_dx_(u, theta_t, x).pow(2).sum(dim=-1)  #shape= (bs, n_x, dim)--> (bs, n_x)
        dr_dt = ((lhs - rhs).pow(2) / p_x).mean()
        return dr_dt + 0 * r #add dummy variable to avoid gradient error
    
    r_0 = torch.zeros(1).to(theta_0) # initial condition for r
    t = torch.tensor([0, T]).to(theta_0)  #time grid, start from a small positive value instead of 0 to avoid nan
    traj = odeint_adjoint(ode_func, y0=r_0, t=t, method='rk4', options={'step_size':T/10}, adjoint_params=no.parameters())
    r_T = traj[-1]
    return r_T


def eval_2(u, no, T, dataloader, device, nt=2):
    no.eval()

    with torch.no_grad():
        L2RE_list = []
        for data in dataloader:
            # --- inference ---
            theta_T = data[0].to(device).float()
            if nt <= 2:
                t = torch.ones(theta_T.shape[0], device=device) * T
                theta_0_pred = no(theta_T, t)
            else:
                t = torch.ones(theta_T.shape[0], device=device) * T / (nt-1)
                theta_0_pred = theta_T
                for _ in range(nt-1):
                    theta_0_pred = no(theta_0_pred, t)

            # --- test ---
            L2RE = hjb.test_3(u, data, theta_0_pred)
            L2RE_list.append(L2RE)
        return sum(L2RE_list)/len(L2RE_list)
    


def train(ckp_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = models.U_HJB(dim=15).to(device)
    no = NO3.NO3_1(dim=u.n_params, width=512, depth=2).to(device)
    print(f"hjb_{u.__class__.__name__}_{no.__class__.__name__}_dim{u.dim}")
    


    bs = 25  #original=100, adjust to fit GPU memory
    start_epoch = 0
    n_epoch = 1000
    lr = 1e-4  #original =1e-4
    print(f'{lr=}')
    T = 1
    eps = 1.
    optimizer = torch.optim.Adam(no.parameters(), lr=lr)

    #---load data---
    dataset_train = hjb.gen_train_dataset(n=10000, dim=u.dim, width=u.width, is_train=True)
    dataload_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True)

    test_bs = 200
    if u.dim == 10:
        dataset = torch.load('checkpoints/hjb_dataset_dim10_params1060_seed10.pt', map_location=device)
    elif u.dim == 5:
        dataset = torch.load('checkpoints/hjb_dataset_dim5_params555_seed10.pt', map_location=device)
    elif u.dim == 15:
        dataset = torch.load('checkpoints/hjb_dataset_dim15_params1565_seed10.pt', map_location=device)
    elif u.dim == 20:
        dataset = torch.load('checkpoints/hjb_dataset_dim20_params2070_seed10.pt', map_location=device)
    n = len(dataset)
    n_train = (n // 10) * 8
    n_valid = n // 10
    n_test = n - n_train - n_valid
    _, dataset_valid, dataset_test = torch.utils.data.random_split(dataset, [n_train, n_valid, n_test])
    dataload_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=test_bs, shuffle=False, drop_last=False,)
    dataload_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=test_bs, shuffle=False, drop_last=False,)

    start_time = time.time()
    best_ckp = {'model_state_dict':None, 'optimizer_state_dict':None, 'test_l2re':1e10, 'valid_l2re':1e10}
    for i in range(start_epoch+1, n_epoch+1):
        for j, data in enumerate(dataload_train):
            no.train()
            theta_0 = data[0].to(device)
            try:
                r_T = loss_(theta_0, u, no, eps, T)
                assert not r_T.isnan(), "r_T is nan"
                
                optimizer.zero_grad()
                loss = r_T
                loss.backward()
                optimizer.step()
            except AssertionError as e:
                print(f"epoch {i}, batch {j}, err:{e}", flush=True)
                print(f"r_T ={r_T.item()}")
                no.load_state_dict(best_ckp['model_state_dict'])
                optimizer.load_state_dict(best_ckp['optimizer_state_dict'])
                continue

            if j % 2 == 0:
                print(f"---- epoch {i}, batch {j} ----")
                print(f"r_T: {r_T.item()},")
                
                valid_l2re = eval_2(u, no, T, dataload_valid, device, nt=2)
                test_l2re = eval_2(u, no, T, dataload_test, device, nt=2)

                print("L2RE valid/test=", valid_l2re, test_l2re)
                print(f'walltime = {time.time()-start_time}', flush=True)

                if valid_l2re < best_ckp['valid_l2re']:
                    best_ckp['valid_l2re'] = valid_l2re
                    best_ckp['test_l2re'] = test_l2re
                    best_ckp['model_state_dict'] = copy.deepcopy(no.state_dict())
                    best_ckp['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())
                    print("best ckp updated")
                    model_path = f"checkpoints/hjb_pino_dim{u.dim}.pth"
                    torch.save(best_ckp, model_path)



if __name__ == '__main__':
    redirect_log_file(exp_name="PINO")
    set_gpu_max_mem(default_device=0, force=False)
    set_seed(0)
    train()