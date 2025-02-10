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


def du_dt_(u, no, theta_0, x, t, dt):
    '''finite difference dtheta/dt.
    Inputs:
        u: instance of U.
        no: instance of NO.
        theta_0: torch.Tensor, shape (bs, n_param)
        x: torch.Tensor, shape (bs, n_x, d)
        t: torch.Tensor, shape (1,)
        dt: float, time step
    Outputs:
        du/dt: torch.Tensor, shape (bs, n_x)
        theta_t: torch.Tensor, shape (bs, n_param)
    '''
    bs = x.shape[0]
    t_rep = t.repeat(bs,)  #shape=(bs,)
    theta_t = no(theta_0, t_rep) #(bs, n_param)*(bs,) --> (bs, n_param) 
    u_t = u.forward_2(theta_t, x)  #shape=(bs, n_x)
    theta_t_dt = no(theta_0, t_rep+dt)
    u_t_dt = u.forward_2(theta_t_dt, x)
    du_dt = (u_t_dt - u_t) / dt
    return du_dt, theta_t

def loss_(theta_0, u, no):
    '''PINN loss'''
    # ---hyperparameters---
    bs = theta_0.shape[0] 
    n_x = 1000 #author mails that n_x=1000
    d=u.dim
    a=-1 #left boundary
    b=1 #right boundary
    T=0.1 #time horizon
    x = a + (b-a)*torch.rand((bs, n_x, d), device=theta_0.device)# quadrature points

    # ---residual loss---
    def ode_func(t, r):
        du_dt, theta_t = du_dt_(u, no, theta_0, x, t, dt=T/100)
        lhs = du_dt
        rhs = heat.laplacian(u, theta_t, x)  #shape=(bs, n_x)
        dr_dt = (lhs - rhs).pow(2).mean(dim=0).mean(dim=0,keepdim=True) * (b-a)**d
        return dr_dt + 0 * r #add dummy variable to avoid gradient error
    
    r_0 = torch.zeros(1).to(theta_0) # initial condition for r
    t = torch.tensor([0, T]).to(theta_0)  #time grid, start from a small positive value instead of 0 to avoid nan
    traj = odeint_adjoint(ode_func, y0=r_0, t=t, method='rk4', options={'step_size':1e-2}, adjoint_params=no.parameters())
    r_T = traj[-1]
    return r_T



def train(ckp_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = models.U(dim=10).to(device)
    no = NO3.NO3_1(dim=u.n_params, width=512, depth=2).to(device)
    print(f"heat_{u.__class__.__name__}_{no.__class__.__name__}_dim{u.dim}")
    

    bs = 128
    start_epoch = 0
    n_epoch = 1000
    lr = 5e-4
    T = 0.1
    optimizer = torch.optim.Adam(no.parameters(), lr=lr)

    if ckp_path:
        checkpoint = torch.load(ckp_path, map_location=device)
        no.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"checkpoint loaded from {ckp_path}")

    #---load data---
    dataset_train = heat.gen_train_dataset(u.n_params, n_1=100000, n_2=50000)
    dataload_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True)
    #dataset_test  = gen_dataset(u.n_params, n_1=0, n_2=100)
    test_bs = 200
    #dataset = torch.load(f'checkpoints/heat_dataset_dim10_size51454.pt')
    if u.dim == 10:
        dataset = torch.load('key_datasets/heat_dataset_dim10_params970_seed10.pt')
    elif u.dim == 15:
        dataset = torch.load('checkpoints/heat_dataset_dim15_params1375_seed10_12.pt')
    elif u.dim == 20:
        dataset = torch.load('checkpoints/heat_dataset_dim20_params1780_seed10_11.pt')
    elif u.dim == 5:
        dataset = torch.load('checkpoints/heat_dataset_dim5_params565_seed10_12.pt')
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
                r_T = loss_(theta_0, u, no)
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

            if j % 10 == 0:
                print(f"---- epoch {i}, batch {j} ----")
                print(f"r_T: {r_T.item()},")
                
                valid_l2re = NO3.eval(u, no, T, dataload_valid, device, nt=2)
                test_l2re = NO3.eval(u, no, T, dataload_test, device, nt=2)

                print("L2RE valid/test=", valid_l2re, test_l2re)
                print(f'walltime = {time.time()-start_time}', flush=True)

                if valid_l2re < best_ckp['valid_l2re']:
                    best_ckp['valid_l2re'] = valid_l2re
                    best_ckp['test_l2re'] = test_l2re
                    best_ckp['model_state_dict'] = copy.deepcopy(no.state_dict())
                    best_ckp['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())
                    print("best ckp updated")
                    model_path = f"checkpoints/heat_pino_dim{u.dim}.pth"
                    torch.save(best_ckp, model_path)



if __name__ == '__main__':
    redirect_log_file(exp_name="PINO")
    set_gpu_max_mem(default_device=1, force=False)
    set_seed(0)
    train(ckp_path="checkpoints/heat_pino_dim10.pth")