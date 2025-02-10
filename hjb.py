import time
import os
import copy 
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint, odeint

import models
from utils import redirect_log_file, timing, set_seed, set_gpu_max_mem
import heat
import NO3
from models import U_HJB




def hjb_solution(x, g, n_y=20000, bs=1, t=0, T=1, eps=1):
    r'''output u(x, t) = -2 eps*ln(\int (p(y)exp(-g(y)/(2eps(T-t))))dy), while p(y) = pdf of N(x;2*eps*(T-t)).
        following paper page 16 the 4-th formula.
        But in textbook [1] L. C. Evans, Partial differential equations. in Graduate studies in mathematics, no. v. 19. Providence, R.I: American Mathematical Society, 1998.
        p. 195, the denominator of g(y) is 2eps instead of 2eps(T-t). 
        also the variation of p is 2*eps*t, instead of 2*eps*(T-t).
        We follow the textbook.
    Inputs:
        x: (bs, N, d) or (N, d)
        g: terminal function (..., d) --> (...,)
        n_y: int, number of quadrature points
        bs: int, batch size
        t: float, time index
        T: float, time horizon
        eps: float, diffusion coefficent.
    Outputs: 
        u: (bs, N,)
    '''
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
        #y = y * math.sqrt(2*eps*(t)) + x   #scale and shift, (bs*N, n_y_bs, d)  #textbook version
        #exp_ = torch.exp(-g(y)/(2*eps*(T-t))) # (bs*N, n_y_bs)  #paper version
        exp_ = torch.exp(-g(y)/(2*eps)) # (bs*N, n_y_bs)  #textbook version
        int_ += exp_.mean(dim=-1)  # (bs*N)
    int_ /= (n_y//n_y_bs)
    out = -2*eps*torch.log(int_)  # (bs*N)  
    return out.view(x_shape[:-1])   # (bs, N)


def gen_train_dataset(n, dim=8, width=50, is_train=True):
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


def gen_test_dataset(n, u, T, eps):
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
    seed = 10
    set_seed(seed)
    bs = 1  #batch size, adjust to fit the GPU.
    n_x = 1000
    theta_T = gen_train_dataset(n=n+bs, dim=u.dim, width=u.width, is_train=False).tensors[0]
    n = len(theta_T)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = [[], [], [], []]
    for i in range(n//bs):
        theta_T_i = theta_T[i*bs:(i+1)*bs].to(device)
        x, p_x = impt_sample(u, theta_T_i, n_x)  # (bs, n_x, dim)
        def g(y): #batched u.
            theta_rep = theta_T_i.unsqueeze(1).repeat(1, n_x, 1).reshape(-1, u.n_params)
            return u.forward_2(theta_rep, y)
        u_0 = hjb_solution(x, g, n_y=20000, bs=bs, T=T, eps=eps) #(bs, n_x)

        test_dataset[0].append(theta_T_i)
        test_dataset[1].append(x)
        test_dataset[2].append(p_x)
        test_dataset[3].append(u_0) 
    test_dataset = [torch.cat(test_dataset[i], dim=0) for i in range(4)]
    test_dataset = torch.utils.data.TensorDataset(*test_dataset)
    filename = f"checkpoints/hjb_dataset_dim{u.dim}_params{u.n_params}_seed{seed}.pt"
    torch.save(test_dataset, filename)
    print("dataset generated, saved to ", filename)
    return test_dataset
        


def impt_sample(u_hjb, theta0, n_x):
    r''' Importance sampling, line 13 page 14 in the paper.
    \rho(x) = 1/width * \sum_{i=1}^{width} N(x-b_1; a_i ^{-2})
    Inputs:
        u_hjb: U_HJB
        theta0: torch.Tensor, shape (bs, n_params)
        n_x: int, number of samples,  preferably be a multiple of width.
    Outputs:
        x: torch.Tensor, shape (bs, n_x, dim)
        p_x: torch.Tensor, shape (bs, n_x)
    '''
    dim, width = u_hjb.dim, u_hjb.width
    bs = theta0.shape[0]
    c_size = dim
    a_size = dim * width
    b_size = dim * width
    b = theta0[:, c_size:c_size+b_size].reshape(bs, dim, width)
    a = theta0[:, c_size+b_size: c_size+b_size+a_size].reshape(bs, dim, width)
    N_x = math.ceil(n_x / width) * width
    m = N_x // width
    
    x = torch.randn(bs, dim, N_x, device=theta0.device) / a.repeat(1,1,m).abs() + b.repeat(1,1,m)
    x = x[:, :, :n_x] # (bs, dim, n_x)
    

    x = x.unsqueeze(2)   # (bs, dim, 1, n_x)
    mu = b.unsqueeze(-1)  #(bs, dim, width, 1)
    sigma = 1/a.abs().unsqueeze(-1) #(bs, dim, width, 1)
    # ref: https://cs229.stanford.edu/section/gaussians.pdf, Section3: The diagonal covariance matrix case
    p_x = (torch.exp(-(x - mu)**2/(2 * sigma**2) ) / (sigma * math.sqrt(2*math.pi))).prod(1).sum(1) / width  # (bs, dim, width, n_x) --> (bs, n_x)
    x = x.squeeze(2).permute(0, 2, 1)  # (bs, n_x, dim)
    return x, p_x  # (bs, n_x, dim), # (bs, n_x)


def test_3(u, data, pred_theta_0, normed=True):
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



def eval_2(u, v, dataloader, device, T=1, verbose=False):
    '''eval v based on pre-generated test_dataset.'''
    v.eval()
    with torch.no_grad():
        L2RE_list = []
        for data in dataloader:
            theta_T = data[0].to(device).float()
            pred_theta_0 = inference(theta_T, v, T=T)
            L2RE = test_3(u, data, pred_theta_0)
            L2RE_list.append(L2RE)

        if verbose: #print last batch
            print(f"l2RE_list={torch.tensor(L2RE_list)}")
        return sum(L2RE_list)/len(L2RE_list)


def inference(theta_0, v, T=1):
    
    def ode_func(t, theta):
        return v(theta) ##dtheta_dt, shape=(bs, n_param)
    t = torch.tensor([0, T]).to(theta_0)  #time grid
    #traj = odeint(ode_func, y0=theta_0, t=t, method='dopri5', rtol=1e-2, atol=1e-7)
    traj = odeint(ode_func, y0=theta_0, t=t, method='rk4', options={'step_size':T/10},)
    thetaT_pred = traj[-1]
    return thetaT_pred


def du_dx_(u, theta_rep, x):
    '''du/dx.
    Inputs:
        u: instance of U.
        theta_rep: torch.Tensor, shape (bs, n_x, n_param)
        x: torch.Tensor, shape (bs, n_x, d)
    Outputs:
        du/dx: torch.Tensor, shape (bs, n_x, d)
    '''
    with torch.enable_grad():
        x.requires_grad_()
        u_output = u.forward_2(theta_rep,x).sum()        
        du_dx = torch.autograd.grad(u_output, x)[0].detach()  #shape=(bs, n_x, d)
        return du_dx


def loss_(theta_0, u, v, eps = 1, T=1):
    '''loss (Eq.6)'''
    n_x = 5000  #NOTE: original:10000
    
    x, p_x = impt_sample(u, theta_0, n_x)  #shape=(bs, n_x, d), (bs, n_x)
    
    def ode_func(t, gamma):
        '''function used in odeint.
        Inputs:
            t: torch.Tensor, shape (1,) or (bs,)
            gamma: tuple of two torch.Tensor, theta and r, shape (bs, n_param) and (1,)
        '''
        
        theta, r = gamma
        dtheta_dt = v(theta) ##shape=(bs, n_param)

        #--- calculate dr_dt (Eq.5) via Importance Sampling---
        theta_rep = theta.unsqueeze(1).repeat(1, n_x,1)  #shape=(bs, n_x, n_param)
        du_dtheta = heat.du_dtheta_(u, theta_rep, x)  #shape=(bs, n_x, n_param)
        du_dt = du_dtheta * dtheta_dt.unsqueeze(1)  #shape=(bs, n_x, n_param)
        du_dt = du_dt.sum(-1)  #shape=(bs, n_x)
        lhs = du_dt
        
        #HJB equation u_t = eps * Delta u - 1/2 * norm(grad u)**2  (Eq.27) #NOTE: eps > 0 
        rhs = eps * heat.laplacian(u, theta_rep, x)  #shape=(bs, n_x)
        rhs -= 0.5 * du_dx_(u, theta_rep, x).pow(2).sum(dim=-1)  #shape= (bs, n_x, dim)--> (bs, n_x)
        dr_dt = ((lhs - rhs).pow(2) / p_x).mean()
        return (dtheta_dt, dr_dt)
    

    r_0 = torch.zeros(1).to(theta_0) # initial condition for r
    t = torch.tensor([0, T]).to(theta_0)  #time grid
    traj = odeint_adjoint(ode_func, y0=(theta_0, r_0), t=t, method='rk4', options={'step_size':T/10}, adjoint_params=v.parameters())
    theta_traj, r_traj = traj
    r_T = r_traj[-1] #final condition for r
    return r_T #shape = (1,)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = U_HJB(dim=10).to(device)  # target net
    #
    dropout_p = 0.1
    #v  = models.GnnV5(u, dropout_p=dropout_p, skip_connection=False, n_hiddens=200)
    v = models.V(dim=u.n_params, dropout_p=dropout_p).to(device) #node model
    v = v.to(device)

    # load datasets
    bs = 100  #batch size, original=100, adjust to fit the GPU.
    dataset_train = gen_train_dataset(n=10000, dim=u.dim, width=u.width, is_train=True)
    dataload_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True, drop_last=False,)

    if u.dim == 10:
        dataset_test = torch.load('checkpoints/hjb_dataset_dim10_params1060_seed10.pt', map_location=device)
    elif u.dim == 5:
        dataset_test = torch.load('checkpoints/hjb_dataset_dim5_params555_seed10.pt', map_location=device)
    elif u.dim == 15:
        dataset_test = torch.load('checkpoints/hjb_dataset_dim15_params1565_seed10.pt', map_location=device)
    elif u.dim == 20:
        dataset_test = torch.load('checkpoints/hjb_dataset_dim20_params2070_seed10.pt', map_location=device)
    n = len(dataset_test)
    n_test = (n // 10) * 6
    n_valid = n  - n_test
    dataset_test, dataset_valid = torch.utils.data.random_split(dataset_test, [n_test, n_valid])
    dataload_test = torch.utils.data.DataLoader(dataset_test, batch_size=n_test, shuffle=False, drop_last=False)
    dataload_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=n_valid, shuffle=False, drop_last=False)
    
    
    # train
    T = 1
    eps = 1.
    n_epoch = 30
    optimizer = torch.optim.Adam(v.parameters(), lr=1e-4)  #original lr=5e-4
    start_time = time.time()
    best_ckp = {'model_state_dict':None, 'optimizer_state_dict':None, 'valid_l2re':1e10, 'test_l2re':1e10}
    os.makedirs("checkpoints", exist_ok=True)

    model_path = f"checkpoints/hjb_{u.__class__.__name__}_{v.__class__.__name__}_dim{u.dim}_vdrop{v.dropout_p:.3f}.pth"
    print("model save at", model_path)
    print(f"num of trainable params={sum(p.numel() for p in v.parameters() if p.requires_grad)}, num of total params={sum(p.numel() for p in v.parameters())}")
    
    for i in range(n_epoch):
        v.train()
        for j, data in enumerate(dataload_train):
            theta_0 = data[0].to(device) #shape=(bs, n_param)
            loss = loss_(theta_0, u, v, T=T, eps=eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if j %2 == 0:
                print('----------------------')
                print(f'epoch {i}, batch {j}, loss {loss.item()}')
                valid_l2re = eval_2(u, v, dataload_valid, device, T=T)
                test_l2re = eval_2(u, v, dataload_test, device, T=T)

                print("L2RE valid/test=", valid_l2re, test_l2re)
                print(f'walltime = {time.time()-start_time}', flush=True)

                if valid_l2re < best_ckp['valid_l2re']:
                    best_ckp['valid_l2re'] = valid_l2re
                    best_ckp['test_l2re'] = test_l2re
                    best_ckp['model_state_dict'] = v.state_dict()
                    best_ckp['optimizer_state_dict'] = optimizer.state_dict()
                    print(f"best ckp updated")
                    torch.save(best_ckp, model_path)

   


# ---- unit test codes ----

def test_gen_test_data():
    T = 1
    eps = 1.
    print(f"{eps=}, {T=}")
    n = 200
    u = U_HJB(dim=20)
    gen_test_dataset(n, u, T, eps)

class DummyV(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,theta):
        return torch.zeros_like(theta)

def test_eval_2():
    device = torch.device('cuda')
    u = U_HJB(dim=10)
    #v = models.V(dim=u.n_params) #[0.0005, 0.0005, 0.0005, 0.0004]
    #v = models.GnnV5(u=u,skip_connection=True)#[0.1131, 0.1108, 0.1155, 0.1249]
    #v = models.GnnV5(u=u,skip_connection=False)#[0.0003, 0.0003, 0.0003, 0.0003]
    v = DummyV()  #eps=0.2, l2re=7.1690e-05,  eps=0.5, l2re=0.0014
    #dataset = torch.load('checkpoints/hjb_dataset_dim8_params858_seed10.pt', map_location=device)
    dataset = torch.load('checkpoints/hjb_dataset_dim10_params1060_seed10.pt', map_location=device)
    test_bs = 100
    dataloader = torch.utils.data.DataLoader(dataset,  batch_size=test_bs, shuffle=False, drop_last=False,)
    T = 1.0
    v = v.to(device)
    eval_2(u=u,v=v,dataloader=dataloader, device=device,T=T, verbose=True)

if __name__ == "__main__":
    set_seed(0)
    redirect_log_file(exp_name="hjb")
    set_gpu_max_mem()
    train()
    #test_gen_test_data()
    #test_eval_2()
    #test_eval()
    #test_ckp()