import math
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

import models
from utils import redirect_log_file, timing, set_seed, set_gpu_max_mem
import heat


class NO3(nn.Module):
    def __init__(self, dim = 970, width=1000, depth=5, sigma=0.5, T=0.1):
        super(NO3, self).__init__()
        self.F = models.ResNet(dim+1, dim, width, depth,  F.relu)
        self.sigma = sigma
        self.T=T
    
    def forward(self, theta, t):
        ''' predict theta(t) given theta(0).
        Inputs:
            theta: torch.Tensor, shape (bs, dim) or (nt, bs, dim)
            t: torch.Tensor, shpae (bs, ) or (nt, bs,)
        Outputs:
            theta_t: torch.Tensor, shape same as input theta
        '''
        input_theta_shape = theta.shape
        
        if theta.dim() > 2:
            theta = theta.reshape(-1, theta.shape[-1])
            t = t.reshape(-1)
        t = t.unsqueeze(-1)#shape=(bs, 1) or (nt*bs, 1)
        data = torch.cat([theta, t], dim=-1)  #shape=(bs, dim+1) or (nt*bs, dim+1)

        # appendix C (page 25) of Consistency Model paper.
        t =  t/self.T #normalize t to [0,1]
        c_skip = self.sigma**2 / (self.sigma**2 + t**2)
        c_out  = (1-c_skip)**0.5 * self.sigma
        theta_t = theta * c_skip + self.F(data) * c_out  #when t=0, c_skip=1, c_out=0, theta_t=theta.
        theta_t = theta_t.reshape(input_theta_shape)
        return theta_t

class NO3_1(nn.Module):
    def __init__(self, dim = 970, width=1000, depth=5, sigma=0.5, T=0.1):
        super(NO3_1, self).__init__()
        self.V = models.V(dim)
        self.E = models.ResNet(1, dim, width, depth,  F.silu)
        
        self.sigma = sigma
        self.T=T
    
    def forward(self, theta, t):
        ''' predict theta(t) given theta(0).
        Inputs:
            theta: torch.Tensor, shape (*bs, n_param)
            t: torch.Tensor, shpae (*bs, )
        Outputs:
            theta_t: torch.Tensor, shape same as input theta
        '''
        input_theta_shape = theta.shape
        
        if theta.dim() > 2:
            theta = theta.reshape(-1, theta.shape[-1])
            t = t.reshape(-1)
        t = t.unsqueeze(-1) #shape=(*bs, 1)

        emb_t = self.E(t) #shape=(*bs, n_param)
        data = F.silu(theta + emb_t)

        # appendix C (page 25) of Consistency Model paper.
        t =  t/self.T #normalize t to [0,1]
        c_skip = self.sigma**2 / (self.sigma**2 + t**2)
        c_out  = (1-c_skip)**0.5 * self.sigma
        theta_t = theta * c_skip + self.V(data) * c_out  #when t=0, c_skip=1, c_out=0, theta_t=theta.
        theta_t = theta_t.reshape(input_theta_shape)
        return theta_t


class NO3_2(nn.Module):
    def __init__(self, dim = 970, width=1000, depth=5, sigma=0.5, T=0.1, time_scale=1):
        super(NO3_2, self).__init__()
        self.V = models.V(dim)
        self.E = models.ResNet(1, dim, width, depth,  F.silu)
        
        self.sigma = sigma
        self.T=T
        self.time_scale=time_scale
    
    def forward(self, theta, t):
        ''' predict theta(t) given theta(0).
        Inputs:
            theta: torch.Tensor, shape (*bs, n_param)
            t: torch.Tensor, shpae (*bs, )
        Outputs:
            theta_t: torch.Tensor, shape same as input theta
        '''
        input_theta_shape = theta.shape
        
        if theta.dim() > 2:
            theta = theta.reshape(-1, theta.shape[-1])
            t = t.reshape(-1)
        t = t.unsqueeze(-1) #shape=(*bs, 1)
        t = t/self.time_scale #-- compared with NO3_1, time scaling.

        emb_t = self.E(t) #shape=(*bs, n_param)
        data = F.silu(theta + emb_t)

        # appendix C (page 25) of Consistency Model paper.
        t =  t/self.T #normalize t to [0,1]
        c_skip = self.sigma**2 / (self.sigma**2 + t**2)
        c_out  = (1-c_skip)**0.5 * self.sigma
        theta_t = theta * c_skip + self.V(data) * c_out  #when t=0, c_skip=1, c_out=0, theta_t=theta.
        theta_t = theta_t.reshape(input_theta_shape)
        return theta_t
    
def ode_fwd(theta_0, v, T=0.1, nt=10, adaptive=True):
    '''
    Inputs:
        theta_0: torch.Tensor, shape (bs, n_param)
        v: torch.nn.Module
        T: float, final time
        nt: int, number of time steps
    Outputs:
        traj: torch.Tensor, shape=(nt, bs, n_param)
        t: torch.Tensor, shape=(nt,)
    '''
    def ode_func(t, theta):
        return v(theta) ##dtheta_dt, shape=(bs, n_param)
    if nt <=2:
        t = torch.tensor([0, T], device=theta_0.device)  #time grid
    else:
        t = torch.linspace(0, T, nt, device=theta_0.device)  #time grid, shape (nt,)
    if adaptive:
        traj = odeint(ode_func, y0=theta_0, t=t, method='dopri5', rtol=1e-2, atol=1e-7) #shape=(nt, bs, n_param)
    else:
        traj = odeint(ode_func, y0=theta_0, t=t, method='rk4', options={'step_size':1e-2},) #shape=(nt, bs, n_param)
    return traj, t  #drop the zero time.

def data_aug_0(theta_traj, t):
    '''
    Inputs:
        theta_traj: torch.Tensor, shape=(nt, bs, n_param)
        t: torch.Tensor, shape=(nt,)  linspace(T).
    Outputs:
        theta_0: torch.Tensor, shape=(nt-1, bs, n_param)
        theta_t: torch.Tensor, shape=(nt-1, bs, n_param)
        dt: torch.Tensor, shape=(nt-1,)
    '''
    T = t[-1]
    dt = T - t
    dt = dt[:-1]
    theta_0 = theta_traj[:-1]
    theta_T = theta_traj[-1]
    theta_t = theta_T.unsqueeze(0).repeat(len(dt), 1, 1)
    return theta_0, theta_t, dt

def data_aug_1(theta_traj, t):
    '''
    Inputs:
        theta_traj: torch.Tensor, shape=(nt, bs, n_param)
        t: torch.Tensor, shape=(nt,)  linspace(T).
    Outputs:
        theta_0: torch.Tensor, shape=(nt*(nt-1)/2, bs, n_param)
        theta_t: torch.Tensor, shape=(nt*(nt-1)/2, bs, n_param)
        dt: torch.Tensor, shape=(nt*(nt-1)/2,)
    '''
    theta_0_list = []
    theta_t_list = []
    dt_list = []
    nt = theta_traj.shape[0]
    for i in range(nt-1):
        theta_0_list.append(theta_traj[i:i+1].repeat(nt-i-1, 1, 1))
        theta_t_list.append(theta_traj[i+1:])
        dt_list.append(t[i+1:] - t[i])
    
    theta_0 = torch.cat(theta_0_list, dim=0)
    theta_t = torch.cat(theta_t_list, dim=0)
    dt = torch.cat(dt_list, dim=0)
    return theta_0, theta_t, dt


def data_aug_2(theta_traj, t):
    '''
    Inputs:
        theta_traj: torch.Tensor, shape=(nt, bs, n_param)
        t: torch.Tensor, shape=(nt,)  linspace(T).
    Outputs:
        theta_0: torch.Tensor, shape=(nt-1, bs, n_param)
        theta_t: torch.Tensor, shape=(nt-1, bs, n_param)
        dt: torch.Tensor, shape=(nt-1,)
    '''
    dt = torch.ones_like(t)[1:] * (t[1] - t[0])
    theta_0 = theta_traj[:-1]
    theta_t = theta_traj[1:]
    return theta_0, theta_t, dt


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = models.U(dim=10).to(device)  # target net
    v = models.V(dim=u.n_params).to(device) #node model
    no = NO3_1(dim=u.n_params, width=512, depth=2).to(device)  # neural operator model

    # load pre-trained node model v
    ckp_path = "checkpoints/heat_model_dim10_ep2_ba100.pth"  #L2RE=0.0372
    checkpoint = torch.load(ckp_path, map_location=device)
    v.load_state_dict(checkpoint['model_state_dict'])
    #no.V.load_state_dict(v.state_dict()) #load pre-trained model v to no.V
    

    

    # load datasets
    bs = 48
    test_bs = 200
    dataset = torch.load(f'checkpoints/heat_dataset_dim10_size51454.pt')
    n = len(dataset)
    n_train = (n // 10) * 8
    n_valid = n // 10
    n_test = n - n_train - n_valid
    dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(dataset, [n_train, n_valid, n_test])
    #print("train with 50000 filtered data points.", flush=True)
    print("train with 150000 non-filtered data points.", flush=True)
    dataset_train = heat.gen_dataset(u.n_params, n_1=100000, n_2=50000)
    dataload_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True, drop_last=True,)
    dataload_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=test_bs, shuffle=False, drop_last=True,)
    dataload_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=test_bs, shuffle=False, drop_last=True,)
    
    

    
    # train
    T = 0.1
    nt = 41
    n_epoch = 30
    optimizer = torch.optim.Adam(no.parameters(), lr=1e-3)
    start_time = time.time()
    best_ckp = {'model_state_dict':None, 'optimizer_state_dict':None, 'valid_l2re':1e10, 'test_l2re':1e10}
    
    #valid_l2re = NO.eval(u, v, dataload_valid, device, nt=nt)
    #test_l2re = NO.eval(u, v, dataload_test, device, nt=nt)
    #print("Teacher model L2RE train/valid/test=", valid_l2re, test_l2re)  #0.0369/0.0371
    
    for i in range(n_epoch):
        no.train()
        for j, data in enumerate(dataload_train):
            theta0 = data[0].to(device) #shape=(bs, n_param)
            theta_traj, t = ode_fwd(theta0, v, T, nt) #theta_traj=(nt, bs, n_param), t=(nt,)
            t = t.unsqueeze(-1).repeat(1, theta0.shape[0])  #shape=(nt,) -> (nt, bs)
           
            theta_0, theta_t, dt = data_aug_0(theta_traj, t) #(nt,...) -> (f(nt),...)
            theta_t_pred = no(theta_0, dt) #shape=(f(nt), bs, n_param)
           
            
                
            loss = heat.test_2(u, theta_t.reshape(-1, u.n_params), theta_t_pred.reshape(-1, u.n_params), 
                                normed=False, bs=bs, nx=1000)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if j % 100 == 0:
                print('----------------------')
                print(f'epoch {i}, batch {j}, loss {loss.item()}')
                valid_l2re = eval(u, no, T, dataload_valid, device, nt=2)
                test_l2re = eval(u, no, T, dataload_test, device, nt=2)

                print("L2RE valid/test=", valid_l2re, test_l2re)
                print(f'walltime = {time.time()-start_time}', flush=True)

                if valid_l2re < best_ckp['valid_l2re']:
                    best_ckp['valid_l2re'] = valid_l2re
                    best_ckp['test_l2re'] = test_l2re
                    best_ckp['model_state_dict'] = no.state_dict()
                    best_ckp['optimizer_state_dict'] = optimizer.state_dict()
                    print(f"best ckp updated")
                    model_path = f"checkpoints/heat_no31_dim{u.dim}.pth"
                    torch.save(best_ckp, model_path)

                

def inference(theta0, no, T=0.1, nt=2):
    if nt <= 2:
        t = torch.ones(theta0.shape[0], device=theta0.device) * T
        thetaT_pred = no(theta0, t)
    else:
        t = torch.ones(theta0.shape[0], device=theta0.device) * T / (nt-1)
        thetaT_pred = theta0
        for _ in range(nt-1):
            thetaT_pred = no(thetaT_pred, t)
    return thetaT_pred

def eval(u, no, T, dataloader, device, nt=2):
    no.eval()

    with torch.no_grad():
        L2RE_list = []
        for theta0, thetaT_label in dataloader:
            theta0 = theta0.to(device)
            thetaT_label = thetaT_label.to(device)
            thetaT_pred = inference(theta0, no, T, nt)
            L2RE = heat.test_2(u, thetaT_label, thetaT_pred)
            L2RE_list.append(L2RE)
        return sum(L2RE_list)/len(L2RE_list)
    



def test_data_aug():
    
    '''
    Inputs:
        theta_traj: torch.Tensor, shape=(nt, bs, n_param)
        t: torch.Tensor, shape=(nt,)  linspace(T).
    Outputs:
        theta_0: torch.Tensor, shape=(nt*(nt-1)/2, bs, n_param)
        theta_t: torch.Tensor, shape=(nt*(nt-1)/2, bs, n_param)
        dt: torch.Tensor, shape=(nt*(nt-1)/2,)
    '''
    bs = 1
    n_param = 1
    nt = 4
    theta_traj = torch.tensor(range(nt)).reshape(nt, bs, n_param).float()
    t = torch.linspace(0, 1, nt)
    theta_0, theta_t, dt = data_aug(theta_traj, t)
    print(f'{theta_0=}')
    print(f'{theta_t=}')
    print(f'{dt=}')


def time_benchmark():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = models.U(dim=10).to(device)  # target net
    v = models.V(dim=u.n_params).to(device) #node model
    v.eval()
    # load pre-trained node model v
    ckp_path = "checkpoints/heat_model_dim10_ep2_ba100.pth"  #L2RE=0.0372
    checkpoint = torch.load(ckp_path, map_location=device)
    v.load_state_dict(checkpoint['model_state_dict'])
    no = NO3_1(dim=u.n_params, width=512, depth=2).to(device)  # neural operator model
    ckp_path = "checkpoints/heat_no31_dim10.pth"
    checkpoint = torch.load(ckp_path, map_location=device)
    no.load_state_dict(checkpoint['model_state_dict'])
    no.eval()

    bs = 1000
    dataset = heat.gen_dataset(u.n_params, n_1=100000, n_2=50000)
    dataload = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=True,)

    T = 0.1
    n_epoch = 5
    nt = 2
    
    start_time = time.time()
    with torch.no_grad():
        for i in range(n_epoch):
            for j, data in enumerate(dataload):
                theta0 = data[0].to(device)
                ode_fwd(theta0, v, T, nt)
    print(f"teacher ode_fwd time={time.time()-start_time}")

    
    start_time = time.time()
    with torch.no_grad():
        for i in range(n_epoch):
            for j, data in enumerate(dataload):
                theta0 = data[0].to(device) #shape=(bs, n_param)
                t = torch.ones(theta0.shape[0], device=device) * T
                no(theta0, t)
    print(f"student no time={time.time()-start_time}")

def test_ckp_(T = 0.1):
    import NO
    test_bs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = models.U(dim=10).to(device)  # target net
    #no = models.NO5().to(device)  # neural operator model
    no = NO3_1(dim=u.n_params, width=512, depth=2).to(device)  # neural operator model
    ckp_path = "checkpoints/heat_no31_dim10.pth"
    checkpoint = torch.load(ckp_path, map_location=device)
    no.load_state_dict(checkpoint['model_state_dict'])
    no.eval()
    
    dataset_test = heat.gen_dataset(u.n_params, n_1=0, n_2=100)
    dataload_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=test_bs, shuffle=False, drop_last=False,)
    l2re = NO.eval_(u, no, T, dataload_test, device)
    print(f"{T=}, {l2re=}", flush=True)

if __name__ == '__main__':
    #redirect_log_file(exp_name="NO3")
    set_gpu_max_mem()
    set_seed(0, cudnn_benchmark=True)
    test_ckp_()
    #train()
    
    #time_benchmark()
    #test_data_aug()