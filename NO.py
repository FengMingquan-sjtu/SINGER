import math
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint, odeint

import models
from utils import redirect_log_file, timing, set_seed, set_gpu_max_mem
import heat





#@timing
def fit_theta(theta0, u, n_x_outer, n_x_inner, theta_init=None):
    '''
    Inputs:
        theta0: (bs, n_params)
        u: instance of models.U
        n_x: number of quadrature points
        theta_init:  initial value of theta
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
    scaler = torch.cuda.amp.GradScaler()

    
    #true_sol = sin_mul_func(x)
    for i in range(0,3001): #original:2001
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if i % 50 == 0:
                x = a + (b-a)*torch.rand((n_x_outer, u.dim), device=theta0.device)
                true_sol = heat.heat_solution_2(x, u0, n_x_inner, bs)  # (bs, n_x_outer)
                #true_sol = sin_mul_func(x)
            
            pred_sol = u.forward_2(theta, x)
            relative_error = ((true_sol - pred_sol)**2).mean(dim=1) / (true_sol**2).mean(dim=1)
            loss = relative_error.mean()
        
        #loss.backward()
        #optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #if i % 50 == 0:
        #    print(f'iteration {i}, loss {loss.item()}', flush=1)
    return theta.detach(), relative_error.detach()   # (bs, n_params), (bs, )


def gen_dataset(u, seed=0):
    set_seed(seed, cudnn_benchmark=True)
    n_params = u.n_params
    bs = 2
    n_x_outer = 350  #original: 500
    n_x_inner = 2500  #original: 2500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    theta_x_dataset = heat.gen_dataset(n_params, n_1=0, n_2=50000)  # n_1=10000, n_2=5000 -->（15000, n_params）
    theta_x_dataloader = torch.utils.data.DataLoader(theta_x_dataset, batch_size=bs, shuffle=False, drop_last=True)

    # --- load previous dataset as initial guess ---
    #dataset = torch.load(f'key_datasets/heat_dataset_dim{u.dim}_seed10.pt', map_location=device)
    #dataset = torch.load(f'checkpoints/heat_dataset_dim15_seed21173.pt', map_location=device)
    dataset = torch.load(f'checkpoints/heat_dataset_dim10_params2130_seed3124428.pt', map_location=device)
    theta_init = dataset.tensors[1][-bs:]

    theta0_lst = list()
    thetaT_lst = list()
    
    thetaT = None
    u.to(device)
    cnt = 0
    start_time = time.time()
    print("start generate dataset...")
    for i, theta0 in enumerate(theta_x_dataloader):
        theta0 = theta0[0].to(device)
        if thetaT is None:
            thetaT = theta0
        thetaT, L2RE = fit_theta(theta0, u, n_x_outer, n_x_inner, theta_init=thetaT)
        msk = L2RE < 2e-3  #original: 1e-3
        if msk.sum() > 0:
            theta0_lst.append(theta0[msk])
            thetaT_lst.append(thetaT[msk])
        cnt += msk.sum()
        #if (i < 500 and i % 10 == 0) or (i>=500 and i%500==0):
        if (i+1)%100==0:
            theta0_ = torch.cat(theta0_lst, dim = 0)
            thetaT_ = torch.cat(thetaT_lst, dim = 0)
            dataset= torch.utils.data.TensorDataset(theta0_, thetaT_)
            print(f'batch {i}, accept ratio = {cnt/((i+1)*bs)}, time = {time.time()-start_time}', flush=True)
            torch.save(dataset, f'checkpoints/heat_dataset_dim{u.dim}_params{theta0.shape[1]}_seed{seed}.pt')

    

def test(u,  theta0, thetaT_label, thetaT_pred):
    n = len(theta0)
    bs = min(3, n)
    n_x_outer = 1000  #old:500
    n_x_inner = 5000 #old:2500

    a=-1 #left boundary
    b=1 #right boundary
    

    
    L2RE_list = []
    for i in range(n//bs):
        theta0_i = theta0[i*bs:(i+1)*bs]
        thetaT_label_i = thetaT_label[i*bs:(i+1)*bs]
        thetaT_pred_i = thetaT_pred[i*bs:(i+1)*bs]
        
        
        def u0(y):
            theta0_rep = theta0_i.unsqueeze(1).repeat(1, n_x_outer, 1).reshape(-1, u.n_params) 
            return u.forward_2(theta0_rep, y)
        
        x = a + (b-a)*torch.rand((n_x_outer, u.dim), device=theta0.device)
        true_sol = heat.heat_solution_2(x, u0, n_x_inner, bs)  # (bs, n_x_outer)
        pred_sol = u.forward_2(thetaT_pred_i, x)
        label_sol = u.forward_2(thetaT_label_i, x)
    
        L2RE_t_p = (((true_sol - pred_sol)**2).mean(dim=1) / (true_sol**2).mean(dim=1)).unsqueeze(1) #shape (bs, 1)
        L2RE_t_l = (((true_sol - label_sol)**2).mean(dim=1) / (true_sol**2).mean(dim=1)).unsqueeze(1) #shape (bs, 1)
        L2RE_l_p = (((label_sol - pred_sol)**2).mean(dim=1) / (label_sol**2).mean(dim=1)).unsqueeze(1) #shape (bs, 1)
        L2RE_list.append(torch.cat([L2RE_t_p, L2RE_t_l, L2RE_l_p], dim=1))
    
    return torch.cat(L2RE_list, dim=0).mean(dim=0) #L2RE_t_p, L2RE_t_l, L2RE_l_p.  shape (n,3)-->(3,)



    
def eval(u, model, dataloader, device, nt=2):
    model.eval()
    with torch.no_grad():
        L2RE_list = []
        for theta0, thetaT_label in dataloader:
            theta0 = theta0.to(device)
            thetaT_label = thetaT_label.to(device)
            #thetaT_pred = model(theta0) #ResNet
            thetaT_pred = ode_fwd(theta0, model, is_train=False, nt=nt) #NODE
            L2RE = heat.test_2(u, thetaT_label, thetaT_pred)
            L2RE_list.append(L2RE)
        return sum(L2RE_list)/len(L2RE_list)

def train(u):
    bs = 300
    #dataset = torch.load(f'checkpoints/heat_dataset_dim5_size22667.pt')
    #dataset = torch.load(f'checkpoints/heat_dataset_dim5_seed0.pt')
    dataset = concat_dataset(dim = u.dim)
    # split dataset into train : valid : test = 8:1:1
    n = len(dataset)
    n_train = (n // 10) * 8
    n_valid = n // 10
    n_test = n - n_train - n_valid
    dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(dataset, [n_train, n_valid, n_test])
    dataload_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True, )
    dataload_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=bs, shuffle=False, )
    dataload_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=bs, shuffle=False, )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.V(dim=u.n_params) #NODE
    #model = models.ResNet(u.n_params, u.n_params, width=u.n_params, depth=3,  activation_func=F.relu, use_dropout=True) #ResNet

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.MSELoss() 
    
    start_time = time.time()
    n_epoch_change = -1
    for epoch in range(4000):
        model.train()
        for theta0, thetaT_label in dataload_train:
            optimizer.zero_grad()
            theta0 = theta0.to(device)
            thetaT_label = thetaT_label.to(device)
            if 1:
            #with torch.autocast(device_type='cuda', dtype=torch.float16):
                #thetaT_pred = model(theta0)  #ResNet
                thetaT_pred = ode_fwd(theta0, model, is_train=True) #NODE
                if epoch <= n_epoch_change:
                    loss = criterion(thetaT_pred, thetaT_label)
                else:
                    loss = heat.test_2(u, thetaT_label, thetaT_pred)
                
                    #loss_2 = test(u,  theta0, thetaT_label, thetaT_pred)[0]  #L2RE_t_p
        
            loss.backward()
            optimizer.step()
            
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()        
            
        if  (epoch <= n_epoch_change and epoch % 10 == 0) or (epoch>n_epoch_change and epoch%20==0):
            print(f'epoch {epoch}, Train loss {loss.item()}')
            print("L2RE_l_p=")
            
            print("Train:", eval(u, model, dataload_train, device))
            print("Valid:", eval(u, model, dataload_valid, device))
            print("Test:", eval(u, model, dataload_test, device))
            print("wall time=", time.time()-start_time, flush=True)
    return 
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            }, f'checkpoints/heat_model_dim{u.dim}.pt')
    

def ode_fwd(theta_0, v, is_train=False, nt=2):
    T=0.1 
    
    def ode_func(t, theta):
        return v(theta) ##dtheta_dt, shape=(bs, n_param)
    
    if nt <=2:
        t = torch.tensor([0, T]).to(theta_0)  #time grid
    else:
        t = torch.linspace(0, T, nt, device=theta_0.device)  #time grid, shape (nt,)
    if is_train:
        traj = odeint_adjoint(ode_func, y0=theta_0, t=t, method='rk4', options={'step_size':1e-2}, adjoint_params=v.parameters())
    else:
        traj = odeint(ode_func, y0=theta_0, t=t, method='dopri5', rtol=1e-2, atol=1e-7)

    thetaT_pred = traj[-1]
    return thetaT_pred


def concat_dataset(dim =10):
    datasets = []
    # traverse all files in 'checkpoints/' starts with 'heat_dataset_dim{dim}_seed'
    directory = 'checkpoints/'
    device = torch.device('cpu')
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.startswith(f'heat_dataset_dim10_params2130_seed') and filename.endswith('.pt'):
            dataset = torch.load(f, map_location=device)
            print(f, len(dataset))
            datasets.append(dataset)

    dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"{len(dataset)=}")  #22667
    torch.save(dataset, f'checkpoints/heat_dataset_dim{dim}_params2130_size{len(dataset)}.pt')
    return dataset

def test_ckp():
    u = models.U(dim=10)
    v = models.V(dim=u.n_params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    u.to(device)
    v.to(device)
    ckp_path = "checkpoints/heat_model_dim10_ep2_ba100.pth"  #L2RE=0.0372
    checkpoint = torch.load(ckp_path, map_location=device)
    v.load_state_dict(checkpoint['model_state_dict'])

    bs = 200
    dataset_test = concat_dataset(dim =10)
    #dataset_test = torch.load(f'checkpoints/heat_dataset_dim10_seed100.pt')
    dataload_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs, shuffle=False)

    print("L2RE=",eval(u, v, dataload_test, device))


def eval_(u, no, T, dataloader, device):
    no.eval()

    with torch.no_grad():
        L2RE_list = []
        for data in dataloader:
            theta0 = data[0].to(device)
            t = torch.ones(theta0.shape[0], device=device) * T
            thetaT_pred = no(theta0, t)
            
            L2RE = test(u, theta0, thetaT_pred, thetaT_pred)[0]
            L2RE_list.append(L2RE)
        return sum(L2RE_list)/len(L2RE_list)
    



if __name__ == '__main__':
    #redirect_log_file()
    set_seed(0, cudnn_benchmark=True)
    u = models.U6(dim = 10)
    concat_dataset(u.dim)

    
    #set_gpu_max_mem(1, force=True)
    #gen_dataset(u, seed=os.getpid())
