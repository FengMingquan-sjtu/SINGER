import math
import time
import os

import torch


import models
from utils import redirect_log_file, timing, set_seed, set_gpu_max_mem
import NO3
import hjb

class NO4(torch.nn.Module):
    def __init__(self, dim = 970, width=1000, depth=5):
        super(NO4, self).__init__()
        self.V = models.V(dim, width, depth)
        
    
    def forward(self, theta, t):
        ''' predict theta(T) given theta(0).
        Inputs:
            theta: torch.Tensor, shape (*bs, n_param)
            t: dummy input, not used.
        Outputs:
            theta_T: torch.Tensor, shape same as input theta
        '''
        input_theta_shape = theta.shape
        
        if theta.dim() > 2:
            theta = theta.reshape(-1, theta.shape[-1])

        theta_T = theta + self.V(theta)
        return theta_T




def data_aug_3(theta_traj, t, nt_bs):
    ''' random sample nt_bs time steps from theta_traj and t.
    Inputs:
        theta_traj: torch.Tensor, shape=(nt, bs, n_param)
        t: torch.Tensor, shape=(nt,)  linspace(T).
    Outputs:
        theta_0: torch.Tensor, shape=(min(nt_bs, nt-1), bs, n_param)
        theta_t: torch.Tensor, shape=(min(nt_bs, nt-1), bs, n_param)
        dt: torch.Tensor, shape=(min(nt_bs, nt-1),)
    '''
    theta_0, theta_t, dt = NO3.data_aug_0(theta_traj, t)
    nt_ = theta_0.shape[0]
    if nt_bs < nt_:
        idx = torch.randperm(nt_)[:nt_bs]
        for i in range(nt_bs//5):
            idx[i] = i # the first several time steps are always included.
        theta_0 = theta_0[idx]
        theta_t = theta_t[idx]
        dt = dt[idx]
    return theta_0, theta_t, dt
    


def data_aug_4(theta_traj, t, nt_bs):
    ''' slice first nt_bs time steps from theta_traj and t.
    Inputs:
        theta_traj: torch.Tensor, shape=(nt, bs, n_param)
        t: torch.Tensor, shape=(nt,)  linspace(T).
    Outputs:
        theta_0: torch.Tensor, shape=(min(nt_bs, nt-1), bs, n_param)
        theta_t: torch.Tensor, shape=(min(nt_bs, nt-1), bs, n_param)
        dt: torch.Tensor, shape=(min(nt_bs, nt-1),)
    '''
    theta_0, theta_t, dt = NO3.data_aug_0(theta_traj, t)
    nt_ = theta_0.shape[0]
    if nt_bs < nt_:
        theta_0 = theta_0[:nt_bs]
        theta_t = theta_t[:nt_bs]
        dt = dt[:nt_bs]
    return theta_0, theta_t, dt

def train():
    # parameters
    use_amp = False #automatic mixed precision
    T = 1.0  #time horizon, original: 1.0
    nt = int(T*400)+1 #time steps of NODE solver
    nt_bs = 40 # batchsize of random sampled time steps used for NO training.
    n_epoch = 200
    lr=1.25e-5
    bs = 20 # train batchsize
    test_bs = 5 #test batchsize
    print(f"{T=}, {nt=}, {nt_bs=}, {n_epoch=}, {lr=}, {bs=}, {test_bs=}, {use_amp=}")
    

    # device
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    
    # define models
    u = hjb.U_HJB(dim=8).to(device)  # target net
    v = models.V(dim=u.n_params).to(device) #node model
    no = NO3.NO3_2(dim=u.n_params, width=512, depth=2, T=T, time_scale=100).to(device)  # neural operator model  #original: width=512, depth=2
    #no = NO4(dim=u.n_params, width=1000, depth=5,).to(device)
    print(f"trainable params of {no.__class__.__name__ }:", sum(p.numel() for p in no.parameters() if p.requires_grad))

    # load pre-trained operator model (Optional)
    #checkpoint = torch.load("checkpoints/hjb_no31_dim8.pth", map_location=device)
    #no.load_state_dict(checkpoint['model_state_dict'])
    # load pre-trained node model v
    checkpoint = torch.load("checkpoints/hjb_dim8.pth", map_location=device)#L2RE=0.01
    v.load_state_dict(checkpoint['model_state_dict'])
    v.eval()
    

    # load datasets
    dataset_train = hjb.gen_dataset(200000, is_train=True)
    dataset_valid = hjb.gen_dataset(10, is_train=False) #original:100
    dataset_test = hjb.gen_dataset(10, is_train=False) #original:100
    dataload_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs, shuffle=True, drop_last=False)
    dataload_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=test_bs, shuffle=False, drop_last=False,)
    dataload_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=test_bs, shuffle=False, drop_last=False,)
    
    
    # train
    optimizer = torch.optim.Adam(no.parameters(), lr=lr)
    #optimizer = torch.optim.AdamW(no.parameters(), lr=lr, weight_decay=1e-4)
    print(f"optimizer={optimizer}")
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataload_train), 
    #                                                epochs=n_epoch, pct_start=0.3,div_factor=5.0, final_div_factor=10.0,)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=1e10)
    print(f"scheduler={scheduler}")
    start_time = time.time()
    best_ckp = {'train_loss':1e10, 'valid_l2re':1e10, 'test_l2re':1e10}
    #scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for i in range(n_epoch):
        no.train()
        for j, data in enumerate(dataload_train):
            optimizer.zero_grad(set_to_none=True)
            theta0 = data[0].to(device) #shape=(bs, n_param)
            #with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            theta_traj, t = NO3.ode_fwd(theta0, v, T, nt, adaptive=False) #theta_traj=(nt, bs, n_param), t=(nt,)
            t = t.unsqueeze(-1).repeat(1, theta0.shape[0])  #shape=(nt,) -> (nt, bs)
            theta_0, theta_t, dt = data_aug_3(theta_traj, t, nt_bs) #(nt,...) -> (nt_bs,...)
            theta_t_pred = no(theta_0, dt) #shape=(nt_bs, bs, n_param)
            loss = hjb.test_2(u, theta_0.reshape(-1, u.n_params), theta_t.reshape(-1, u.n_params), 
                            theta_t_pred.reshape(-1, u.n_params), 
                                normed=True, bs=nt_bs*bs)


                
            
            loss.backward()
            optimizer.step()
            #scaler.scale(loss).backward()
            
            #scaler.step(optimizer)
            #scaler.update()
            scheduler.step()
            
            
            if j % 2 == 0:
                print('----------------------')
                print(f'epoch {i}, batch {j}, loss {loss.item():.2e}' ,flush=True)
                print(f'Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
                #print(f"gpu util {torch.cuda.utilization()}, storage {torch.cuda.memory_allocated()/(1024 ** 3):.2f} GB")
                
                valid_l2re = eval(u, no, T, dataload_valid, device, nt=2, bs=test_bs)
                test_l2re = eval(u, no, T, dataload_test, device, nt=2, bs=test_bs)

                print("L2RE valid/test=", valid_l2re, test_l2re)
                
                print(f'walltime = {time.time()-start_time}', flush=True)
                #if loss.item() < best_ckp['train_loss']:
                #    best_ckp['train_loss'] = loss.item()
                    
                    

                if valid_l2re < best_ckp['valid_l2re']:
                    best_ckp['valid_l2re'] = valid_l2re
                    best_ckp['test_l2re'] = test_l2re
                    best_ckp['model_state_dict'] = no.state_dict()
                    best_ckp['optimizer_state_dict'] = optimizer.state_dict()
                    #best_ckp["scaler"] = scaler.state_dict()
                    print(f"best ckp updated")
                    model_path = f"checkpoints/hjb_no31_dim{u.dim}.pth"
                    torch.save(best_ckp, model_path)

                

def eval(u, no, T, dataloader, device, nt=2, bs = 1):
    no.eval()
    with torch.no_grad():
        L2RE_list = []
        for data in dataloader:
            theta0 = data[0].to(device)
            if nt <= 2: #one-step eval of NO
                t = torch.ones(theta0.shape[0], device=device) * T
                thetaT_pred = no(theta0, t)
            else: #recurrent eval of NO
                t = torch.ones(theta0.shape[0], device=device) * T / (nt-1)
                thetaT_pred = theta0
                for _ in range(nt-1): 
                    thetaT_pred = no(thetaT_pred, t)
            L2RE = hjb.test(u, theta0, thetaT_pred, bs=bs, T=T)
            L2RE_list.append(L2RE)
        return sum(L2RE_list)/len(L2RE_list)

def test_ckp(T = 0.5):
    dim = 8
    test_bs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = hjb.U_HJB(dim=dim).to(device)  # target net
    #no = NO3.NO3_1(dim=u.n_params,width=512, depth=2, T=T).to(device)  # neural operator model
    #no = NO3.NO3_2(dim=u.n_params, width=512, depth=2, T=T, time_scale=100).to(device)  # neural operator model
    no = models.NO5(dim=u.n_params, width=512, depth=2, T=T, time_scale=100).to(device)  # neural operator model
    model_path = f"checkpoints/hjb_no31_dim{dim}.pth"
    checkpoint = torch.load(model_path, map_location=device)
    #no.load_state_dict(checkpoint['model_state_dict'])

    dataset_test = hjb.gen_dataset(100, is_train=False)
    dataload_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=test_bs, shuffle=False, drop_last=False,)
    l2re = eval(u, no, T, dataload_test, device)
    print(f"{T=}, {l2re=}", flush=True)



if __name__ == "__main__":
    redirect_log_file()
    set_seed(0)
    set_gpu_max_mem(0)
    #train()
    for T in [0.5, 1, 1.5]:
        test_ckp(T)
    
   