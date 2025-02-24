import torch.nn as nn
import torch.nn.functional as F
import torch
from torchdiffeq import odeint_adjoint, odeint

def inference(theta_0, v, T=0.1):
    '''Simple ODE forward.'''
    def ode_func(t, theta):
        return v(theta) ##dtheta_dt, shape=(bs, n_param)
        
    t = torch.tensor([0, T]).to(theta_0)  #time grid
    traj = odeint(ode_func, y0=theta_0, t=t, method='rk4', options={'step_size':T/10})
    thetaT_pred = traj[-1] 

    return thetaT_pred

def laplacian(u, theta_rep, x):
    # div \cdot grad u.
    # Inputs:
    #     u: instance of U.
    #     theta_rep: torch.Tensor, shape (bs, n_x, n_param)
    #     x: torch.Tensor, shape (bs, n_x, d)
    # Outputs:
    #     div_u: torch.Tensor, shape (bs, n_x)
    
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

def loss_heat(theta_0, u, v, eqn):
    '''loss (Eq.6)'''
    bs = theta_0.shape[0] 
    n_x = 1000 #author mails that n_x=1000
    device = theta_0.device
    T = eqn.total_time
    d = u.dim
    
    a = eqn.a
    b = eqn.b
    x = eqn.sample(bs*n_x, device=device)
    x = x.reshape(bs, n_x, d)
    
    base_seed = torch.randint(0, 100000, (1,)).item()

    def ode_func(t, gamma):
        '''function used in odeint.
        Inputs:
            t: torch.Tensor, shape (1,) or (bs,)
            gamma: tuple of two torch.Tensor, theta and r, shape (bs, n_param) and (1,)
        '''
        
        theta, r = gamma
        g = torch.Generator(device=device).manual_seed(base_seed + int(t.item()*10000))
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
    traj = odeint_adjoint(ode_func, y0=(theta_0, r_0), t=t, method='rk4', options={'step_size':T/10}, adjoint_params=v.parameters())
    
    theta_traj, r_traj = traj
    theta_T = theta_traj[-1] #shape = (bs, n_param)
    theta_T_norm = torch.linalg.vector_norm(theta_T, ord=2, dim=-1).mean()#shape =  (bs, n_param)-->(bs,)-->(1,)
    r_T = r_traj[-1] #final condition for r
    return r_T, theta_T_norm #shape = (1,),   (1,)

def loss_hjb(theta_0, u, v, eqn):
    '''loss (Eq.6)'''
    T = eqn.total_time
    eps = eqn.eps
    n_x = 5000
    
    x, p_x = eqn.sample(theta_0, n_x)  #shape=(bs, n_x, d), (bs, n_x)
    
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
        du_dtheta = du_dtheta_(u, theta_rep, x)  #shape=(bs, n_x, n_param)
        du_dt = du_dtheta * dtheta_dt.unsqueeze(1)  #shape=(bs, n_x, n_param)
        du_dt = du_dt.sum(-1)  #shape=(bs, n_x)
        lhs = du_dt
        
        #HJB equation u_t = eps * Delta u - 1/2 * norm(grad u)**2  (Eq.27) #NOTE: eps > 0 
        rhs = eps * laplacian(u, theta_rep, x)  #shape=(bs, n_x)
        rhs -= 0.5 * du_dx_(u, theta_rep, x).pow(2).sum(dim=-1)  #shape= (bs, n_x, dim)--> (bs, n_x)
        dr_dt = ((lhs - rhs).pow(2) / p_x).mean()
        return (dtheta_dt, dr_dt)
    

    r_0 = torch.zeros(1).to(theta_0) # initial condition for r
    t = torch.tensor([0, T]).to(theta_0)  #time grid
    traj = odeint_adjoint(ode_func, y0=(theta_0, r_0), t=t, method='rk4', options={'step_size':T/10}, adjoint_params=v.parameters())
    theta_traj, r_traj = traj
    theta_T = theta_traj[-1] #shape = (bs, n_param)
    theta_T_norm = torch.linalg.vector_norm(theta_T, ord=2, dim=-1).mean()#shape =  (bs, n_param)-->(bs,)-->(1,)
    r_T = r_traj[-1] #final condition for r
    return r_T, theta_T_norm #shape = (1,), (1,)

def loss_pino_heat(theta_0, u, no, eqn):
    '''PINN loss'''
    # ---hyperparameters---
    bs = theta_0.shape[0] 
    n_x = 1000 #author mails that n_x=1000
    device = theta_0.device
    T = eqn.total_time
    d = u.dim
    
    a = eqn.a
    b = eqn.b
    x = eqn.sample(bs*n_x, device=device)
    x = x.reshape(bs, n_x, d)

    # ---residual loss---
    def ode_func(t, r):
        du_dt, theta_t = du_dt_(u, no, theta_0, x, t, dt=T/100)
        lhs = du_dt
        rhs = laplacian(u, theta_t, x)  #shape=(bs, n_x)
        dr_dt = (lhs - rhs).pow(2).mean(dim=0).mean(dim=0,keepdim=True) * (b-a)**d
        return dr_dt + 0 * r #add dummy variable to avoid gradient error
    
    r_0 = torch.zeros(1).to(theta_0) # initial condition for r
    t = torch.tensor([0, T]).to(theta_0)  #time grid, start from a small positive value instead of 0 to avoid nan
    traj = odeint_adjoint(ode_func, y0=r_0, t=t, method='rk4', options={'step_size':1e-2}, adjoint_params=no.parameters())
    r_T = traj[-1]
    return r_T

def loss_pino_hjb(theta_0, u, no, eqn):
    '''PINN loss'''
    # ---hyperparameters---
    T = eqn.total_time
    eps = eqn.eps
    n_x = 5000
    
    x, p_x = eqn.sample(theta_0, n_x)  #shape=(bs, n_x, d), (bs, n_x)

    # ---residual loss---
    def ode_func(t, r):
        du_dt, theta_t = du_dt_(u, no, theta_0, x, t, dt=T/100)
        lhs = du_dt
        rhs = laplacian(u, theta_t, x)  #shape=(bs, n_x)
        #HJB equation u_t = eps * Delta u - 1/2 * norm(grad u)**2  (Eq.27) #NOTE: eps > 0 
        rhs = eps * laplacian(u, theta_t, x)  #shape=(bs, n_x)
        rhs -= 0.5 * du_dx_(u, theta_t, x).pow(2).sum(dim=-1)  #shape= (bs, n_x, dim)--> (bs, n_x)
        dr_dt = ((lhs - rhs).pow(2) / p_x).mean()
        return dr_dt + 0 * r #add dummy variable to avoid gradient error
    
    r_0 = torch.zeros(1).to(theta_0) # initial condition for r
    t = torch.tensor([0, T]).to(theta_0)  #time grid, start from a small positive value instead of 0 to avoid nan
    traj = odeint_adjoint(ode_func, y0=r_0, t=t, method='rk4', options={'step_size':T/10}, adjoint_params=no.parameters())
    r_T = traj[-1]
    return r_T
