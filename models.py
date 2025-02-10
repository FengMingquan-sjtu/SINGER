import math
from collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
#--- for GNN only ----
from torch_geometric.nn import  GCN
from torch_geometric.utils import to_undirected
#from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
#from torch_geometric.utils import cumsum, sort_edge_index, subgraph
#from torch_geometric.utils.num_nodes import maybe_num_nodes
# --------------------

class FFNet(torch.nn.Module):
    # --- copied from neural control codes
    def __init__(self,d,dd,width,depth,activation_func, *args, **kwargs):
        '''
        Inputs:
            d: input dims
            dd: output dims
            width: hidden dims,
            depth: num hiddens
        '''
        super(FFNet, self).__init__()
        is_mlp = not kwargs.get('mlp', False)
        self.use_dropout = kwargs.get('dropout',False)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        self.d = d
        # first layer is from input (d = dim) to 1st hidden layer (d = width)
        self.linearIn = nn.Linear(d, width, bias=is_mlp)

        # create hidden layers
        self.linear = nn.ModuleList()
        for _ in range(depth):
            self.linear.append(nn.Linear(width, width,bias=is_mlp))

        # output layer is linear
        self.linearOut = nn.Linear(width, dd,bias = False)
        self.activation = activation_func

    def forward(self, x):
        # compute the 1st layer (from input layer to 1st hidden layer)
        x = self.activation(self.linearIn(x)) # Match dimension
        # compute from i to i+1 layer
        for layer in self.linear:
            if self.use_dropout:
                y = self.dropout(x)
            else:
                y = x
            x_temp = self.activation(layer(y))
            x = x_temp
        # return the output layer
        return self.linearOut(x)

class ResNet(torch.nn.Module):
    # --- copied from neural control codes
    def __init__(self,d,dd,width,depth,activation_func, *args, **kwargs):
        super(ResNet, self).__init__()
        self.use_dropout = kwargs.get('dropout',False)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        is_mlp = not kwargs.get('mlp', False)
        self.d = d
        # first layer is from input (d = dim) to 1st hidden layer (d = width)
        self.linearIn = nn.Linear(d, width, bias=is_mlp)

        # create hidden layers
        self.linear = nn.ModuleList()
        for _ in range(depth):
            self.linear.append(nn.Linear(width, width,bias=is_mlp))

        # output layer is linear
        self.linearOut = nn.Linear(width, dd,bias = False)
        self.activation = activation_func
    def forward(self, x):
        # compute the 1st layer (from input layer to 1st hidden layer)
        x = self.activation(self.linearIn(x)) # Match dimension
        # x = self.linearIn(x) # Match dimension
        # compute from i to i+1 layer
        for layer in self.linear:
            if self.use_dropout:
                y = self.dropout(x)
            else:
                y = x
            x_temp =  x + self.activation(layer(y))
            x = x_temp
        # return the output layer
        return self.linearOut(x)
    


class U(nn.Module):
    def __init__(self, dim = 10, width=80):
        super().__init__()
        self.dim = dim
        self.width = width

        # Papemeters, Section 5.2.1. in the paper.
        # beta \in R^d
        # a \in R^{d*width}
        # b \in R^{width}
        # c \in R^{width}
        
        self.a = nn.Parameter(torch.randn(self.dim,self.width))
        self.b = nn.Parameter(torch.randn(self.width))
        self.c = nn.Parameter(torch.randn(self.width))
        self.beta = nn.Parameter(torch.randn(self.dim))
        
        self.n_params = nn.utils.parameters_to_vector(self.parameters()).shape[0]
        self.dims = [[dim], [dim, width], [width], [width]]
        self.connections = ['F', 'F', 'E']
    
    def forward(self, x):
        ''' Single forward,  demo ONLY.
        Inputs:
            x: torch.Tensor, shape (batch_size, d)
        Outputs:
            u: torch.Tensor, shape (batch_size)
        '''
        a, b, c, beta = self.a, self.b, self.c, self.beta
        return torch.tanh(torch.sin(math.pi * (x- beta)) @ a - b) @ c
    
    def forward_2(self, theta, x):
        ''' batched forward used in training and inference.
        Inputs:
            theta: torch.Tensor, shape (batch_size, n_x,  len_theta) or (batch_size, len_theta)
            x: torch.Tensor, shape (batch_size, n_x, d) or (n_x, d)
        Outputs:
            u: torch.Tensor, shape (batch_size, n_x)
        '''
        
        bs = theta.shape[0]
        nx = x.shape[-2]
        dim, width = self.dim, self.width
        a_size = dim * width
        b_size = width
        c_size = width
        beta_size = dim

        if theta.dim() == 2:
            theta = theta.unsqueeze(1).repeat(1, nx, 1)  #shape=(bs, n_x, n_param)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)   #shape=(bs, n_x, d)

        idx = 0
        beta = theta[..., : beta_size]  #shape=(bs, nx, d)
        idx += beta_size
        a = theta[..., idx: idx+a_size].reshape(bs, nx, dim, width)
        idx += a_size
        b = theta[..., idx: idx+b_size].reshape(bs, nx, 1, width)
        idx += b_size
        c = theta[..., idx: idx+c_size].reshape(bs, nx, width, 1)
        

        u0 = torch.sin(math.pi * (x- beta)) #shape=(bs, nx,  d)
        u0 = u0.unsqueeze(-2)  #shape=(bs, nx, 1, d)
        u0 = torch.tanh(u0 @ a - b)  #shape=(bs, nx, 1, width)
        u0 = u0 @ c   #shape=(bs, nx, 1, 1)
        return u0.squeeze(-2).squeeze(-1) #shape=(bs,nx)


class U3(nn.Module):
    def __init__(self, dim = 10, width=80):
        ''' delete b from U, remian 3 variables'''
        super().__init__()
        self.dim = dim
        self.width = width
        
        self.a = nn.Parameter(torch.randn(self.dim,self.width))
        self.b = nn.Parameter(torch.randn(self.width))
        self.c = nn.Parameter(torch.randn(self.width))
        
        self.n_params = nn.utils.parameters_to_vector(self.parameters()).shape[0]
        self.dims = [[dim, width], [width], [width]]
        self.connections = ['F', 'E']
    
    def forward(self, x):
        ''' Single forward,  demo ONLY.
        Inputs:
            x: torch.Tensor, shape (batch_size, d)
        Outputs:
            u: torch.Tensor, shape (batch_size)
        '''
        a, b, c = self.a, self.b, self.c
        return torch.tanh(torch.sin(math.pi * (x)) @ a - b ) @ c
    
    def forward_2(self, theta, x):
        ''' batched forward used in training and inference.
        Inputs:
            theta: torch.Tensor, shape (batch_size, n_x,  len_theta) or (batch_size, len_theta)
            x: torch.Tensor, shape (batch_size, n_x, d) or (n_x, d)
        Outputs:
            u: torch.Tensor, shape (batch_size, n_x)
        '''
        
        bs = theta.shape[0]
        nx = x.shape[-2]
        dim, width = self.dim, self.width
        a_size = dim * width
        b_size = width
        c_size = width
        #beta_size = dim

        if theta.dim() == 2:
            theta = theta.unsqueeze(1).repeat(1, nx, 1)  #shape=(bs, n_x, n_param)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)   #shape=(bs, n_x, d)

        idx = 0
        #beta = theta[..., : beta_size]  #shape=(bs, nx, d)
        #idx += beta_size
        a = theta[..., idx: idx+a_size].reshape(bs, nx, dim, width)
        idx += a_size
        b = theta[..., idx: idx+b_size].reshape(bs, nx, 1, width)
        idx += b_size
        c = theta[..., idx: idx+c_size].reshape(bs, nx, width, 1)
        

        u0 = torch.sin(math.pi * (x)) #shape=(bs, nx,  d)
        u0 = u0.unsqueeze(-2)  #shape=(bs, nx, 1, d)
        u0 = torch.tanh(u0 @ a - b)  #shape=(bs, nx, 1, width)
        u0 = u0 @ c   #shape=(bs, nx, 1, 1)
        return u0.squeeze(-2).squeeze(-1) #shape=(bs,nx)

class U5(nn.Module):
    def __init__(self, dim = 10, width=80):
        '''add new varibale to U, 5 variables in total'''
        super().__init__()
        self.dim = dim
        self.width = width
        
        self.a = nn.Parameter(torch.randn(self.dim,self.width))
        self.b = nn.Parameter(torch.randn(self.width))
        self.c = nn.Parameter(torch.randn(self.width, 1))
        self.d = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(self.dim))
        
        self.n_params = nn.utils.parameters_to_vector(self.parameters()).shape[0]
        self.dims = [[dim], [dim, width], [width], [width, 1], [1]]
        self.connections = ['F', 'F', 'F', 'F']
    
    def forward(self, x):
        ''' Single forward,  demo ONLY.
        Inputs:
            x: torch.Tensor, shape (batch_size, d)
        Outputs:
            u: torch.Tensor, shape (batch_size)
        '''
        a, b, c, d, beta = self.a, self.b, self.c, self.d, self.beta
        return torch.tanh(torch.sin(math.pi * (x- beta)) @ a - b) @ c - d
    
    def forward_2(self, theta, x):
        ''' batched forward used in training and inference.
        Inputs:
            theta: torch.Tensor, shape (batch_size, n_x,  len_theta) or (batch_size, len_theta)
            x: torch.Tensor, shape (batch_size, n_x, d) or (n_x, d)
        Outputs:
            u: torch.Tensor, shape (batch_size, n_x)
        '''
        
        bs = theta.shape[0]
        nx = x.shape[-2]
        dim, width = self.dim, self.width
        a_size = dim * width
        b_size = width
        c_size = width
        d_size = 1
        beta_size = dim

        if theta.dim() == 2:
            theta = theta.unsqueeze(1).repeat(1, nx, 1)  #shape=(bs, n_x, n_param)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)   #shape=(bs, n_x, d)

        idx = 0
        beta = theta[..., : beta_size]  #shape=(bs, nx, d)
        idx += beta_size
        a = theta[..., idx: idx+a_size].reshape(bs, nx, dim, width)
        idx += a_size
        b = theta[..., idx: idx+b_size].reshape(bs, nx, 1, width)
        idx += b_size
        c = theta[..., idx: idx+c_size].reshape(bs, nx, width, 1)
        idx += c_size
        d = theta[..., idx: idx+d_size].reshape(bs, nx, 1, 1)
        

        u0 = torch.sin(math.pi * (x- beta)) #shape=(bs, nx,  d)
        u0 = u0.unsqueeze(-2)  #shape=(bs, nx, 1, d)
        u0 = torch.tanh(u0 @ a - b)  #shape=(bs, nx, 1, width)
        u0 = u0 @ c - d   #shape=(bs, nx, 1, 1)
        return u0.squeeze(-2).squeeze(-1) #shape=(bs,nx)


class U6(nn.Module):
    '''U with 6 layers (base U is 4-layered)'''
    def __init__(self, dim = 10, width=40, width_1=40):
        super().__init__()
        self.dim = dim
        self.width = width
        self.width_1 = width_1    
        
        self.a = nn.Parameter(torch.randn(self.dim,self.width))
        self.b = nn.Parameter(torch.randn(self.width))
        self.c = nn.Parameter(torch.randn(self.width, self.width_1))
        self.d = nn.Parameter(torch.randn(self.width_1))
        self.e = nn.Parameter(torch.randn(self.width_1))
        self.beta = nn.Parameter(torch.randn(self.dim))
        
        self.n_params = nn.utils.parameters_to_vector(self.parameters()).shape[0]
        self.dims = [[dim], [dim, width], [width], [width, width_1], [width_1], [width_1]]
        self.connections = ['F', 'F', 'F', 'F', 'E']
    
    def forward(self, x):
        '''
        Inputs:
            x: torch.Tensor, shape (batch_size, d)
        Outputs:
            u: torch.Tensor, shape (batch_size)
        '''
        a, b, c, d, e, beta = self.a, self.b, self.c, self.d, self.e, self.beta
        #return torch.tanh(torch.tanh(torch.sin(math.pi * (x- beta)) @ a - b) @ c - d) @ e   #original
        return torch.tanh(torch.relu(torch.sin(math.pi * (x- beta)) @ a - b) @ c - d) @ e  #simplified
    
    def forward_2(self, theta, x):
        '''
        Inputs:
            theta: torch.Tensor, shape (batch_size, n_x,  len_theta) or (batch_size, len_theta)
            x: torch.Tensor, shape (batch_size, n_x, d) or (n_x, d)
        Outputs:
            u: torch.Tensor, shape (batch_size, n_x)
        '''
        
        bs = theta.shape[0]
        nx = x.shape[-2]
        dim, width, width_1 = self.dim, self.width, self.width_1
        beta_size = torch.numel(self.beta)
        a_size = torch.numel(self.a)
        b_size = torch.numel(self.b)
        c_size = torch.numel(self.c)
        d_size = torch.numel(self.d)
        e_size = torch.numel(self.e)
        
        if theta.dim() == 2:
            theta = theta.unsqueeze(1).repeat(1, nx, 1)  #shape=(bs, n_x, n_param)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)   #shape=(bs, n_x, d)
        idx = 0
        beta = theta[..., idx:idx+beta_size]  #shape=(bs, nx, d)
        idx += beta_size
        a = theta[..., idx:idx+a_size].reshape(bs, nx, dim, width)
        idx += a_size
        b = theta[..., idx: idx+b_size].reshape(bs, nx, 1, width)
        idx += b_size
        c = theta[..., idx: idx+c_size].reshape(bs, nx, width, width_1)
        idx += c_size
        d = theta[..., idx: idx+d_size].reshape(bs, nx, 1, width_1)
        idx += d_size
        e = theta[..., idx: idx+e_size].reshape(bs, nx, width_1, 1)
        

        
        
        u0 = torch.sin(math.pi * (x- beta)) #shape=(bs, nx,  d)
       
        u0 = u0.unsqueeze(-2)  #shape=(bs, nx, 1, d)
        
        #u0 = torch.tanh(u0 @ a - b)  #shape=(bs, nx, 1, width)  #original
        u0 = torch.relu(u0 @ a - b)  #shape=(bs, nx, 1, width)            #simplified
        u0 = torch.tanh(u0 @ c - d)  #shape=(bs, nx, 1, width_1)
        u0 = u0 @ e #shape=(bs, nx, 1, 1)
        return u0.squeeze(-2).squeeze(-1) #shape=(bs,nx)



class U_HJB(nn.Module):
    def __init__(self, dim = 8, width=50):
        super().__init__()
        self.dim = dim
        self.width = width

        # Papemeters, Section 5.2.3. in the paper.
        # a \in R^{d*width}
        # b \in R^{d*width}
        # w \in R^{width}
        self.c = nn.Parameter(torch.randn(self.dim)) #add a bias layer to make the graph connected.
        self.b = nn.Parameter(torch.randn(self.dim,self.width))
        self.a = nn.Parameter(torch.randn(self.dim,self.width))
        self.w = nn.Parameter(torch.randn(self.width)) 
        
        self.n_params = nn.utils.parameters_to_vector(self.parameters()).shape[0]
        self.dims = [[dim], [dim, width], [dim, width], [width]]
        self.connections = ['F', 'E', 'F', ]
    
    def forward(self, x):
        '''
        Inputs:
            x: torch.Tensor, shape (bs, dim).  bs denotes batch size.
        Outputs:
            u: torch.Tensor, shape (bs)
        '''
        a, b, w, c = self.a, self.b, self.w, self.c
        a = a.unsqueeze(0) # shape = (1,  dim, width)
        b = b.unsqueeze(0) # shape = (1,  dim, width)
        c = c.unsqueeze(0).unsqueeze(2) # shape = (1, dim, 1)
        x = x.unsqueeze(2) # shape = (bs, dim, 1)

        out = a * (x - c - b) # shape = (bs, dim, width)
        out = torch.linalg.vector_norm(out, dim=-2) # shape = (bs, width)
        out = torch.exp(-out/2) # shape = (bs, width)
        out = torch.mv(out, w) #(bs, width)*(width,) --> (bs,)
        out = out
        return out 
    
    def forward_2(self, theta, x):
        '''
        Inputs:
            theta: torch.Tensor, shape (bs, n_x,  n_params) or (bs, n_params)
            x: torch.Tensor, shape (bs, n_x, d) or (n_x, d)
        Outputs:
            u: torch.Tensor, shape (bs, n_x)
        '''
        
        bs = theta.shape[0]
        nx = x.shape[-2]
        dim, width = self.dim, self.width
        c_size = dim 
        a_size = dim * width
        b_size = dim * width
        w_size = width
        

        if theta.dim() == 2:
            theta = theta.unsqueeze(1).repeat(1, nx, 1)  #shape=(bs, n_x, n_param)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)   #shape=(bs, n_x, d)

        # parameters must be ordered as computation in forward function.
        idx = 0
        c = theta[..., :c_size].reshape(bs, nx, dim, 1)
        idx += c_size
        b = theta[..., idx:idx+b_size].reshape(bs, nx, dim, width)
        idx += b_size
        a = theta[..., idx:idx+a_size].reshape(bs, nx, dim, width)
        idx += a_size
        w = theta[..., idx:idx+w_size]  #shape=(bs, nx, width)
        
        x = x.unsqueeze(-1) # shape = (bs, n_x, d, 1)

        #print(a.shape, b.shape, w.shape, x.shape)
        out = a * (x -c - b) # shape = (bs, n_x, d, width)
        out = torch.linalg.vector_norm(out, dim=-2) # shape = (bs, n_x, width)
        out = torch.exp(-out/2) # shape =  (bs, n_x, width)
        out = (out * w).sum(-1) # shape = (bs, n_x)
        out = out 
        return out



class U_PRICE(nn.Module):
    def __init__(self, dim = 10, width=80):
        super().__init__()
        self.dim = dim
        self.width = width

        self.beta = nn.Parameter(torch.randn(self.dim))
        self.a = nn.Parameter(torch.randn(self.dim,self.width))
        self.b = nn.Parameter(torch.randn(self.width))
        self.c = nn.Parameter(torch.randn(self.width))
        
        self.n_params = nn.utils.parameters_to_vector(self.parameters()).shape[0]
        self.dims = [[dim], [dim, width], [width], [width]]
        self.connections = ['F', 'F', 'E']
    
    def forward(self, x):
        ''' Single forward,  demo ONLY.
        Inputs:
            x: torch.Tensor, shape (batch_size, d)
        Outputs:
            u: torch.Tensor, shape (batch_size)
        '''
        a, b, c, beta = self.a, self.b, self.c, self.beta
        return torch.tanh((x- beta) @ a - b) @ c
    
    def forward_2(self, theta, x):
        ''' batched forward used in training and inference.
        Inputs:
            theta: torch.Tensor, shape (batch_size, n_x,  len_theta) or (batch_size, len_theta)
            x: torch.Tensor, shape (batch_size, n_x, d) or (n_x, d)
        Outputs:
            u: torch.Tensor, shape (batch_size, n_x)
        '''
        
        bs = theta.shape[0]
        nx = x.shape[-2]
        dim, width = self.dim, self.width
        a_size = dim * width
        b_size = width
        c_size = width
        beta_size = dim

        if theta.dim() == 2:
            theta = theta.unsqueeze(1).repeat(1, nx, 1)  #shape=(bs, n_x, n_param)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(bs, 1, 1)   #shape=(bs, n_x, d)

        idx = 0
        beta = theta[..., : beta_size]  #shape=(bs, nx, d)
        idx += beta_size
        a = theta[..., idx: idx+a_size].reshape(bs, nx, dim, width)
        idx += a_size
        b = theta[..., idx: idx+b_size].reshape(bs, nx, 1, width)
        idx += b_size
        c = theta[..., idx: idx+c_size].reshape(bs, nx, width, 1)
        

        u0 = x- beta #shape=(bs, nx,  d)
        u0 = u0.unsqueeze(-2)  #shape=(bs, nx, 1, d)
        u0 = torch.tanh(u0 @ a - b)  #shape=(bs, nx, 1, width)
        u0 = u0 @ c   #shape=(bs, nx, 1, 1)
        return u0.squeeze(-2).squeeze(-1) #shape=(bs,nx)



class V(nn.Module):
    r"""V network in the paper, mapping from theta to dtheta_dt."""
    def __init__(self, dim = 970, width=1000, depth=5, dropout_p=0, activation_func=F.relu, shape_strict=True, shapes=None):
        super(V, self).__init__()
        self.dim = dim
        self.width = width
        self.shape_strict = shape_strict
        self.shapes = shapes #a dict of shapes old and new. None if shape_strict is True.

        # Papemeters, Section 5.1. in the paper.
        self.S = FFNet(dim, 1, width, depth,  F.sigmoid, )  
        self.R = ResNet(dim, dim, width, depth,  activation_func, )
        self.E = FFNet(dim, dim, width, depth,  activation_func, )

        self.dropout_p = dropout_p
    
    def forward(self, theta, generator=None):
        '''
        Inputs:
            theta: torch.Tensor, shape (batch_size, dim)
        Outputs:
            dtheta_dt: torch.Tensor, shape (batch_size, dim)
        '''
        if self.shape_strict:
            S = self.S(theta)
            R = self.R(theta)
            E = self.E(theta)
            
            dtheta_dt = S * (R + E*theta)
            dtheta_dt = dropout(dtheta_dt, p=self.dropout_p, training=self.training, g=generator) #add drop out
        
        else: #handle different shape of theta
            dim, width = self.shapes['dim'], self.shapes['width']
            new_dim, new_width = self.shapes['new_dim'], self.shapes['new_width']
            a_size = dim * width
            b_size = width
            c_size = width
            beta_size = dim
            new_a_size = new_dim * new_width
            new_b_size = new_width
            new_c_size = new_width
            new_beta_size = new_dim

            #--- convert newshape to old shape (by class U) ---
            new_idxs = torch.arange(0, theta.shape[-1], device=theta.device, dtype=torch.long)
            new_to_old_idxs = []
            idx = 0
            new_to_old_idxs.append(new_idxs[:beta_size])
            idx += new_beta_size
            new_to_old_idxs.append(new_idxs[idx:idx+new_a_size].reshape(new_dim, new_width)[:dim, :width].flatten())
            idx += new_a_size
            new_to_old_idxs.append(new_idxs[idx:idx+b_size])
            idx += new_b_size
            new_to_old_idxs.append(new_idxs[idx:idx+c_size])
            new_to_old_idxs = torch.cat(new_to_old_idxs, dim=-1)    
            old_theta = theta[..., new_to_old_idxs]

            # --- same inference as above ---
            S = self.S(old_theta)
            R = self.R(old_theta)
            E = self.E(old_theta)
            old_dtheta_dt = S * (R + E*old_theta)
            old_dtheta_dt = dropout(old_dtheta_dt, p=self.dropout_p, training=self.training, g=generator)


            # --- convert back to new_shape ---
            dtheta_dt = torch.zeros_like(theta)
            dtheta_dt[..., new_to_old_idxs] = old_dtheta_dt

            
        return dtheta_dt


def dropout(x, p, g, training):
    r""" iid dropout + rescale of tensor.
    Inputs:
        x: torch.Tensor, shape (batch_size, dim)
        p: float, dropout rate
        g: torch.Generator
        training: bool, whether to apply dropout
    Outputs:
        x: torch.Tensor, shape (batch_size, dim)
    """
    if p > 0 and training:
        mask = torch.bernoulli((1-p)*torch.ones_like(x), generator=g)
        x = x * mask / (1-p)
    return x



class MyGCNConv3(MessagePassing):
    def __init__(self, in_channels, out_channels, v_width, v_depth, m_width, m_depth):
        super().__init__(aggr='add')  
        self.M = ResNet(in_channels, out_channels, width=m_width, depth=m_depth, activation_func=F.relu) 
        self.V = V(out_channels, width=v_width, depth=v_depth, activation_func=F.relu) 
        


    def forward(self, x, edge_index):
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # remove self-loop,since already added in get_nn_dual_edge_index
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        # x_j shape = [E, in_channels]
        # out shape = [E, out_channels]
        return self.M(x_j)

    def update(self, x):
        return self.V(x) 


class MyGCN3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, v_width, v_depth, m_width, m_depth, skip_connection=True, activation_func=None):
        super().__init__()
        self.conv1 = MyGCNConv3(in_channels, hidden_channels, v_width, v_depth, m_width, m_depth)
        self.conv2 = MyGCNConv3(hidden_channels, hidden_channels, v_width, v_depth, m_width, m_depth)
        self.conv3 = MyGCNConv3(hidden_channels, in_channels, v_width, v_depth, m_width, m_depth)
        self.skip_connection = skip_connection
        self.activation_func = activation_func
    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        if not self.activation_func is None:
            x1 = self.activation_func(x1)
        x2 = self.conv2(x1, edge_index)
        if self.skip_connection: 
            x2 += x1
        if not self.activation_func is None:
            x2 = self.activation_func(x2)
        x3 = self.conv3(x2, edge_index)
        if self.skip_connection: 
            x3 += x
        return x3

class MyGCN4(torch.nn.Module):
    def __init__(self, n_layers, in_channels, hidden_channels, v_width, v_depth, m_width, m_depth, skip_connection=True, activation_func=None):
        super().__init__()
        self.convs = nn.ModuleList([])
        self.convs.append(MyGCNConv3(in_channels, hidden_channels, v_width, v_depth, m_width, m_depth))
        for _ in range(n_layers-2):
            self.convs.append(MyGCNConv3(hidden_channels, hidden_channels, v_width, v_depth, m_width, m_depth))
        self.convs.append(MyGCNConv3(hidden_channels, in_channels, v_width, v_depth, m_width, m_depth))
        self.skip_connection = skip_connection
        self.activation_func = activation_func
    def forward(self, x, edge_index):
        for conv in self.convs:
            if not self.skip_connection:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index) + x
            if not self.activation_func is None:
                x = self.activation_func(x)
        return x
    
def get_nn_dual_edge_index(dims, connections, device, undirected=True, self_loop=False):
    """ Generate Dual-graph for neural network, where the nodes are weights and the edges are neurons.
    Input:
        dims: list of list of int, each inner list is the param.shape of the layer i.
            For matrix product, we assume forward func is y = x @ w, thus the shape of w is exactly the input-ouput shape of the layer.
        connections: list of char, each char is the connection type of the layer i to i+1. 
            choose char from {'E':"element-wise", 'F':"fully-connected"}.keys()
        undirected: bool, whether to add backward edges.
        self_loop: bool, whether to add self-loop edges.
    Output:
        edge_index: torch.Tensor, type long, shape (2, num_edges), where num_edges = num of edges in the neural network
        node_label: torch.Tensor, type float, shape (num_nodes, len(dims)), one-hot encoding of the weights.
    """
    nodes = []
    node_idx = 0
    for d in dims:
        n_node_in_layer = math.prod(d)
        node = torch.arange(node_idx, node_idx + n_node_in_layer, device=device, dtype=torch.long).reshape(d)
        nodes.append(node)
        node_idx += n_node_in_layer
    node_label = torch.zeros(node_idx, len(dims), dtype=torch.bool, device=device)
    for i, node in enumerate(nodes):
        node_label[node.flatten(), i] = 1 #node labels: the nodes in i-th layer is labeled with one-hot encoding at i.
    
    edges = []
    for i, cnt in enumerate(connections):
        if cnt == 'E':
            src_nodes = nodes[i].reshape(-1)
            tgt_nodes = nodes[i+1].reshape(-1)
            edges.extend(list(zip(src_nodes, tgt_nodes)))
        elif cnt == 'F':
            MAX_IDX = 1000000000 #max number of nodes in u.
            src = nodes[i] * MAX_IDX
            tgt = nodes[i+1]
            if len(src.shape) == 1 and len(tgt.shape) == 2:
                src = src.unsqueeze(1)
            elif len(src.shape) == 2 and len(tgt.shape) == 2:
                src = src.unsqueeze(2)
                tgt = tgt.unsqueeze(0)
            pair = (src + tgt).flatten()  # use torch broadcasting to get pair of src and tgt nodes.
            src_nodes = pair // MAX_IDX
            tgt_nodes = pair % MAX_IDX
            edges.extend(list(zip(src_nodes, tgt_nodes)))
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


    if undirected:
        edge_index = to_undirected(edge_index) 
    if self_loop:
        edge_index = add_self_loops(edge_index, num_nodes=node_idx)[0]
    return edge_index, node_label
    
        

class GnnV5(nn.Module):
    r"""basic GNN for V network, mapping from theta to dtheta_dt in graph space."""
    def __init__(self, u, dropout_p=0.1, n_hiddens=100, skip_connection=True, undirected=False, activation_func=None):
        super().__init__()
        self.dim = u.n_params  # num of input channels = num of output channels = num of u.parameters
        self.dims = u.dims  # dims of each layer of u
        self.cnts = u.connections # connection type of each layer of u
        self.dropout_p = dropout_p
        self.skip_connection = skip_connection
        self.undirected = undirected
        self.cache = {}
        
        self.G =  MyGCN3(in_channels=len(self.dims), hidden_channels=n_hiddens, v_width=n_hiddens, v_depth=2, m_width=n_hiddens, m_depth=2, skip_connection=skip_connection, activation_func=activation_func)  
        
    def get_graph(self, batch_size, device):
        if not (batch_size, str(device)) in self.cache:
            # define edges and initial node_feature.
            print("initial graph cache")
            edge_index, node_label = get_nn_dual_edge_index(dims=self.dims, connections=self.cnts, device=device, undirected=self.undirected, self_loop=True)
            n_nodes = edge_index.max().item() + 1

            edge_index = torch.cat([edge_index + i*n_nodes for i in range(batch_size)], dim=1)
            node_label = torch.cat([node_label for i in range(batch_size)], dim=0)
            self.cache[(batch_size, str(device))] = edge_index, node_label
        return [i.clone() for i in self.cache[(batch_size, str(device))]]


    def forward(self, theta, generator=None): 
        '''
        Inputs:
            theta: torch.Tensor, shape (batch_size, dim)
        Outputs:
            dtheta_dt: torch.Tensor, shape (batch_size, dim)
        '''
        
        # convert theta to edge features
        theta_node_feature = theta.reshape(-1, 1) #shape=(n_params * batch_size, 1), n_params = u.n_params = 970

        batch_size = theta.shape[0]
        edge_index, node_label  = self.get_graph(batch_size=batch_size, device=theta.device)
        node_features = theta_node_feature * node_label.float() #shape=(n_nodes * batch_size, 4), where 4 is the number of layers.

        # graph Conv
        out = self.G(node_features, edge_index) #shape = [n_nodes * batch_size, out_channels]
                       
        theta_node_feature = out[node_label] #shape = [n_params * batch_size, 1] (unchanged from input)
        dtheta_dt =  theta_node_feature.reshape(batch_size, -1) 
        if self.skip_connection:
            dtheta_dt += theta #skip connections
        
        #Dropout
        dtheta_dt = dropout(dtheta_dt, p=self.dropout_p, training=self.training, g=generator) 

        return dtheta_dt
