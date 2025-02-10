import sys
import time
from functools import wraps
import os
import datetime
from socket import gethostname
import random
from operator import eq
import pickle

import numpy as np
import torch 
import pynvml   #pip install nvidia-ml-py3



def set_gpu_max_mem(default_device=0, force=False):
    '''Set GPU with maximum free memory.'''
    if not force:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            free_memory = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory.append((i, mem_info.free))
            
            pynvml.nvmlShutdown()

            max_free_device = max(free_memory, key=lambda x: x[1])[0]
            torch.cuda.set_device(max_free_device)
            print(f"Selected GPU {max_free_device} with maximum free memory.", flush=True)
        except:
            print(f"Failed auto set GPU, use default GPU {default_device}.", flush=True)
            torch.cuda.set_device(default_device)
    else:
        torch.cuda.set_device(default_device)
        print(f"Use default GPU {default_device}.", flush=True)



def set_seed(seed=None, cudnn_benchmark=True):
    if seed is None:
        seed = os.getpid()
    print(f"All {seed=}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cudnn_benchmark  #set True for acceleration


def list_cat(ll):
    '''joint a iterable of lists into a single list.
    '''
    return sum(ll, [])

def save_load(filename, obj=None, is_save=True):
    folder = os.path.split(filename)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

    if is_save:
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
    else:
        with open(filename, 'rb') as file:
            return pickle.load(file)


def switch(value, comp=eq):
    return lambda match: comp(match, value)


def time_to_str(dt):
    if dt < 60:
        t = "{:.4f} sec".format(dt)
    elif dt < 3600:
        t = "{:.4f} min".format(dt/60)
    else:
        t = "{:.4f} hour".format(dt/3600)
    return t

def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        dt = te - ts
        t = time_to_str(dt)

        print("%r took %s " % (f.__name__, t))
        sys.stdout.flush()
        return result

    return wrapper

def timing_with_return(f):
    """Decorator for measuring the execution time of methods, added to fun return."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        dt = te - ts
        t = time_to_str(dt)

        print("%r took %s " % (f.__name__, t))
        sys.stdout.flush()
        return result, dt

    return wrapper

    
def redirect_log_file(log_root = "./log", exp_name=""):
    t = str(datetime.datetime.now())

    if exp_name == "":
        exp_name = sys.argv[0].split("/")[-1].split(".")[0]  #filename without extension
    
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    out_file = os.path.join(log_root, t[2:][:-7] + " || " + exp_name + ".txt")
    
    print("Redirect log to: ", out_file)
    print("pid=", os.getpid(), flush=True)
    sys.stdout = open(out_file, 'a', buffering=30000)
    sys.stderr = open(out_file, 'a', buffering=30000)
    print("Start time:", t)
    print("Script name:", sys.argv[0])
    print("Running at:", gethostname(), "pid=", os.getpid(), flush=True)
    return out_file