# some basic function 

import torch 
import math
import timeit
import re

def torch_cuda_identify(print_info = True):        
    if torch.cuda.is_available():
        if print_info:
            print(' PyTorch version:', torch.__version__)
            print(' CUDA version \t:', torch.version.cuda)
            print(' GPU cuda:({}) \t: {}'.format(torch.cuda.current_device(), torch.cuda.get_device_name()),'\n', "-" * 50)
        return torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        print('cuda is not avaliable !')
        return torch.device('cpu')
    
def time_stamp_cudasync():
    torch.cuda.synchronize()
    return timeit.default_timer()   
    
def set_dtype(ts: torch.Tensor, dtype: str):
    if dtype == "fp32":
        return ts.float()
    elif dtype == "fp16":
        return ts.half()
    raise RuntimeError(f"Unsupported dtype {dtype}")

def transpose_for_scores(x, n_heads, head_size):
    # (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
    new_x_shape = x.size()[:-1] + (n_heads, head_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

def transpose_for_scores1(x):
    new_x_shape = x.size()[:-1] + (12, 64)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

def seqlen_to_mask(lengths, max_len):
    batch_size = lengths.numel()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    return mask


def read_config_file(file_path):
    config_data = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            match = re.match(r'num_warp(\d+)\s*\|\s*m(\d+)n(\d+)\s*\|\s*bs:(\d+)\s*\|\s*seq:(\d+)\s*\|\s*([\d.]+)\s*ms/iter\s*\|\s*Speedup/FA2:\s*([\d.]+)', line)
            if match:
                num_warps = int(match.group(1))
                block_m = int(match.group(2))
                block_n = int(match.group(3))
                bs = int(match.group(4))
                seq_len = int(match.group(5))
                
                if bs not in config_data:
                    config_data[bs] = {}
                if seq_len not in config_data[bs]:
                    config_data[bs][seq_len] = []
                    
                config_data[bs][seq_len].append((block_m, block_n, num_warps))
    
    return config_data


def get_best_config(config_data, batch_size, seq_len):
    if batch_size in config_data and seq_len in config_data[batch_size]:
        configs = config_data[batch_size][seq_len]
        return configs[0]
    else:
        raise ValueError(f"Error !")

