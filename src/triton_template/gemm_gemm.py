import triton
import triton.language as tl
import torch
import time
import torch.nn.functional as F
import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
from triton.runtime import driver

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

def early_config_prune(configs, named_args, **kwargs):
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability()
    A = named_args['a'] 
    
    pruned = []
    for cfg in configs:
        BLOCK_M = cfg.kwargs['BLOCK_SIZE_M']
        BLOCK_H = cfg.kwargs['BLOCK_SIZE_H']
        BLOCK_K = cfg.kwargs['BLOCK_SIZE_K']
        BLOCK_N = cfg.kwargs['BLOCK_SIZE_N']
        num_warps = cfg.num_warps
        

        if (BLOCK_M * BLOCK_K) > 1024 or (BLOCK_N * BLOCK_H) > 1024:
            continue
            
        max_shared = driver.active.utils.get_device_properties(device)["max_shared_mem"]

        gemm1_shared = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * A.element_size() * 2  
        gemm2_shared = (BLOCK_M * BLOCK_N + BLOCK_N * BLOCK_H) * A.element_size() * 2  
        if (gemm1_shared + gemm2_shared) > max_shared * 0.8: 
            continue
            
        if A.dtype == torch.float16:
            if BLOCK_K % 16 != 0 or BLOCK_N % 16 != 0:
                continue
            if capability[0] >= 8 and (BLOCK_M % 8 != 0 or BLOCK_H % 8 != 0):
                continue 
                
        pruned.append(cfg)
        
    if capability[0] >= 8:
        # Ampere+
        config_groups = {}
        for cfg in pruned:
            key = (cfg.kwargs['BLOCK_SIZE_M'], cfg.kwargs['BLOCK_SIZE_N'], num_warps)
            config_groups.setdefault(key, []).append(cfg)
        
        final_configs = []
        for group in config_groups.values():

            comp_intensity = []
            for cfg in group:
                m, n, k, h = [cfg.kwargs[f'BLOCK_SIZE_{d}'] for d in ['M','N','K','H']]
                ops = 2 * m * n * k + 2 * m * h * n
                mem_access = (m*k + k*n + m*n) + (m*n + n*h + m*h)
                comp_intensity.append(ops / mem_access)
            
            best_indices = sorted(range(len(comp_intensity)), key=lambda i: -comp_intensity[i])[:3]
            final_configs.extend([group[i] for i in best_indices])
            
        return final_configs
    else:
        return [cfg for cfg in pruned if cfg.num_stages <= 2 and cfg.num_warps >= 4]


configs = [
    triton.Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_H": block_h, 'BLOCK_SIZE_K': block_k, 'BLOCK_SIZE_N': block_n},
        num_warps = num_warps
    )
    for block_m in [16, 32, 64, 128]
    for block_h in [32, 64, 128]
    for block_k in [32, 64, 128]
    for block_n in [16, 32, 64, 128]
    for num_warps in [1, 2, 4]   
]


@triton.autotune(
    configs=configs,
    
    key=['M', 'K', 'N', 'H'],
    
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': None,
        'top_k': 5
    },
    warmup=10,
    rep=20
)


@triton.jit
def triton_matmul_batch_kernel(
    T_einsum_1, b, a, d, bias,
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_b = tl.program_id(axis=0)  
    pid_x = tl.program_id(axis=1)  
    pid_y = tl.program_id(axis=2)  
    
    row_start = pid_y * BLOCK_SIZE_M
    col_start = pid_x * BLOCK_SIZE_H
    
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_H)
    
    T_einsum_1_data = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_H], dtype=tl.float32)
    
    n_blocks = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    for n_idx in range(n_blocks):
        n_offset = n_idx * BLOCK_SIZE_N
        n_offsets = n_offset + tl.arange(0, BLOCK_SIZE_N)
        
        d_mask = (n_offsets[:, None] < N) & (col_offsets[None, :] < H)
        d_1_data = tl.load(d + pid_b * (N * H) + n_offsets[:, None] * H + col_offsets[None, :], mask=d_mask, other=0.0)
        
        T_einsum_2_data = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        
        k_blocks = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        for k_idx in range(k_blocks):
            k_offset = k_idx * BLOCK_SIZE_K
            k_offsets = k_offset + tl.arange(0, BLOCK_SIZE_K)
            
            a_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
            a_1_data = tl.load(a + pid_b * (M * K) + row_offsets[:, None] * K + k_offsets[None, :], mask=a_mask, other=0.0)
            
            b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
            b_1_data = tl.load(b + pid_b * (K * N) + k_offsets[:, None] * N + n_offsets[None, :], mask=b_mask, other=0.0)
            
            T_einsum_2_data += tl.dot(a_1_data, b_1_data)
        
        bias_mask = n_offsets < N
        bias_data = tl.load(bias + pid_b * N + n_offsets, mask=bias_mask, other=0.0)
        T_einsum_2_data += bias_data[None, :]
        
        sqrt_2_over_pi = 0.7978845608028654
        x = T_einsum_2_data
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        T_einsum_2_data = 0.5 * x * (1.0 + tanh(inner))
        
        T_einsum_1_data += tl.dot(T_einsum_2_data.to(tl.float16), d_1_data)
    
    output_mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < H)
    tl.store(T_einsum_1 + pid_b * (M * H) + row_offsets[:, None] * H + col_offsets[None, :], T_einsum_1_data, mask=output_mask)


def triton_matmul_batch(a, b, d, bias):
    B, M, K = a.shape
    _, N = b.shape
    _, H = d.shape
    
    grid = lambda META: (
        B,
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(H, META['BLOCK_SIZE_H'])
    )
    output = torch.empty((B, M, H), device=a.device, dtype=torch.float16)
    
    triton_matmul_batch_kernel[grid](
        output, b, a, d, bias,
        B, M, K, N, H,
    )
    return output



def pytorch_matmul_batch(a, b, d, bias):
    c = torch.einsum('bik,bkj->bij', a, b)
    c = c + bias
    c = F.gelu(c)
    e = torch.einsum('bij,bjk->bik', c, d)
    return e

import itertools
import matplotlib.pyplot as plt
import numpy as np


def benchmark_implementations():
    batch_sizes = [1, 8, 16]
    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    hidden_sizes = [512]
    results = []
    
    for batch_size, hidden_size, seq_len in itertools.product(batch_sizes, hidden_sizes, seq_lens ):
        print(f"Benchmarking batch={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
        
        # Create random input data
        A = torch.randn((batch_size, seq_len, hidden_size), device="cuda", dtype=torch.float16)
        B = torch.randn((batch_size, hidden_size, 4*hidden_size), device="cuda", dtype=torch.float16)
        D = torch.randn((batch_size, 4*hidden_size, hidden_size), device="cuda", dtype=torch.float16)
        bias = torch.randn((hidden_size*4), device="cuda", dtype=torch.float16)
        
        # Warm up
        for _ in range(10):
            y_triton = triton_matmul_batch(A, B, D, bias)
            y_pytorch = pytorch_matmul_batch(A, B, D, bias)
        
        # Verify correctness
        y_triton = triton_matmul_batch(A, B, D, bias)
        y_pytorch = pytorch_matmul_batch(A, B, D, bias)
        max_error = torch.max(torch.abs(y_triton - y_pytorch)).item()
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_triton = triton_matmul_batch(A, B, D, bias)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_pytorch = pytorch_matmul_batch(A, B, D, bias)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 100
        speedup = pytorch_time / triton_time
        results.append((batch_size, seq_len, hidden_size, triton_time, pytorch_time, speedup, max_error))
        
        print(f"Triton time: {triton_time*1000:.3f} ms")
        print(f"PyTorch time: {pytorch_time*1000:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Max absolute error: {max_error}\n")
    
    print("\nPerformance Summary:")
    print("Batch | SeqLen | Hidden | Triton (ms) | PyTorch (ms) | Speedup | Max Error")
    print("-" * 80)
    for r in results:
        print(f"{r[0]:5} | {r[1]:6} | {r[2]:6} | {r[3]*1000:.3f}  | {r[4]*1000:.3f}   | {r[5]:.2f}x   | {r[6]:.6f}")
    
    return results

if __name__ == "__main__":
    benchmark_implementations()