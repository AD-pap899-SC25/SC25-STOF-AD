import torch
import triton
import triton.language as tl
import time
import os 
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

configs = [
    triton.Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": block_k},
        # num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_m in [32, 64, 128]
    for block_k in [32, 64, 128]
    for num_warps in [4]
    # for num_stages in [4, 6]
]
# configs = [
#     triton.Config(
#         {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": block_k},
#         # num_stages=num_stages,
#         num_warps=num_warps,
#     )
#     for block_m in [128]
#     for block_k in [32]
#     for num_warps in [4]
#     # for num_stages in [4, 6]
# ]
@triton.autotune(configs=configs, key=["BATCH", "M", "N", "K"])

@triton.jit
def batch_matmul_bias_layernorm_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    BATCH, M, N: tl.constexpr, K,
    stride_ab, stride_am, stride_ak,
    # stride_bb, 
    stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_batch = tl.program_id(0) 
    pid_m = tl.program_id(1)   

    batch_idx = pid_batch
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, N)
    A_batch_ptr = A_ptr + batch_idx * stride_ab

    B_batch_ptr = B_ptr
    A_block_ptr = A_batch_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_block_ptr = B_batch_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn    
    C_accum = tl.zeros((BLOCK_SIZE_M, N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        A_block = tl.load(A_block_ptr, mask=offs_m[:, None] < M, other=0.0)
        B_block = tl.load(B_block_ptr, mask=offs_k[:, None] < K, other=0.0)
        C_accum += tl.dot(A_block, B_block)
        A_block_ptr += BLOCK_SIZE_K * stride_ak
        B_block_ptr += BLOCK_SIZE_K * stride_bk

    Bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    C_accum += Bias[None, :]
    mean = tl.sum(C_accum, axis=1) / N
    mean = mean[:, None]

    var = tl.sum((C_accum - mean) * (C_accum - mean), axis=1) / N
    var = var[:, None]
    C_accum = (C_accum - mean) / tl.sqrt(var + eps)

    C_ptrs = C_ptr + batch_idx * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, C_accum, mask=offs_m[:, None] < M)
    
def triton_matmul_bias_layernorm(A, B, bias, eps=1e-5):
    batch_size, M, K = A.shape
    # _, K_b, N = B.shape
    K_b, N = B.shape


    assert K == K_b, f"Miss Match: A的K={K}, B的K={K_b}"

    C = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)

    grid = lambda META: (batch_size, triton.cdiv(M, META['BLOCK_SIZE_M']))
    batch_matmul_bias_layernorm_kernel[grid](
        A, B, bias, C,
        batch_size, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        eps,
    )
    return C



def pytorch_matmul_bias_layernorm(A, B, bias, eps=1e-5):
    C = torch.einsum('bik,kj->bij', A, B) + bias
    return torch.nn.functional.layer_norm(C, normalized_shape=C.shape[1:], eps=eps)

import itertools
import matplotlib.pyplot as plt
import numpy as np

def benchmark_implementations():
    batch_sizes = [1,8,16]
    seq_lens = [128,256,512,1024,2048,4096]
    hidden_sizes = [512, 1024]
    results = []
    
    for batch_size, hidden_size, seq_len in itertools.product(batch_sizes, hidden_sizes, seq_lens ):
        print(f"Benchmarking batch={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
        
        A = torch.randn((batch_size, seq_len, hidden_size*4), device="cuda", dtype=torch.float16)
        B = torch.randn((hidden_size*4, hidden_size), device="cuda", dtype=torch.float16)
        bias = torch.randn((hidden_size), device="cuda", dtype=torch.float16)
        
        for _ in range(10):
            y_triton = triton_matmul_bias_layernorm(A, B, bias)
            y_pytorch = pytorch_matmul_bias_layernorm(A, B, bias)
        
        y_triton = triton_matmul_bias_layernorm(A, B, bias)
        y_pytorch = pytorch_matmul_bias_layernorm(A, B, bias)
        max_error = torch.max(torch.abs(y_triton - y_pytorch)).item()
        
        # Benchmark Triton implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_triton = triton_matmul_bias_layernorm(A, B, bias)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100
        
        # Benchmark PyTorch implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_pytorch = pytorch_matmul_bias_layernorm(A, B, bias)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 100
        speedup = pytorch_time / triton_time
        results.append((batch_size, seq_len, hidden_size, triton_time, pytorch_time, speedup, max_error))
        
        print(f"Triton time: {triton_time*1000:.3f} ms")
        print(f"PyTorch time: {pytorch_time*1000:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Max absolute error: {max_error}\n")
    
    # Print summary table
    print("\nPerformance Summary:")
    print("Batch | SeqLen | Hidden | Triton (ms) | PyTorch (ms) | Speedup | Max Error")
    print("-" * 80)
    for r in results:
        print(f"{r[0]:5} | {r[1]:6} | {r[2]:6} | {r[3]*1000:.3f}  | {r[4]*1000:.3f}   | {r[5]:.2f}x   | {r[6]:.6f}")

    
    return results

benchmark_implementations()