import torch
import triton
import triton.language as tl
import time
import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['M'],
)

@triton.jit
def batched_bias_layer_norm_fused_kernel(
    X, Y, Bias,   
    W, B,  
    Mean, Rstd, 
    batch_stride, seq_stride, hidden_size, eps,
    M,
    BLOCK_SIZE: tl.constexpr,
):

    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    batch_offset = batch_id * batch_stride
    seq_offset = seq_id * seq_stride
    
    x_ptr = X + batch_offset + seq_offset
    y_ptr = Y + batch_offset + seq_offset
    
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        bias = tl.load(Bias + cols, mask=mask, other=0.).to(tl.float32)
        biased_val = x + bias
        _mean += tl.where(mask, biased_val, 0.)
    mean = tl.sum(_mean, axis=0) / hidden_size
    

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        bias = tl.load(Bias + cols, mask=mask, other=0.).to(tl.float32)
        biased_val = x + bias
        diff = tl.where(mask, biased_val - mean, 0.)
        _var += diff * diff
    var = tl.sum(_var, axis=0) / hidden_size
    rstd = 1 / tl.sqrt(var + eps)
    

    mean_idx = batch_id * seq_stride + seq_id
    tl.store(Mean + mean_idx, mean)
    tl.store(Rstd + mean_idx, rstd)
    

    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        bias = tl.load(Bias + cols, mask=mask, other=0.).to(tl.float32)

        biased_val = x + bias
        x_hat = (biased_val - mean) * rstd
        y = x_hat * w + b
        tl.store(y_ptr + cols, y, mask=mask)


def triton_batched_bias_layernorm(x, bias, weight, bias_ln, eps=1e-5):

    batch_size, seq_len, hidden_size = x.shape
    M = seq_len
    y = torch.empty_like(x)
    mean = torch.empty((batch_size * seq_len,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((batch_size * seq_len,), dtype=torch.float32, device=x.device)

    grid = (batch_size, seq_len)
    
    batched_bias_layer_norm_fused_kernel[grid](
        x, y, bias, weight, bias_ln, mean, rstd,
        x.stride(0), x.stride(1), hidden_size, eps, 
        M , 
 )
    return y

def pytorch_batched_bias_layernorm(x, bias, weight, bias_ln, eps=1e-5):
    x_biased = x + bias
    norm_shape = (x.shape[2],)
    y = torch.nn.functional.layer_norm(x_biased, norm_shape, weight, bias_ln, eps)
    return y

import torch
import time
import itertools
import matplotlib.pyplot as plt
import numpy as np

def benchmark_implementations():
    batch_sizes = [1,8]
    seq_lens = [4096]
    hidden_sizes = [512]
    
    results = []
    
    for batch_size, hidden_size, seq_len in itertools.product(batch_sizes, hidden_sizes, seq_lens ):
        print(f"Benchmarking batch={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
        
        x = torch.randn((batch_size, seq_len, hidden_size), device="cuda", dtype=torch.float16)
        bias = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
        weight = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
        bias_ln = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
        

        for _ in range(10):
            y_triton = triton_batched_bias_layernorm(x, bias, weight, bias_ln)
            y_pytorch = pytorch_batched_bias_layernorm(x, bias, weight, bias_ln)
        
        y_triton = triton_batched_bias_layernorm(x, bias, weight, bias_ln)
        y_pytorch = pytorch_batched_bias_layernorm(x, bias, weight, bias_ln)
        max_error = torch.max(torch.abs(y_triton - y_pytorch)).item()
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_triton = triton_batched_bias_layernorm(x, bias, weight, bias_ln)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_pytorch = pytorch_batched_bias_layernorm(x, bias, weight, bias_ln)
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


benchmark_implementations()


