import torch
import triton
import triton.language as tl
import time

@triton.jit
def batched_bias_layer_norm_fused_kernel(
    X, Y, Bias,   
    W, B,  
    Mean, Rstd, 
    batch_stride, seq_stride, hidden_size, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Map program ID to the batch and sequence position
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    # Calculate starting positions
    batch_offset = batch_id * batch_stride
    seq_offset = seq_id * seq_stride
    
    # Pointer to current sequence position
    x_ptr = X + batch_offset + seq_offset
    y_ptr = Y + batch_offset + seq_offset
    
    # Calculate mean (including bias)
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        bias = tl.load(Bias + cols, mask=mask, other=0.).to(tl.float32)
        # Apply bias
        biased_val = x + bias
        _mean += tl.where(mask, biased_val, 0.)
    mean = tl.sum(_mean, axis=0) / hidden_size
    
    # Calculate variance (including bias)
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        bias = tl.load(Bias + cols, mask=mask, other=0.).to(tl.float32)
        # Apply bias
        biased_val = x + bias
        diff = tl.where(mask, biased_val - mean, 0.)
        _var += diff * diff
    var = tl.sum(_var, axis=0) / hidden_size
    rstd = 1 / tl.sqrt(var + eps)
    
    # Write mean / rstd
    mean_idx = batch_id * seq_stride + seq_id
    tl.store(Mean + mean_idx, mean)
    tl.store(Rstd + mean_idx, rstd)
    
    # Normalize and apply linear transformation (using biased values)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        bias = tl.load(Bias + cols, mask=mask, other=0.).to(tl.float32)
        # Apply bias, normalize, then apply layernorm scale and shift
        biased_val = x + bias
        x_hat = (biased_val - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(y_ptr + cols, y, mask=mask)

def triton_batched_bias_layernorm(x, bias, weight, bias_ln, eps=1e-5):
    # Check input shapes
    batch_size, seq_len, hidden_size = x.shape
    
    # Allocate output memory
    y = torch.empty_like(x)
    mean = torch.empty((batch_size * seq_len,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((batch_size * seq_len,), dtype=torch.float32, device=x.device)
    
    # Determine block size
    BLOCK_SIZE = min(128, triton.next_power_of_2(hidden_size))
    
    # Determine grid and block size
    grid = (batch_size, seq_len)
    
    # Call kernel
    batched_bias_layer_norm_fused_kernel[grid](
        x, y, bias, weight, bias_ln, mean, rstd,
        x.stride(0), x.stride(1), hidden_size, eps, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def pytorch_batched_bias_layernorm(x, bias, weight, bias_ln, eps=1e-5):
    x_biased = x + bias
    norm_shape = (x.shape[2],)  # Normalize over hidden dimension
    y = torch.nn.functional.layer_norm(x_biased, norm_shape, weight, bias_ln, eps)
    return y

import torch
import time
import itertools
import matplotlib.pyplot as plt
import numpy as np

def benchmark_implementations():
    batch_sizes = [1,8]
    seq_lens = [128,4096]
    hidden_sizes = [512, 1024]
    
    results = []
    
    for hidden_size, batch_size,seq_len in itertools.product(hidden_sizes, batch_sizes, seq_lens ):
        print(f"Benchmarking hidden_size={hidden_size}, batch={batch_size}, seq_len={seq_len}")
        
        # Create random input data
        x = torch.randn((batch_size, seq_len, hidden_size), device="cuda", dtype=torch.float16)
        bias = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
        weight = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
        bias_ln = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
        
        # Warm up
        for _ in range(10):
            y_triton = triton_batched_bias_layernorm(x, bias, weight, bias_ln)
            y_pytorch = pytorch_batched_bias_layernorm(x, bias, weight, bias_ln)
        
        # Verify correctness
        y_triton = triton_batched_bias_layernorm(x, bias, weight, bias_ln)
        y_pytorch = pytorch_batched_bias_layernorm(x, bias, weight, bias_ln)
        max_error = torch.max(torch.abs(y_triton - y_pytorch)).item()
        
        # Benchmark Triton implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_triton = triton_batched_bias_layernorm(x, bias, weight, bias_ln)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100
        
        # Benchmark PyTorch implementation
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
    
    # Print summary table
    print("\nPerformance Summary:")
    print("Batch | SeqLen | Hidden | Triton (ms) | PyTorch (ms) | Speedup | Max Error")
    print("-" * 80)
    for r in results:
        print(f"{r[0]:5} | {r[1]:6} | {r[2]:6} | {r[3]*1000:.3f}  | {r[4]*1000:.3f}   | {r[5]:.2f}x   | {r[6]:.6f}")
    
    # 绘制性能对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(18, 15))  # 创建3行3列的子图
    axes = axes.flatten()  # 将axes展平成一维数组
    batch_hidden_combinations = [(b, h) for b in batch_sizes for h in hidden_sizes]
    
    for i, (batch_size, hidden_size) in enumerate(batch_hidden_combinations):  # 9个组合
        ax = axes[i]
        subset = [r for r in results if r[0] == batch_size and r[2] == hidden_size]
        seq_lens = [r[1] for r in subset]
        triton_times = [r[3] * 1000 for r in subset]  # 转换为 ms
        pytorch_times = [r[4] * 1000 for r in subset]

        x = np.arange(len(seq_lens))  # X 轴位置
        width = 0.3

        ax.bar(x - width/2, triton_times, width, label="Triton", color="blue")
        ax.bar(x + width/2, pytorch_times, width, label="PyTorch", color="orange")

        ax.set_xticks(x)
        ax.set_xticklabels(seq_lens)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Execution Time (ms)")
        ax.set_title(f"Batch {batch_size}, Hidden {hidden_size}")
        ax.legend()

    plt.tight_layout()
    plt.savefig("./benchmark_results.png")
    
    return results

# 运行基准测试
benchmark_implementations()


