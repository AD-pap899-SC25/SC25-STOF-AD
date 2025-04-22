import triton
import triton.language as tl
import torch
import time
import torch.nn.functional as F


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_H': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 128}),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_N': 128}),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_H': 16}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_H': 32}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_H': 64}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_H': 64}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_H': 64}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_H': 64}),
    ],
    key=['M', 'K', 'N', 'H'],
)
@triton.jit
def triton_matmul_batch_kernel(
    T_einsum_1, b, a, d, bias,
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_b = tl.program_id(axis=0)  # Batch 维度索引
    pid_x = tl.program_id(axis=1)  # 列方向线程块 ID
    pid_y = tl.program_id(axis=2)  # 行方向线程块 ID
    
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
        
        # bias_mask = n_offsets < N
        # bias_data = tl.load(bias + pid_b * N + n_offsets, mask=bias_mask, other=0.0)
        # T_einsum_2_data += bias_data[None, :]
        
        # sqrt_2_over_pi = 0.7978845608028654
        # x = T_einsum_2_data
        # x_cubed = x * x * x
        # inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        # T_einsum_2_data = 0.5 * x * (1.0 + tanh(inner))
        
        T_einsum_1_data += tl.dot(T_einsum_2_data.to(tl.float16), d_1_data)
    
    output_mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < H)
    tl.store(T_einsum_1 + pid_b * (M * H) + row_offsets[:, None] * H + col_offsets[None, :], T_einsum_1_data, mask=output_mask)


def triton_matmul_batch(a, b, d, bias):
    B, M, K = a.shape
    _, K_, N = b.shape
    _, N_, H = d.shape
    
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

# def pytorch_matmul(a, b, d):
#     # 使用爱因斯坦求和约定计算 A @ B @ D
#     c = torch.einsum('mk,kn,np->mp', a, b, d)
#     return c

def pytorch_matmul_batch(a, b, d, bias):
    c = torch.einsum('bik,bkj->bij', a, b)
    # c = c + bias
    # c = F.gelu(c)
    e = torch.einsum('bij,bjk->bik', c, d)
    return e

import itertools
import matplotlib.pyplot as plt
import numpy as np

# 性能测试函数
def benchmark_implementations():
    batch_sizes = [1,8]
    seq_lens = [128,4096]
    hidden_sizes = [512, 1024]
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
        
        # Benchmark Triton implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_triton = triton_matmul_batch(A, B, D, bias)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100
        
        # Benchmark PyTorch implementation
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
    
    # Print summary table
    print("\nPerformance Summary:")
    print("Batch | SeqLen | Hidden | Triton (ms) | PyTorch (ms) | Speedup | Max Error")
    print("-" * 80)
    for r in results:
        print(f"{r[0]:5} | {r[1]:6} | {r[2]:6} | {r[3]*1000:.3f}  | {r[4]*1000:.3f}   | {r[5]:.2f}x   | {r[6]:.6f}")
    
    # # 绘制性能对比柱状图
    # fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # 创建3行3列的子图
    # axes = axes.flatten()  # 将axes展平成一维数组
    # batch_hidden_combinations = [(b, h) for b in batch_sizes for h in hidden_sizes]
    
    # for i, (batch_size, hidden_size) in enumerate(batch_hidden_combinations):  # 9个组合
    #     ax = axes[i]
    #     subset = [r for r in results if r[0] == batch_size and r[2] == hidden_size]
    #     seq_lens = [r[1] for r in subset]
    #     triton_times = [r[3] * 1000 for r in subset]  # 转换为 ms
    #     pytorch_times = [r[4] * 1000 for r in subset]

    #     x = np.arange(len(seq_lens))  # X 轴位置
    #     width = 0.3

    #     ax.bar(x - width/2, triton_times, width, label="Triton", color="blue")
    #     ax.bar(x + width/2, pytorch_times, width, label="PyTorch", color="orange")

    #     ax.set_xticks(x)
    #     ax.set_xticklabels(seq_lens)
    #     ax.set_xlabel("Sequence Length")
    #     ax.set_ylabel("Execution Time (ms)")
    #     ax.set_title(f"Batch {batch_size}, Hidden {hidden_size}")
    #     ax.legend()

    # plt.tight_layout()
    # plt.savefig("./benchmark_results.png")
    
    return results

# 运行基准测试
benchmark_implementations()