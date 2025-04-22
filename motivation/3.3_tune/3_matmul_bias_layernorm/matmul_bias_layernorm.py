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
    # 获取线程块索引
    pid_batch = tl.program_id(0)  # 批处理维度的线程块ID
    pid_m = tl.program_id(1)      # M维度的线程块ID
    # 计算当前线程块处理的批次和行索引
    batch_idx = pid_batch
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, N)
    # 计算当前批次的A和B的基础指针
    A_batch_ptr = A_ptr + batch_idx * stride_ab
    # B_batch_ptr = B_ptr + batch_idx * stride_bb

    B_batch_ptr = B_ptr
    # 计算当前块的A和B指针
    A_block_ptr = A_batch_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_block_ptr = B_batch_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn    
    # 初始化累加器
    C_accum = tl.zeros((BLOCK_SIZE_M, N), dtype=tl.float32)
    # 执行矩阵乘法
    for k in range(0, K, BLOCK_SIZE_K):
        # 加载A和B块
        A_block = tl.load(A_block_ptr, mask=offs_m[:, None] < M, other=0.0)
        B_block = tl.load(B_block_ptr, mask=offs_k[:, None] < K, other=0.0)
        # 执行点积并累加
        C_accum += tl.dot(A_block, B_block)
        # 更新指针到下一个块
        A_block_ptr += BLOCK_SIZE_K * stride_ak
        B_block_ptr += BLOCK_SIZE_K * stride_bk
    # 加载偏置并添加到结果
    Bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    C_accum += Bias[None, :]
    # 层归一化: 首先计算均值
    mean = tl.sum(C_accum, axis=1) / N
    mean = mean[:, None]  # 将均值转换为列向量，便于广播
    # 计算方差
    var = tl.sum((C_accum - mean) * (C_accum - mean), axis=1) / N
    var = var[:, None]
    # 应用层归一化公式
    C_accum = (C_accum - mean) / tl.sqrt(var + eps)

    C_ptrs = C_ptr + batch_idx * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, C_accum, mask=offs_m[:, None] < M)
    
def triton_matmul_bias_layernorm(A, B, bias, eps=1e-5):
    batch_size, M, K = A.shape
    # _, K_b, N = B.shape
    K_b, N = B.shape

    # 确保K维度匹配
    assert K == K_b, f"矩阵乘法维度不匹配: A的K={K}, B的K={K_b}"
    # 分配输出张量
    C = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)
    # 计算网格尺寸（批次数量和M维度的块数）
    grid = lambda META: (batch_size, triton.cdiv(M, META['BLOCK_SIZE_M']))
    # 调用Triton内核
    batch_matmul_bias_layernorm_kernel[grid](
        A, B, bias, C,
        batch_size, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        eps,
    )
    return C

# def pytorch_matmul_bias_layernorm(A, B, bias, eps=1e-5):
#     C = torch.bmm(A, B) + bias
#     return torch.nn.functional.layer_norm(C, normalized_shape=C.shape[1:], eps=eps)

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
        
        # Create random input data
        A = torch.randn((batch_size, seq_len, hidden_size*4), device="cuda", dtype=torch.float16)
        B = torch.randn((hidden_size*4, hidden_size), device="cuda", dtype=torch.float16)
        bias = torch.randn((hidden_size), device="cuda", dtype=torch.float16)
        
        # Warm up
        for _ in range(10):
            y_triton = triton_matmul_bias_layernorm(A, B, bias)
            y_pytorch = pytorch_matmul_bias_layernorm(A, B, bias)
        
        # Verify correctness
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