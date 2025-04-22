// Q(B, H, S, W) @ K^T(B, H, W, S) -> mask -> softmax-> (B, H, S, S)
// (B, H, S, S) @ V(B, H, S, W) -> (B, H, S, W)   -trans-> (B, S, H, W) -merge-> (B, S, D)
//  
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Half-precision exponential function approximation
__device__ __half half_exp(__half x) {
    float float_x = __half2float(x);
    return __float2half(expf(float_x));
}

// Half-precision square root function
__device__ __half half_sqrt(__half x) {
    float float_x = __half2float(x);
    return __float2half(sqrtf(float_x));
}

// Half-precision division
__device__ __half half_divide(__half a, __half b) {
    return __float2half(__half2float(a) / __half2float(b));
}

__global__ void rowwise_attn_mask_kernel(
    const __half *q, const __half *k, const __half *v, bool is_causal, __half *result,
    const int stride_0, const int stride_1, const int stride_2, const __half *row_mask,
    const int batch_size, const int head_num, const int seq_len, const int head_size)
{
    const int batch = blockIdx.y;
    const int head = threadIdx.y + blockIdx.z * head_num / 3;

    const int row_idx = blockIdx.x;
    const int head_channel = threadIdx.x;
    const int head_channel2 = head_channel + WARP_SIZE;

    __half sum_exp_score = __float2half(0.0f);
    __half score_0 = __float2half(0.0f);
    __half score_1 = __float2half(0.0f);

    __half lane_score;
    __half exp_score;

    int offset_res = head_num * seq_len * head_size * batch + head_num * head_size * row_idx + head_size * head;
    int offset_common = stride_0 * batch + stride_1 * head;
    int offset_q = offset_common + stride_2 * row_idx;

    int right_bound = is_causal ? row_idx : seq_len - 1;

    for (int col = 0; col <= right_bound; ++col) 
    {
        if (__hgt(row_mask[row_idx * seq_len + col], __float2half(0.0f))) // 如果对应位置非 0 
        {
            int offset_kv = offset_common + stride_2 * col;

            lane_score = __hmul(q[offset_q + head_channel], k[offset_kv + head_channel]);
            lane_score = __hadd(lane_score, __hmul(q[offset_q + head_channel2], k[offset_kv + head_channel2]));
            
            for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
            {
                lane_score = __hadd(lane_score, __shfl_xor_sync(0xffffffff, lane_score, i, WARP_SIZE));
            }

            if (head_channel == 0)
            {
                exp_score = half_exp(__hdiv(lane_score, half_sqrt(__float2half(static_cast<float>(head_size)))));
                sum_exp_score = __hadd(sum_exp_score, exp_score);
            }
            exp_score = __shfl_sync(0xffffffff, exp_score, 0);

            // (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W) -merge-> (B, S, D)
            score_0 = __hadd(score_0, __hmul(exp_score, v[offset_kv + head_channel]));
            score_1 = __hadd(score_1, __hmul(exp_score, v[offset_kv + head_channel2]));
        }
    }
    sum_exp_score = __shfl_sync(0xffffffff, sum_exp_score, 0);

    result[offset_res + head_channel] = half_divide(score_0, sum_exp_score);
    result[offset_res + head_channel2] = half_divide(score_1, sum_exp_score);
}


void launcher_rowwise_attn_mask(
    const __half* q, const __half* k, const __half* v, bool is_causal, __half* result, 
    const int stride_0, const int stride_1, const int stride_2, const __half* row_mask,
    const int batch_size, const int head_num, const int seq_len, const int head_size)
{
    
    dim3 blockSize(head_size / 2, head_num / 3);
    dim3 gridSize(seq_len, batch_size, 3);
    
    rowwise_attn_mask_kernel<<<gridSize, blockSize>>>(
        q, k, v, is_causal, result,
        stride_0, stride_1, stride_2, row_mask,
        batch_size, head_num, seq_len, head_size);
}
