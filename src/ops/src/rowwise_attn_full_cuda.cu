// 2025.03.16 Sun
//  
// 统一思想，认为这种算子是 row 上和 col 上分块比较极端的情况
// 
//  
// Attention:
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

__global__ void rowwise_attn_full_kernel(
    const __half *q, const __half *k, const __half *v, bool is_causal, __half *result,
    const int stride_0, const int stride_1, const int stride_2,
    const int batch_size, const int head_num, const int seq_len, const int head_size)
{
    const int batch = blockIdx.y;
    // const int head = threadIdx.y + blockIdx.z * head_num / 3;
    const int head = blockIdx.z;

    const int row_idx = blockIdx.x;
    const int head_channel = threadIdx.x;
    const int head_channel2 = head_channel + WARP_SIZE;

    float sum_exp_score = 0.0f;
    float score = 0.0f;
    float score2 = 0.0f;

    float lane_score, exp_score;
    float lane_score2, exp_score2;
    float lane_score3, exp_score3;
    float lane_score4, exp_score4;

    int offset_res = head_num * seq_len * head_size * batch + head_num * head_size * row_idx + head_size * head;
    int offset_common = stride_0 * batch + stride_1 * head;
    int offset_q = offset_common + stride_2 * row_idx;

    int right_bound = is_causal ? row_idx : seq_len - 1;

    float q_head_reg = __half2float(q[offset_q + head_channel]);
    float q_head_reg2 = __half2float(q[offset_q + head_channel2]);
    
    for (int col = 0; col <= right_bound;) 
    {
        if (col + 3 <= right_bound)
        {
            int offset_kv = offset_common + stride_2 * col;
            lane_score = q_head_reg * __half2float(k[offset_kv + head_channel]);
            lane_score += q_head_reg2 * __half2float(k[offset_kv + head_channel2]);

            int offset_kv2 = offset_common + stride_2 * (col + 1);
            lane_score2 =  q_head_reg * __half2float(k[offset_kv2 + head_channel]);
            lane_score2 += q_head_reg2 * __half2float(k[offset_kv2 + head_channel2]);

            int offset_kv3 = offset_common + stride_2 * (col + 2);
            lane_score3 =  q_head_reg * __half2float(k[offset_kv3 + head_channel]);
            lane_score3 += q_head_reg2 * __half2float(k[offset_kv3 + head_channel2]);

            int offset_kv4 = offset_common + stride_2 * (col + 3);
            lane_score4 =  q_head_reg * __half2float(k[offset_kv4 + head_channel]);
            lane_score4 += q_head_reg2 * __half2float(k[offset_kv4 + head_channel2]);
            

            for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
            {
                lane_score +=  __shfl_xor_sync(0xffffffff, lane_score, i, WARP_SIZE);
                lane_score2 +=  __shfl_xor_sync(0xffffffff, lane_score2, i, WARP_SIZE);
                lane_score3 +=  __shfl_xor_sync(0xffffffff, lane_score3, i, WARP_SIZE);
                lane_score4 +=  __shfl_xor_sync(0xffffffff, lane_score4, i, WARP_SIZE);
            }

            if (head_channel == 0)
            {
                exp_score = __expf(__fdividef(lane_score, 8.0f));
                exp_score2 = __expf(__fdividef(lane_score2, 8.0f));
                exp_score3 = __expf(__fdividef(lane_score3, 8.0f));
                exp_score4 = __expf(__fdividef(lane_score4, 8.0f));
                sum_exp_score += (exp_score + exp_score2 + exp_score3 + exp_score4);
            }
            exp_score = __shfl_sync(0xffffffff, exp_score, 0);
            exp_score2 = __shfl_sync(0xffffffff, exp_score2, 0);
            exp_score3 = __shfl_sync(0xffffffff, exp_score3, 0);
            exp_score4 = __shfl_sync(0xffffffff, exp_score4, 0);

            // (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W) -merge-> (B, S, D)
            score += (exp_score * __half2float(v[offset_kv + head_channel])
                        + exp_score2 * __half2float(v[offset_kv2 + head_channel])
                        + exp_score3 * __half2float(v[offset_kv3 + head_channel])
                        + exp_score4 * __half2float(v[offset_kv4 + head_channel]));
            score2 += (exp_score * __half2float(v[offset_kv + head_channel2])+ 
                        exp_score2 * __half2float(v[offset_kv2 + head_channel2])+
                        exp_score3 * __half2float(v[offset_kv3 + head_channel2])+
                        exp_score4 * __half2float(v[offset_kv4 + head_channel2]));

            col += 4;
        }

        else
        {
            int offset_kv = offset_common + stride_2 * col;
            lane_score = q_head_reg * __half2float(k[offset_kv + head_channel]);
            lane_score += q_head_reg2 * __half2float(k[offset_kv + head_channel2]);
            
            for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
            {
                lane_score +=  __shfl_xor_sync(0xffffffff, lane_score, i, WARP_SIZE);
            }

            if (head_channel == 0)
            {
                exp_score = __expf(__fdividef(lane_score, 8.0f));
                sum_exp_score += exp_score;
            }
            exp_score = __shfl_sync(0xffffffff, exp_score, 0);

            // (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W) -merge-> (B, S, D)
            score += exp_score * __half2float(v[offset_kv + head_channel]);
            score2 += exp_score * __half2float(v[offset_kv + head_channel2]);

            col++;
        }
        
    }
    sum_exp_score = __shfl_sync(0xffffffff, sum_exp_score, 0);

    result[offset_res + head_channel] = __float2half(__fdividef(score, sum_exp_score));
    result[offset_res + head_channel2] = __float2half(__fdividef(score2, sum_exp_score));
}


void launcher_rowwise_attn_full(
    const __half* q, const __half* k, const __half* v, bool is_causal, __half* result,
    const int stride_0, const int stride_1, const int stride_2,
    const int batch_size, const int head_num, const int seq_len, const int head_size)
{
    
    dim3 blockSize(head_size / 2);
    dim3 gridSize(seq_len, batch_size, head_num);

    // dim3 blockSize(head_size / 2, head_num / 3);
    // dim3 gridSize(seq_len, batch_size, 3);
    
    rowwise_attn_full_kernel<<<gridSize, blockSize>>>(
        q, k, v, is_causal, result,
        stride_0, stride_1, stride_2,
        batch_size, head_num, seq_len, head_size);
}
