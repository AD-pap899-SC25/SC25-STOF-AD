#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <stdio.h>
#include <string.h>
#include "include/attention.h"
#include "include/reduce.h"

using namespace nvcuda;
using namespace bytetransformer;

#define SKEW_HALF 8 // offset for avoding bank conflict
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <const int max_seq_len, const int size_per_head, const int split_seq_len>
__global__ void bytetr_attn_kernel(const half2 *qkv, const half2 *qkv_bias,
                                           const __half *attention_mask, __half *attention_output,
                                           const int seq_len, const float scale)
{

  using namespace nvcuda;
  extern __shared__ __half base[];
  __half(*s_kv)[size_per_head + SKEW_HALF] = (__half(*)[size_per_head + SKEW_HALF]) base;
  __half(*s_query)[size_per_head + SKEW_HALF] =
      (__half(*)[size_per_head + SKEW_HALF])(base + (max_seq_len) * (size_per_head + SKEW_HALF));
  __half(*s_logits)[max_seq_len + SKEW_HALF] = (__half(*)[max_seq_len + SKEW_HALF])(
      base + (split_seq_len + max_seq_len) * (size_per_head + SKEW_HALF));

  const int warpNums = (split_seq_len / 16) * (max_seq_len / 16); //(blockDim.x  >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  const int thread_offset = blockIdx.x * (size_per_head / 2) + warp_tid;

  const int batch_seq_offset = blockIdx.z * seq_len;
  const int block_seq_len = min(split_seq_len, seq_len - (int)blockIdx.y * split_seq_len);
  const int batch_seq_block_offset = batch_seq_offset + blockIdx.y * split_seq_len;

  const int from_size = split_seq_len / 16;
  const int to_size = max_seq_len / 16;

  // loading Query
  half2 q_bias = __ldg(&qkv_bias[thread_offset]);
  for (int seq_id = warpId; seq_id < block_seq_len; seq_id += warpNums)
  {
    int pos = (batch_seq_block_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hadd2(__ldg(&qkv[pos]), q_bias);
  }

  // loading Key
  half2 k_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim]);
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums)
  {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_kv + offset) = __hadd2(__ldg(&qkv[pos + half_hidden_dim]), k_bias);
  }
  __syncthreads();

  if (warpId < from_size * to_size)
  {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QK_mat;
    // wmma::fill_fragment(QK_mat, 0.0f);
    wmma::fill_fragment(QK_mat, __float2half(0.0f)); // rmf
    const int warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

#pragma unroll
    for (int k = 0; k < 4; k++)
    {
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();

  // softmax
  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums)
  {
    float max_val = -1e20f;

    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

#pragma unroll
    for (int i = 0; i < n; i++)
    {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = -1e20f;

      if (to_id[i] < seq_len)
      {
        float mask =
            // (float)__ldg(&attention_mask[(batch_seq_block_offset + from_id) * seq_len + to_id[i]]);
            __half2float(__ldg((__half *)&attention_mask[(batch_seq_block_offset + from_id) * seq_len + to_id[i]]));

        mask = (1.0f - mask) * (-10000.0f);
        // logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + mask; rmf
        logits[i] = __half2float(s_logits[from_id][to_id[i]]) * scale + mask;
      }
      max_val = max(max_val, logits[i]);
    }
    max_val = warpReduceMax(max_val);

    float sum_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n; i++)
    {
      logits[i] = __expf(logits[i] - max_val);
      sum_val += logits[i];
    }
    sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < max_seq_len)
        // s_logits[from_id][to_id[i]] = (__half)__fdividef(logits[i], sum_val); rmf
        s_logits[from_id][to_id[i]] = __float2half(__fdividef(logits[i], sum_val)); // rmf
  }

  // loading Value
  half2 v_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim * 2]);
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums)
  {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    ((__half2 *)(s_kv[seq_id]))[warp_tid] =
        __hadd2(__ldg(&qkv[pos + half_hidden_dim * 2]), v_bias);
  }

  // K dim clear 0
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  __syncthreads();

  //* V
  if (warpId < (from_size << 2))
  {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Logits_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QKV_mat;
    // wmma::fill_fragment(QKV_mat, 0.0f); rmf
    wmma::fill_fragment(QKV_mat, __float2half(0.0f)); // rmf
    const int warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < to_size; k++)
    {
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();

  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums)
  {
    int pos = (batch_seq_block_offset + from_id) * half_hidden_dim + thread_offset;
    ((__half2 *)(attention_output))[pos] = ((__half2 *)(s_query[from_id]))[warp_tid];
  }
}

void launcher_bytetr_attn(__half *qkv, __half *qkv_bias, __half *mask, __half *attention_output,
                        const int batch_size, const int head_num, const int seq_len,const int head_size,
                         cudaStream_t stream)
{
  const half2 *qkv_ptr = (const half2 *)qkv;
  const half2 *qkv_bias_ptr = (const half2 *)qkv_bias;
  float scale = (1.0f / sqrt(head_size * 1.0f));
  dim3 grid, block;
  int shared_memory_size = 0;
  const int split_count = (seq_len + 15) / 16;
  switch (split_count)
  {
  case 8:
  {
    constexpr int SEQ_LEN = 128; 
    constexpr int SPLIT_LEN = 64;
    constexpr int SIZE_PER_HEAD = 64;
    grid.x = head_num;
    grid.y = (SEQ_LEN + SPLIT_LEN - 1) / SPLIT_LEN;
    grid.z = batch_size;
    block.x = 32 * (SPLIT_LEN / 16 * split_count);
    shared_memory_size = ((SEQ_LEN + SPLIT_LEN) * (SIZE_PER_HEAD + SKEW_HALF) + SPLIT_LEN * (SEQ_LEN + SKEW_HALF)) * 2;
    bytetr_attn_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>
        <<<grid, block, shared_memory_size, stream>>>(qkv_ptr, qkv_bias_ptr, (const __half *)mask, attention_output, seq_len, scale);
    break;
  }
  case 16:
  {
    constexpr int SEQ_LEN = 256;
    constexpr int SPLIT_LEN = 32;
    constexpr int SIZE_PER_HEAD = 64;
    grid.x = head_num;
    grid.y = (SEQ_LEN + SPLIT_LEN - 1) / SPLIT_LEN;
    grid.z = batch_size;
    block.x = 32 * (SPLIT_LEN / 16 * split_count);
    shared_memory_size = ((SEQ_LEN + SPLIT_LEN) * (SIZE_PER_HEAD + SKEW_HALF) + SPLIT_LEN * (SEQ_LEN + SKEW_HALF)) * 2;
    if (shared_memory_size > 48 * 1024)                                                             \
    cudaFuncSetAttribute(bytetr_attn_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>,           \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024); 
    bytetr_attn_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>
        <<<grid, block, shared_memory_size, stream>>>(qkv_ptr, qkv_bias_ptr, (const __half *)mask, attention_output, seq_len, scale);
    break;
  }
  default:
  {
    constexpr int SEQ_LEN = 256;
    constexpr int SPLIT_LEN = 16;
    constexpr int SIZE_PER_HEAD = 64;
    grid.x = head_num;
    grid.y = (SEQ_LEN + SPLIT_LEN - 1) / SPLIT_LEN;
    grid.z = batch_size;
    block.x = 32 * (SPLIT_LEN / 16 * split_count);
    shared_memory_size = ((SEQ_LEN + SPLIT_LEN) * (SIZE_PER_HEAD + SKEW_HALF) + SPLIT_LEN * (SEQ_LEN + SKEW_HALF)) * 2;
    if (shared_memory_size > 48 * 1024)                                                             \
    cudaFuncSetAttribute(bytetr_attn_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>,           \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024); 
    bytetr_attn_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>
        <<<grid, block, shared_memory_size, stream>>>(qkv_ptr, qkv_bias_ptr, (const __half *)mask, attention_output, seq_len, scale);
    break;
  }
  }
}