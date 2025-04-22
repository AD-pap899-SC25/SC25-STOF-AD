// Q(B, H, S, W) @ K^T(B, H, W, S) -> mask -> softmax-> (B, H, S, S)
//   (B, H, S, S) @ V(B, H, S, W) -> (B, H, S, W)
//
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <stdio.h>
using namespace nvcuda;

#define SKEW_HALF 16

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#define WMMA_N_PARTITION_WIDTH 4
#define WMMA_M_PARTITION_HEIGHT 8

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__inline__ __device__ float warpReduceMax(float val)
{
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1){
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask, WARP_SIZE));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) 
{
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask, WARP_SIZE);
    }
    return val;
}

__device__ void softmax_n32_warplevel(__half *score, float *last_rowmax, float *last_rowsum, float *global_rowmax, float *global_rowsum, const int lane_width, const float softmax_scale)
{
    const int laneId = threadIdx.x % WARP_SIZE;
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float score_tmp = __half2float(score[laneId]) * softmax_scale;

    local_max = warpReduceMax(score_tmp);
    local_sum = warpReduceSum(__expf(score_tmp - local_max));

    // mi(j) = max(mi(j - 1), rowmax(Si(j)))
    float global_max = fmaxf(*global_rowmax, local_max); 

    // Pi(j) = expf(Si(j) - mi(j))
    score[laneId] = __float2half(__expf(score_tmp - global_max));

    // li(j) = li(j-1)expf(mi(j-1) - mi(j)) + rowsum(Pi(j))
    float global_sum = *global_rowsum * __expf(*global_rowmax - global_max) + local_sum * __expf(local_max - global_max); 
    
    *last_rowmax = *global_rowmax;
    *last_rowsum = *global_rowsum;
    *global_rowmax = global_max;
    *global_rowsum = global_sum;
}

__device__ void softmax_n64_warplevel(__half *score, float *last_rowmax, float *last_rowsum, float *global_rowmax, float *global_rowsum, const int lane_width, const float softmax_scale)
{
    const int laneId = threadIdx.x % WARP_SIZE;
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    half2 *score_vec = reinterpret_cast<half2*>(score);
    float2 f_val = __half22float2(score_vec[laneId]);
    float score_tmp0 = f_val.x * softmax_scale;
    float score_tmp1 = f_val.y * softmax_scale;

    local_max = warpReduceMax(fmaxf(score_tmp0, score_tmp1));
    float exp_val0 = __expf(score_tmp0 - local_max);
    float exp_val1 = __expf(score_tmp1 - local_max);
    local_sum = warpReduceSum(exp_val0 + exp_val1);

    // mi(j) = max(mi(j - 1), rowmax(Si(j)))
    float global_max = fmaxf(*global_rowmax, local_max); 

    // Pi(j) = expf(Si(j) - mi(j))
    float delta_max = global_max - local_max;
    float exp_delta = __expf(delta_max);
    score_vec[laneId] = __floats2half2_rn(exp_val0 * exp_delta, exp_val1 * exp_delta);

    // li(j) = li(j-1)expf(mi(j-1) - mi(j)) + rowsum(Pi(j))
    float global_sum = *global_rowsum * __expf(*global_rowmax - global_max) + local_sum * __expf(local_max - global_max); 
    
    *last_rowmax = *global_rowmax;
    *last_rowsum = *global_rowsum;
    *global_rowmax = global_max;
    *global_rowsum = global_sum;
}

__device__ void softmax_n16(__half *score, float *last_rowmax, float *last_rowsum, float *global_rowmax, float *global_rowsum, int lane_width, const float softmax_scale)
{
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float score_tmp[16];     
    half2 *score_vec = reinterpret_cast<half2*>(score);

    for (int i = 0; i < lane_width / 2; ++i)
    {
        half2 half2_score = score_vec[i];
        score_tmp[i*2] = __half2float(half2_score.x) * softmax_scale;
        score_tmp[i*2 + 1] = __half2float(half2_score.y) * softmax_scale;
        
        float local_max_last = local_max;
        local_max = fmaxf(local_max, score_tmp[i*2]);
        local_sum = local_sum * __expf(local_max_last - local_max) + __expf(score_tmp[i*2] - local_max);

        local_max_last = local_max;
        local_max = fmaxf(local_max, score_tmp[i*2 + 1]);
        local_sum = local_sum * __expf(local_max_last - local_max) + __expf(score_tmp[i*2 + 1] - local_max);       
    }

    // mi(j) = max(mi(j - 1), rowmax(Si(j)))
    float global_max = fmaxf(*global_rowmax, local_max);

    // Pi(j) = expf(Si(j) - mi(j))
    # pragma unroll
    for (int i = 0; i < lane_width / 2; ++i)
    {
        score_vec[i] = __floats2half2_rn(__expf(score_tmp[i*2] - global_max), __expf(score_tmp[i*2 + 1] - global_max));
    }

    // li(j) = li(j-1)expf(mi(j-1) - mi(j)) + rowsum(Pi(j))
    float global_sum = *global_rowsum * __expf(*global_rowmax - global_max) + local_sum * __expf(local_max - global_max); 
    
    *last_rowmax = *global_rowmax;
    *last_rowsum = *global_rowsum;
    *global_rowmax = global_max;
    *global_rowsum = global_sum;
}

__device__ void softmax_n32(__half *score, float *last_rowmax, float *last_rowsum, float *global_rowmax, float *global_rowsum, int lane_width, const float softmax_scale)
{
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float score_tmp[32];
    half2 *score_vec = reinterpret_cast<half2*>(score);

    for (int i = 0; i < lane_width / 2; ++i)
    {
        half2 half2_score = score_vec[i];
        score_tmp[i*2] = __half2float(half2_score.x) * softmax_scale;
        score_tmp[i*2 + 1] = __half2float(half2_score.y) * softmax_scale;
        
        float local_max_last = local_max;
        local_max = fmaxf(local_max, score_tmp[i*2]);
        local_sum = local_sum * __expf(local_max_last - local_max) + __expf(score_tmp[i*2] - local_max);

        local_max_last = local_max;
        local_max = fmaxf(local_max, score_tmp[i*2 + 1]);
        local_sum = local_sum * __expf(local_max_last - local_max) + __expf(score_tmp[i*2 + 1] - local_max);       
    }

    // mi(j) = max(mi(j - 1), rowmax(Si(j)))
    float global_max = fmaxf(*global_rowmax, local_max);

    // Pi(j) = expf(Si(j) - mi(j))
    # pragma unroll
    for (int i = 0; i < lane_width / 2; ++i)
    {
        score_vec[i] = __floats2half2_rn(__expf(score_tmp[i*2] - global_max), __expf(score_tmp[i*2 + 1] - global_max));
    }

    // li(j) = li(j-1)expf(mi(j-1) - mi(j)) + rowsum(Pi(j))
    float global_sum = *global_rowsum * __expf(*global_rowmax - global_max) + local_sum * __expf(local_max - global_max); 
    
    *last_rowmax = *global_rowmax;
    *last_rowsum = *global_rowsum;
    *global_rowmax = global_max;
    *global_rowsum = global_sum;
}

__device__ void softmax_n64(__half *score, float *last_rowmax, float *last_rowsum, float *global_rowmax, float *global_rowsum, int lane_width, const float softmax_scale)
{
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float score_tmp[64];
    half2 *score_vec = reinterpret_cast<half2*>(score);

    for (int i = 0; i < lane_width / 2; ++i)
    {
        half2 half2_score = score_vec[i];
        score_tmp[i*2] = __half2float(half2_score.x) * softmax_scale;
        score_tmp[i*2 + 1] = __half2float(half2_score.y) * softmax_scale;
        
        float local_max_last = local_max;
        local_max = fmaxf(local_max, score_tmp[i*2]);
        local_sum = local_sum * __expf(local_max_last - local_max) + __expf(score_tmp[i*2] - local_max);

        local_max_last = local_max;
        local_max = fmaxf(local_max, score_tmp[i*2 + 1]);
        local_sum = local_sum * __expf(local_max_last - local_max) + __expf(score_tmp[i*2 + 1] - local_max);       
    }

    // mi(j) = max(mi(j - 1), rowmax(Si(j)))
    float global_max = fmaxf(*global_rowmax, local_max);

    // Pi(j) = expf(Si(j) - mi(j))
    # pragma unroll
    for (int i = 0; i < lane_width / 2; ++i)
    {
        score_vec[i] = __floats2half2_rn(__expf(score_tmp[i*2] - global_max), __expf(score_tmp[i*2 + 1] - global_max));
    }

    // li(j) = li(j-1)expf(mi(j-1) - mi(j)) + rowsum(Pi(j))
    float global_sum = *global_rowsum * __expf(*global_rowmax - global_max) + local_sum * __expf(local_max - global_max); 
    
    *last_rowmax = *global_rowmax;
    *last_rowsum = *global_rowsum;
    *global_rowmax = global_max;
    *global_rowsum = global_sum;
}

__device__ void softmax_n128(__half *score, float *last_rowmax, float *last_rowsum, float *global_rowmax, float *global_rowsum, int lane_width, const float softmax_scale)
{
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float score_tmp[64];
    half2 *score_vec = reinterpret_cast<half2*>(score);

    for (int i = 0; i < lane_width / 2; ++i)
    {
        half2 half2_score = score_vec[i];
        score_tmp[i*2] = __half2float(half2_score.x) * softmax_scale;
        score_tmp[i*2 + 1] = __half2float(half2_score.y) * softmax_scale;
        
        float local_max_last = local_max;
        local_max = fmaxf(local_max, score_tmp[i*2]);
        local_sum = local_sum * __expf(local_max_last - local_max) + __expf(score_tmp[i*2] - local_max);

        local_max_last = local_max;
        local_max = fmaxf(local_max, score_tmp[i*2 + 1]);
        local_sum = local_sum * __expf(local_max_last - local_max) + __expf(score_tmp[i*2 + 1] - local_max);       
    }

    // mi(j) = max(mi(j - 1), rowmax(Si(j)))
    float global_max = fmaxf(*global_rowmax, local_max);

    // Pi(j) = expf(Si(j) - mi(j))
    # pragma unroll
    for (int i = 0; i < lane_width / 2; ++i)
    {
        score_vec[i] = __floats2half2_rn(__expf(score_tmp[i*2] - global_max), __expf(score_tmp[i*2 + 1] - global_max));
    }

    // li(j) = li(j-1)expf(mi(j-1) - mi(j)) + rowsum(Pi(j))
    float global_sum = *global_rowsum * __expf(*global_rowmax - global_max) + local_sum * __expf(local_max - global_max); 
    
    *last_rowmax = *global_rowmax;
    *last_rowsum = *global_rowsum;
    *global_rowmax = global_max;
    *global_rowsum = global_sum;
}


// -------------------------- warpNum = 1 ------------------------------
// ---------------------------------------------------------------------
// warp1 n16
__global__ void block_attn_mask_1warp_n16_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    
    int num_tid_for_M;
    int per_tid_iter_for_M;
    if(BLOCK_M <= WARP_SIZE){
        num_tid_for_M = WARP_SIZE / BLOCK_M;
    }
    else{
        per_tid_iter_for_M = BLOCK_M / WARP_SIZE;
    }

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / WARP_SIZE;
    const int per_thread_deal_kv = size_kv_shared / WARP_SIZE;
    const int per_thread_deal_acc = size_acc_shared / WARP_SIZE;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 40; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + laneId; // 40 =  (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 32; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 40;
            int shared_ptr = kv_offset_shared_half2_ptr + laneId;

            int kv_offset_global_half2_ptr = i * 32;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WMMA_N)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M;
                int bCol = tile_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 40;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 32;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();


        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;   
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N) / 2;

            int part_block_offset = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N);
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 64;  // 2 * num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/8) * 16 + tid % 8;

                int mask_global_offset = i * 32; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        if(BLOCK_M <= WARP_SIZE)
        {
            int BLOCK_M_row_idx = tid / num_tid_for_M; // tid/2
            if (tid % num_tid_for_M == 0)
            {
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n16(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        else
        {
            for(int i = 0; i < per_tid_iter_for_M; i++)
            {
                int BLOCK_M_row_idx = tid * per_tid_iter_for_M + i;
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n16(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WMMA_N)
            {
                int aRow = tile_M;
                int bCol = tile_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 40; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 32; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp1 n32
__global__ void block_attn_mask_1warp_n32_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    
    int num_tid_for_M;
    int per_tid_iter_for_M;
    if(BLOCK_M <= WARP_SIZE){
        num_tid_for_M = WARP_SIZE / BLOCK_M;
    }
    else{
        per_tid_iter_for_M = BLOCK_M / WARP_SIZE;
    }

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / WARP_SIZE;
    const int per_thread_deal_kv = size_kv_shared / WARP_SIZE;
    const int per_thread_deal_acc = size_acc_shared / WARP_SIZE;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 40; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + laneId; // 40 =  (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 32; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 40;
            int shared_ptr = kv_offset_shared_half2_ptr + laneId;

            int kv_offset_global_half2_ptr = i * 32;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WMMA_N)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M;
                int bCol = tile_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 40;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 32;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();


        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;   
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N) / 2;

            int part_block_offset = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N);
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 48;  // 2 * num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/16) * 24 + tid % 16;

                int mask_global_offset = i * 32; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        if(BLOCK_M <= WARP_SIZE)
        {
            int BLOCK_M_row_idx = tid / num_tid_for_M; // tid/2
            if (tid % num_tid_for_M == 0)
            {
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n32(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        else
        {
            for(int i = 0; i < per_tid_iter_for_M; i++)
            {
                int BLOCK_M_row_idx = tid * per_tid_iter_for_M + i;
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n32(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WMMA_N)
            {
                int aRow = tile_M;
                int bCol = tile_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 40; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 32; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp1 n64
__global__ void block_attn_mask_1warp_n64_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;

    int num_tid_for_M;
    int per_tid_iter_for_M;
    if(BLOCK_M <= WARP_SIZE){
        num_tid_for_M = WARP_SIZE / BLOCK_M;
    }
    else{
        per_tid_iter_for_M = BLOCK_M / WARP_SIZE;
    }

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / WARP_SIZE;
    const int per_thread_deal_kv = size_kv_shared / WARP_SIZE;
    const int per_thread_deal_acc = size_acc_shared / WARP_SIZE;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 40; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + tid; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 32; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 40;
            int shared_ptr = kv_offset_shared_half2_ptr + tid;

            int kv_offset_global_half2_ptr = i * 32;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WMMA_N)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M;
                int bCol = tile_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 40;
            int shared_ptr = kv_offset_shared_half2_ptr + tid;

            int kv_offset_global_half2_ptr = i * 32;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 40;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + tid; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 32; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        if(BLOCK_M <= WARP_SIZE)
        {
            int BLOCK_M_row_idx = tid / num_tid_for_M; // tid/2
            if (tid % num_tid_for_M == 0)
            {
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n64(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        else
        {
            for(int i = 0; i < per_tid_iter_for_M; i++)
            {
                int BLOCK_M_row_idx = tid * per_tid_iter_for_M + i;
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n64(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WMMA_N)
            {
                int aRow = tile_M;
                int bCol = tile_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + tid / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + tid / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 40; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + tid;

        int q_offset_global_half2_ptr = i * 32; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp1 n128
__global__ void block_attn_mask_1warp_n128_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;

    int num_tid_for_M;
    int per_tid_iter_for_M;
    if(BLOCK_M <= WARP_SIZE){
        num_tid_for_M = WARP_SIZE / BLOCK_M;
    }
    else{
        per_tid_iter_for_M = BLOCK_M / WARP_SIZE;
    }

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / WARP_SIZE;
    const int per_thread_deal_kv = size_kv_shared / WARP_SIZE;
    const int per_thread_deal_acc = size_acc_shared / WARP_SIZE;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 40; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + tid; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 32; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();


    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 40;
            int shared_ptr = kv_offset_shared_half2_ptr + tid;

            int kv_offset_global_half2_ptr = i * 32;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WMMA_N)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M;
                int bCol = tile_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 40;
            int shared_ptr = kv_offset_shared_half2_ptr + tid;

            int kv_offset_global_half2_ptr = i * 32;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i/2 * 72;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + i%2 * 32 + tid; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 32; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        if(BLOCK_M <= WARP_SIZE)
        {
            int BLOCK_M_row_idx = tid / num_tid_for_M; // tid/2
            if (tid % num_tid_for_M == 0)
            {
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n128(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        else
        {
            for(int i = 0; i < per_tid_iter_for_M; i++)
            {
                int BLOCK_M_row_idx = tid * per_tid_iter_for_M + i;
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n128(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WMMA_N)
            {
                int aRow = tile_M;
                int bCol = tile_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + tid / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + tid / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 40; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + tid;

        int q_offset_global_half2_ptr = i * 32; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}


// -------------------------- warpNum = 2 ------------------------------
// ---------------------------------------------------------------------
// warp2 m32... n16 
__global__ void block_attn_mask_2warp_n16_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    
    int num_tid_for_M;
    int per_tid_iter_for_M;
    if(BLOCK_M <= WARP_SIZE){
        num_tid_for_M = 64 / BLOCK_M;
    }
    else{
        per_tid_iter_for_M = BLOCK_M / 64;
    }

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 64;
    const int per_thread_deal_kv = size_kv_shared / 64;
    const int per_thread_deal_acc = size_acc_shared / 64;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 80; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr  + warpId * 40 + laneId; // 40 =  (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 64; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 80;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 64;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WMMA_N)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M + warpId * WMMA_M;;
                int bCol = tile_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 80;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 64;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();


        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;   
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N) / 2;

            int part_block_offset = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N);
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 128;  // 2 * num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/8) * 16 + tid % 8;

                int mask_global_offset = i * 64; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        if(BLOCK_M <= WARP_SIZE)
        {
            int BLOCK_M_row_idx = tid / num_tid_for_M; // tid/2
            if (tid % num_tid_for_M == 0)
            {
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n16(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        else
        {
            for(int i = 0; i < per_tid_iter_for_M; i++)
            {
                int BLOCK_M_row_idx = tid * per_tid_iter_for_M + i;
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n16(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WMMA_N)
            {
                int aRow = tile_M + warpId * WMMA_M;
                int bCol = tile_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 80; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 64; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp2 m32n32 m64n32
__global__ void block_attn_mask_2warp_n32_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    
    int num_tid_for_M;
    int per_tid_iter_for_M;
    if(BLOCK_M <= WARP_SIZE){
        num_tid_for_M = 64 / BLOCK_M;
    }
    else{
        per_tid_iter_for_M = BLOCK_M / 64;
    }

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 64;
    const int per_thread_deal_kv = size_kv_shared / 64;
    const int per_thread_deal_acc = size_acc_shared / 64;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 80; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 =  (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 64; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 80;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 64;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WARP_SIZE)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M;
                int bCol = tile_N + warpId * WMMA_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 80;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 64;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();


        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;   
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N) / 2;

            int part_block_offset = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N);
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 96;  // 2 * num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/16) * 24 + tid % 16;

                int mask_global_offset = i * 64; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        if(BLOCK_M <= WARP_SIZE)
        {
            int BLOCK_M_row_idx = tid / num_tid_for_M; // tid/2
            if (tid % num_tid_for_M == 0)
            {
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n32(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        else
        {
            for(int i = 0; i < per_tid_iter_for_M; i++)
            {
                int BLOCK_M_row_idx = tid * per_tid_iter_for_M + i;
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n32(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N +=WARP_SIZE)
            {
                int aRow = tile_M;
                int bCol = tile_N + warpId * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 80; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 64; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp2 n64
__global__ void block_attn_mask_2warp_n64_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    
    int num_tid_for_M;
    int per_tid_iter_for_M;
    if(BLOCK_M <= WARP_SIZE){
        num_tid_for_M = 64 / BLOCK_M;
    }
    else{
        per_tid_iter_for_M = BLOCK_M / 64;
    }

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 64;
    const int per_thread_deal_kv = size_kv_shared / 64;
    const int per_thread_deal_acc = size_acc_shared / 64;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 80; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 =  (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 64; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 80;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 64;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WARP_SIZE)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M;
                int bCol = tile_N + warpId * WMMA_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 80;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 64;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();


        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;   
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N) / 2;

            int part_block_offset = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N);
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 80;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/32) * 40 + tid % 32; 

                int mask_global_offset = i * 64; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        if(BLOCK_M <= WARP_SIZE)
        {
            int BLOCK_M_row_idx = tid / num_tid_for_M; // tid/2
            if (tid % num_tid_for_M == 0)
            {
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n64(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        else
        {
            for(int i = 0; i < per_tid_iter_for_M; i++)
            {
                int BLOCK_M_row_idx = tid * per_tid_iter_for_M + i;
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n64(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WARP_SIZE)
            {
                int aRow = tile_M;
                int bCol = tile_N + warpId * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 80; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 64; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp2 n128
__global__ void block_attn_mask_2warp_n128_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;

    int num_tid_for_M = 64 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 64;
    const int per_thread_deal_kv = size_kv_shared / 64;
    const int per_thread_deal_acc = size_acc_shared / 64;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 80; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 64; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();


    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 80;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 64;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WARP_SIZE)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M;
                int bCol = tile_N + warpId * WMMA_N;

                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 80;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 64;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 72;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + tid; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 64; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------

        int BLOCK_M_row_idx = tid / num_tid_for_M; // tid/2
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n128(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WARP_SIZE)
            {
                int aRow = tile_M;
                int bCol = tile_N + warpId * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + tid / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + tid / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 80; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 64; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}



// -------------------------- warpNum = 4 ------------------------------
// ---------------------------------------------------------------------
// warp4 m64n16
__global__ void block_attn_mask_4warp_n16_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpM = warpId;
    const int num_tid_for_M = 128 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 128;
    const int per_thread_deal_kv = size_kv_shared / 128;
    const int per_thread_deal_acc = size_acc_shared / 128;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 =  (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 128; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 160;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 128;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF; // 转置后的二者的公共维度
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += (WARP_SIZE*2))
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WMMA_N)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 160;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 128;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();


        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;   
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N) / 2;

            int part_block_offset = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N);
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 256;  // num_warps * (BLOCK_N + SKEW_HALF)
                int acc_shared_ptr = acc_shared_offset + (tid/8) * 16 + tid % 8;

                int mask_global_offset = i * 128; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        int BLOCK_M_row_idx = tid / num_tid_for_M; // tid/2
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n16(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += (WARP_SIZE*2))
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WMMA_N)
            {
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 128; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp4 m64n32  m32n32
__global__ void block_attn_mask_4warp_n32_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpM = warpId / 2;
    const int warpN = warpId % 2;
    const int num_tid_for_M = 128 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 128;
    const int per_thread_deal_kv = size_kv_shared / 128;
    const int per_thread_deal_acc = size_acc_shared / 128;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 =  (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 128; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 160;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 128;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF; // 转置后的二者的公共维度
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WARP_SIZE)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 160;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 128;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();


        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;   
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N) / 2;

            int part_block_offset = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N);
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 192;  // 2 * num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/16) * 24 + tid % 16;

                int mask_global_offset = i * 128; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        int BLOCK_M_row_idx = tid / num_tid_for_M; // tid/2
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n32(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        #pragma unroll
        for(int warps_for_softmax = 0; warps_for_softmax < BLOCK_M; warps_for_softmax += num_warps)
        {
            int BLOCK_M_row_idx = warps_for_softmax + warpId;
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }
            softmax_n32_warplevel(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WARP_SIZE)
            {
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 128; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp4 m16n64
__global__ void block_attn_mask_4warp_m16n64_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
    {
        const int batch_idx = blockIdx.y;
        const int head_idx = blockIdx.z;
        const int BLOCK_M_idx = blockIdx.x;
    
        const int tid = threadIdx.x;
        const int warpId = tid / WARP_SIZE;
        const int laneId = tid % WARP_SIZE;

        const int warpN = warpId;
        const int num_tid_for_M = 128 / BLOCK_M;
    
        const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
        const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;
    
        const int size_q_shared = BLOCK_M * head_size; 
        const int size_kv_shared = BLOCK_N * head_size;
        const int size_acc_shared = BLOCK_M * BLOCK_N;
    
        const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
        const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
        const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);
    
        const int per_thread_deal_q = size_q_shared / 128;
        const int per_thread_deal_kv = size_kv_shared / 128;
        const int per_thread_deal_acc = size_acc_shared / 128;
        const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
        const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
        const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;
    
        extern __shared__ __half shared_mem[];
        __half *q_shared = shared_mem;
        __half *kv_shared = q_shared + skew_size_q_half;
        __half *acc_shared = kv_shared + skew_size_kv_half;
        __half *res_shared = acc_shared + skew_size_acc_half;
        
        float *last_rowmax = (float *)(res_shared + skew_size_q_half);
        float *last_rowsum = last_rowmax + BLOCK_M;
        float *global_rowmax = last_rowsum + BLOCK_M;
        float *global_rowsum = global_rowmax + BLOCK_M;
    
        half2 *q_vec = reinterpret_cast<half2*>(q);
        half2 *k_vec = reinterpret_cast<half2*>(k);
        half2 *v_vec = reinterpret_cast<half2*>(v);
        half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
        half2 *res_vec = reinterpret_cast<half2*>(res_shared);
        half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);
    
        const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
        const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
        int full_num_id = 0; 
        int part_num_id = 0;
    
        #pragma unroll
        for(int i = 0; i < per_thread_deal_q_half2; i++) {
            int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
            int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 = (head_size + SKEW_HALF) / 2
    
            int q_offset_global_half2_ptr = i * 128; // num_warps * WARPSIZE
            int global_ptr = q_offset_global_half2_ptr + tid;
    
            half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));
    
            half2 half2_val = __float2half2_rn(0.0f);
            res_vec[shared_ptr] = half2_val;
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        
    
        for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){
    
            int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
            int offset_BLOCK_N = load_block_col_idx * BLOCK_N;
    
            #pragma unroll
            for (int i = 0; i < per_thread_deal_kv_half2; i++) {
                int kv_offset_shared_half2_ptr = i * 160;
                int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;
    
                int kv_offset_global_half2_ptr = i * 128;
                int global_ptr = kv_offset_global_half2_ptr + tid;
                
                half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
                __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
            }
            __pipeline_commit();
            __pipeline_wait_prior(0);
            __syncthreads();
    
    
            // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
            // ---------------------------------------------------------------------------
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
            int lda = head_size + SKEW_HALF;
            int ldb = head_size + SKEW_HALF;
            int ldc = BLOCK_N + SKEW_HALF;
    
            #pragma unroll
            for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
            {
                for (int tile_N = 0; tile_N < BLOCK_N; tile_N += (WARP_SIZE *2))
                {        
                    wmma::fill_fragment(frag_acc, __float2half(0.0f));
                    int aRow = tile_M;
                    int bCol = tile_N + warpN * WMMA_N;

                    for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                    {
                        int aCol = k_idx;
                        int bRow = k_idx;
    
                        wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                        wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                    }
                    wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
                }
            }
            __syncthreads();
    
            #pragma unroll
            for (int i = 0; i < per_thread_deal_kv_half2; i++) {
                int kv_offset_shared_half2_ptr = i * 160;
                int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;
    
                int kv_offset_global_half2_ptr = i * 128;
                int global_ptr = kv_offset_global_half2_ptr + tid;
                
                half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
                __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
            }
            __pipeline_commit();
    
    
    
            // -------------------------------- Mask -------------------------------------
            // ---------------------------------------------------------------------------
            if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
            {
                full_num_id++;
            }
    
            else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
            {
                int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
                part_num_id++;
            
                #pragma unroll
                for (int i = 0; i < per_thread_deal_acc_half2; i++)
                {
                    int acc_shared_offset = i * 160;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                    int acc_shared_ptr = acc_shared_offset + (tid/32) * 40 + tid % 32; // 40 = (BLOCK_N + SKEW_HALF)/2
                    // 20 = (BLOCK_N + SKEW_HALF)/4
    
                    int mask_global_offset = i * 128; // num_warps * WARP_SIZE
                    int global_ptr = mask_global_offset + tid;
                    
                    acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
                }
                __syncthreads();
            }
    
            else continue;
            
    
            // -------------------------- P = SoftMax(S) ---------------------------------
            // ---------------------------------------------------------------------------
            int BLOCK_M_row_idx = tid / num_tid_for_M;
            if (tid % num_tid_for_M == 0)
            {
                if (load_num_id == 0)
                {
                    last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    last_rowsum[BLOCK_M_row_idx] = 0.0f;
                    global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                    global_rowsum[BLOCK_M_row_idx] = 0.0f;
                }
    
                softmax_n64(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            }
            __pipeline_wait_prior(0);
            __syncthreads();
    
            // #pragma unroll
            // for(int warps_for_softmax = 0; warps_for_softmax < BLOCK_M; warps_for_softmax += num_warps)
            // {
            //     int BLOCK_M_row_idx = warps_for_softmax + warpId;
            //     if (load_num_id == 0)
            //     {
            //         last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
            //         last_rowsum[BLOCK_M_row_idx] = 0.0f;
            //         global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
            //         global_rowsum[BLOCK_M_row_idx] = 0.0f;
            //     }
            //     softmax_n64_warplevel(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
            // }
            // __pipeline_wait_prior(0);
            // __syncthreads();
    
    
            // ---------------------------- Res = P * V ----------------------------------
            // ---------------------------------------------------------------------------
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
            lda = BLOCK_N + SKEW_HALF;
            ldb = head_size + SKEW_HALF;
            ldc = head_size + SKEW_HALF;
    
            #pragma unroll
            for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
            {
                for (int tile_N = 0; tile_N < head_size; tile_N += (WARP_SIZE*2))
                {
                    int aRow = tile_M;
                    int bCol = tile_N + warpN * WMMA_N;
                    wmma::fill_fragment(frag_res, __float2half(0.0f));
    
                    for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                    {
                        int aCol = k_idx;
                        int bRow = k_idx;
    
                        wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                        wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                        wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                    }
    
                    if(load_num_id > 0)
                    {
                        wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);
    
                        int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                        float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);
    
                        frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                        frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));
    
                        frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                        frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                        
                        update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                        alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);
    
                        frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                        frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));
    
                        frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                        frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                    }
                    
                    wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
                }
            }
            __syncthreads();
    
        }
    
        // ---------------------------- Write Result ----------------------------------
        // ----------------------------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < per_thread_deal_q_half2; i++) 
        {   
            int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
            int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;
    
            int q_offset_global_half2_ptr = i * 128; // num_warps * WARP_SIZE
            int global_ptr = q_offset_global_half2_ptr + tid;
    
            float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 
    
            half2 half2_val = res_vec[shared_ptr];
            half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
            half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));
    
            q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
        }   
    }

// warp4 m32n64 m64n64
__global__ void block_attn_mask_4warp_n64_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpM = warpId / 2;
    const int warpN = warpId % 2;
    const int num_tid_for_M = 128 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 128;
    const int per_thread_deal_kv = size_kv_shared / 128;
    const int per_thread_deal_acc = size_acc_shared / 128;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 128; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 160;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 128;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WARP_SIZE)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 160;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 128;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 160;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/32) * 40 + tid % 32; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 128; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        int BLOCK_M_row_idx = tid / num_tid_for_M;
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n64(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();

        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WARP_SIZE)
            {
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 128; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp4 m16n128
__global__ void block_attn_mask_4warp_m16n128_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpN = warpId;
    const int num_tid_for_M = 128 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 128;
    const int per_thread_deal_kv = size_kv_shared / 128;
    const int per_thread_deal_acc = size_acc_shared / 128;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 128; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 160;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 128;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += (WARP_SIZE * 2))
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M;
                int bCol = tile_N + warpN * WMMA_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 160;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 128;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 160;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/32) * 40 + tid % 32; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 128; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        int BLOCK_M_row_idx = tid / num_tid_for_M;
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n128(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += (WARP_SIZE*2))
            {
                int aRow = tile_M;
                int bCol = tile_N + warpN * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 128; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp4 m32n128
__global__ void block_attn_mask_4warp_m32n128_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpM = warpId / 2; // 0,1
    const int warpN = warpId % 2; // 0,1
    const int num_tid_for_M = 128 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 128;
    const int per_thread_deal_kv = size_kv_shared / 128;
    const int per_thread_deal_acc = size_acc_shared / 128;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 128; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 160;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 128;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WARP_SIZE)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;

                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 160;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 128;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 160;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/32) * 40 + tid % 32; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 128; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        int BLOCK_M_row_idx = tid / num_tid_for_M;
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n128(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WARP_SIZE)
            {
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 160; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 128; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}



// -------------------------- warpNum = 8 ------------------------------
// ---------------------------------------------------------------------
// warp8 m64n32
__global__ void block_attn_mask_8warp_n32_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpM = warpId % 4;// 0,1,2,3
    const int warpN = warpId / 4;// 0,1
    const int num_tid_for_M = 256 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 256;
    const int per_thread_deal_kv = size_kv_shared / 256;
    const int per_thread_deal_acc = size_acc_shared / 256;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 320; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 =  (head_size + SKEW_HALF) / 2
        
        int q_offset_global_half2_ptr = i * 256; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 320;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 256;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF; // 转置后的二者的公共维度
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += (WARP_SIZE * 2))
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += WARP_SIZE)
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 320;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 256;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();


        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;   
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N) / 2;

            int part_block_offset = (part_row_ptr[BLOCK_M_idx] + part_num_id) * (BLOCK_M * BLOCK_N);
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 384;  // 2 * num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/16) * 24 + tid % 16;

                int mask_global_offset = i * 256; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        #pragma unroll
        for(int warps_for_softmax = 0; warps_for_softmax < BLOCK_M; warps_for_softmax += num_warps)
        {
            int BLOCK_M_row_idx = warps_for_softmax + warpId;
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }
            softmax_n32_warplevel(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += (WARP_SIZE*2))
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += WARP_SIZE)
            {
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 320; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 256; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp8 m32n64 m64n64
__global__ void block_attn_mask_8warp_n64_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpM = warpId / 4;//0,1
    const int warpN = warpId % 4;//0,1,2,3
    const int num_tid_for_M = 256 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 256;
    const int per_thread_deal_kv = size_kv_shared / 256;
    const int per_thread_deal_acc = size_acc_shared / 256;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 320; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 256; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 320;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 256;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += (WARP_SIZE*2))
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 320;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 256;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 320;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/32) * 40 + tid % 32; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 256; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        int BLOCK_M_row_idx = tid / num_tid_for_M;
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n64(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += (WARP_SIZE * 2))
            {
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 320; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 256; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp8 m16n128
__global__ void block_attn_mask_8warp_m16n128_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpN = warpId;
    const int warpN2 = warpId % 4;
    const int num_tid_for_M = 256 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 256;
    const int per_thread_deal_kv = size_kv_shared / 256;
    const int per_thread_deal_acc = size_acc_shared / 256;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 320; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 256; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 320;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 256;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += (WARP_SIZE * 4))
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M;
                int bCol = tile_N + warpN * WMMA_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 320;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 256;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 320;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/32) * 40 + tid % 32; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 256; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        int BLOCK_M_row_idx = tid / num_tid_for_M;
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n128(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WMMA_M)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += (WARP_SIZE*2))
            {
                int aRow = tile_M;
                int bCol = tile_N + warpN2 * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 320; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 256; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp8 m32n128
__global__ void block_attn_mask_8warp_m32n128_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpM = warpId / 4; // 0,1
    const int warpN = warpId % 4; // 0,1,2,3
    const int num_tid_for_M = 256 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 256;
    const int per_thread_deal_kv = size_kv_shared / 256;
    const int per_thread_deal_acc = size_acc_shared / 256;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 320; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 256; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 320;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 256;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += (WARP_SIZE * 2))
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;

                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 320;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 256;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 320;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/32) * 40 + tid % 32; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 256; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        int BLOCK_M_row_idx = tid / num_tid_for_M;
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n128(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += (WARP_SIZE*2) )
            {
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 320; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 256; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}



// -------------------------- warpNum = 16 ------------------------------
// ---------------------------------------------------------------------
// warp16 m64n64
__global__ void block_attn_mask_16warp_n64_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpM = warpId / 4;//0,1,2,3
    const int warpN = warpId % 4;//0,1,2,3
    const int num_tid_for_M = 512 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 512;
    const int per_thread_deal_kv = size_kv_shared / 512;
    const int per_thread_deal_acc = size_acc_shared / 512;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 640; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 512; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 640;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 512;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += (WARP_SIZE*2))
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += (WARP_SIZE*2))
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 640;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 512;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 640;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/32) * 40 + tid % 32; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 512; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        int BLOCK_M_row_idx = tid / num_tid_for_M;
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n64(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += (WARP_SIZE*2))
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += (WARP_SIZE * 2))
            {
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 640; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 512; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}

// warp16 m32n128
__global__ void block_attn_mask_16warp_n128_kernel(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int head_size)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int BLOCK_M_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int warpM = warpId / 8;//0,1
    const int warpN = warpId % 8;//0,1,2,3,4,5,6,7
    const int warpN2 = warpId % 4; //0,1,2,3
    const int num_tid_for_M = 512 / BLOCK_M;

    const int offset_BLOCK_M = BLOCK_M_idx * BLOCK_M;
    const int offset_common = stride_0 * batch_idx + stride_1 * head_idx;

    const int size_q_shared = BLOCK_M * head_size; 
    const int size_kv_shared = BLOCK_N * head_size;
    const int size_acc_shared = BLOCK_M * BLOCK_N;

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    const int per_thread_deal_q = size_q_shared / 512;
    const int per_thread_deal_kv = size_kv_shared / 512;
    const int per_thread_deal_acc = size_acc_shared / 512;
    const int per_thread_deal_q_half2 = per_thread_deal_q >> 1;
    const int per_thread_deal_kv_half2 = per_thread_deal_kv >> 1;
    const int per_thread_deal_acc_half2 = per_thread_deal_acc >> 1;

    extern __shared__ __half shared_mem[];
    __half *q_shared = shared_mem;
    __half *kv_shared = q_shared + skew_size_q_half;
    __half *acc_shared = kv_shared + skew_size_kv_half;
    __half *res_shared = acc_shared + skew_size_acc_half;
    
    float *last_rowmax = (float *)(res_shared + skew_size_q_half);
    float *last_rowsum = last_rowmax + BLOCK_M;
    float *global_rowmax = last_rowsum + BLOCK_M;
    float *global_rowsum = global_rowmax + BLOCK_M;

    half2 *q_vec = reinterpret_cast<half2*>(q);
    half2 *k_vec = reinterpret_cast<half2*>(k);
    half2 *v_vec = reinterpret_cast<half2*>(v);
    half2 *acc_vec = reinterpret_cast<half2*>(acc_shared);
    half2 *res_vec = reinterpret_cast<half2*>(res_shared);
    half2 *part_block_mask_vec = reinterpret_cast<half2*>(part_block_mask);

    const int full_num = full_row_ptr[BLOCK_M_idx + 1] - full_row_ptr[BLOCK_M_idx];
    const int part_num = part_row_ptr[BLOCK_M_idx + 1] - part_row_ptr[BLOCK_M_idx];
    int full_num_id = 0; 
    int part_num_id = 0;

    #pragma unroll
    for(int i = 0; i < per_thread_deal_q_half2; i++) {
        int q_offset_shared_half2_ptr = i * 640; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId; // 40 = (head_size + SKEW_HALF) / 2

        int q_offset_global_half2_ptr = i * 512; // num_warps * WARPSIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        half2 *src_q = q_vec + (offset_common + offset_BLOCK_M * stride_2) / 2 + global_ptr;
        __pipeline_memcpy_async(&q_shared[shared_ptr*2], src_q, sizeof(half2));

        half2 half2_val = __float2half2_rn(0.0f);
        res_vec[shared_ptr] = half2_val;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    

    for(int load_num_id = 0; load_num_id < full_num + part_num; load_num_id++){

        int load_block_col_idx = load_col_idx[load_row_ptr[BLOCK_M_idx] + load_num_id];
        int offset_BLOCK_N = load_block_col_idx * BLOCK_N;

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 640;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 512;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_k = k_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_k, sizeof(half2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();


        // ----- S = QK^T  q_shared(S, W) @ k_shared_T(W, S) -> qk_score(S, S) -------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_k;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_acc;
        int lda = head_size + SKEW_HALF;
        int ldb = head_size + SKEW_HALF;
        int ldc = BLOCK_N + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < BLOCK_N; tile_N += (WARP_SIZE * 4))
            {        
                wmma::fill_fragment(frag_acc, __float2half(0.0f));
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN * WMMA_N;
                for (int k_idx = 0; k_idx < head_size; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_q, q_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_k, kv_shared + bCol * ldb + bRow, ldb);
                    wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                }
                wmma::store_matrix_sync(acc_shared + aRow * ldc + bCol, frag_acc, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < per_thread_deal_kv_half2; i++) {
            int kv_offset_shared_half2_ptr = i * 640;
            int shared_ptr = kv_offset_shared_half2_ptr + warpId * 40 + laneId;

            int kv_offset_global_half2_ptr = i * 512;
            int global_ptr = kv_offset_global_half2_ptr + tid;
            
            half2 *src_v = v_vec + (offset_common + offset_BLOCK_N * stride_2) / 2 + global_ptr;
            __pipeline_memcpy_async(&kv_shared[shared_ptr*2], src_v, sizeof(half2));
        }
        __pipeline_commit();



        // -------------------------------- Mask -------------------------------------
        // ---------------------------------------------------------------------------
        if(full_num_id < full_num && load_block_col_idx == full_col_idx[full_row_ptr[BLOCK_M_idx] + full_num_id])
        {
            full_num_id++;
        }

        else if(part_num_id < part_num && load_block_col_idx == part_col_idx[part_row_ptr[BLOCK_M_idx] + part_num_id])
        {
            int part_block_offset_half2 = (part_row_ptr[BLOCK_M_idx] + part_num_id) * size_acc_shared / 2;
            part_num_id++;
        
            #pragma unroll
            for (int i = 0; i < per_thread_deal_acc_half2; i++)
            {
                int acc_shared_offset = i * 640;  // num_warps * (BLOCK_N + SKEW_HALF) / 2
                int acc_shared_ptr = acc_shared_offset + (tid/32) * 40 + tid % 32; // 40 = (BLOCK_N + SKEW_HALF)/2
                // 20 = (BLOCK_N + SKEW_HALF)/4

                int mask_global_offset = i * 512; // num_warps * WARP_SIZE
                int global_ptr = mask_global_offset + tid;
                
                acc_vec[acc_shared_ptr] = __hadd2(acc_vec[acc_shared_ptr], part_block_mask_vec[part_block_offset_half2 + global_ptr]);
            }
            __syncthreads();
        }

        else continue;
        

        // -------------------------- P = SoftMax(S) ---------------------------------
        // ---------------------------------------------------------------------------
        int BLOCK_M_row_idx = tid / num_tid_for_M;
        if (tid % num_tid_for_M == 0)
        {
            if (load_num_id == 0)
            {
                last_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                last_rowsum[BLOCK_M_row_idx] = 0.0f;
                global_rowmax[BLOCK_M_row_idx] = -INFINITY; 
                global_rowsum[BLOCK_M_row_idx] = 0.0f;
            }

            softmax_n128(acc_shared + BLOCK_M_row_idx * (BLOCK_N + SKEW_HALF), &last_rowmax[BLOCK_M_row_idx], &last_rowsum[BLOCK_M_row_idx], &global_rowmax[BLOCK_M_row_idx], &global_rowsum[BLOCK_M_row_idx], BLOCK_N, softmax_scale);
        }
        __pipeline_wait_prior(0);
        __syncthreads();


        // ---------------------------- Res = P * V ----------------------------------
        // ---------------------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_s;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_v;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> frag_res_last;
        lda = BLOCK_N + SKEW_HALF;
        ldb = head_size + SKEW_HALF;
        ldc = head_size + SKEW_HALF;

        #pragma unroll
        for (int tile_M = 0; tile_M < BLOCK_M; tile_M += WARP_SIZE)
        {
            for (int tile_N = 0; tile_N < head_size; tile_N += (WARP_SIZE*2) )
            {
                int aRow = tile_M + warpM * WMMA_M;
                int bCol = tile_N + warpN2 * WMMA_N;
                wmma::fill_fragment(frag_res, __float2half(0.0f));

                for (int k_idx = 0; k_idx < BLOCK_N; k_idx += WMMA_K)
                {
                    int aCol = k_idx;
                    int bRow = k_idx;

                    wmma::load_matrix_sync(frag_s, acc_shared + aRow * lda + aCol, lda);
                    wmma::load_matrix_sync(frag_v, kv_shared + bRow * ldb + bCol, ldb);
                    wmma::mma_sync(frag_res, frag_s, frag_v, frag_res);
                }

                if(load_num_id > 0)
                {
                    wmma::load_matrix_sync(frag_res_last, res_shared + aRow * ldc + bCol, ldc, wmma::mem_row_major);

                    int update_BLOCK_M_row_idx = aRow + laneId / WMMA_N_PARTITION_WIDTH;
                    float alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[0] = __hadd(frag_res.x[0], __hmul(frag_res_last.x[0], __float2half(alpha)));
                    frag_res.x[1] = __hadd(frag_res.x[1], __hmul(frag_res_last.x[1], __float2half(alpha)));

                    frag_res.x[4] = __hadd(frag_res.x[4], __hmul(frag_res_last.x[4], __float2half(alpha)));
                    frag_res.x[5] = __hadd(frag_res.x[5], __hmul(frag_res_last.x[5], __float2half(alpha)));
                    
                    update_BLOCK_M_row_idx = aRow + WMMA_M_PARTITION_HEIGHT + laneId / WMMA_N_PARTITION_WIDTH;
                    alpha = expf(last_rowmax[update_BLOCK_M_row_idx] - global_rowmax[update_BLOCK_M_row_idx]);

                    frag_res.x[2] = __hadd(frag_res.x[2], __hmul(frag_res_last.x[2], __float2half(alpha)));
                    frag_res.x[3] = __hadd(frag_res.x[3], __hmul(frag_res_last.x[3], __float2half(alpha)));

                    frag_res.x[6] = __hadd(frag_res.x[6], __hmul(frag_res_last.x[6], __float2half(alpha)));
                    frag_res.x[7] = __hadd(frag_res.x[7], __hmul(frag_res_last.x[7], __float2half(alpha)));
                }
                
                wmma::store_matrix_sync(res_shared + aRow * ldc + bCol, frag_res, ldc, wmma::mem_row_major);
            }
        }
        __syncthreads();

    }

    // ---------------------------- Write Result ----------------------------------
    // ----------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < per_thread_deal_q_half2; i++) 
    {   
        int q_offset_shared_half2_ptr = i * 640; // num_warps * (head_size + SKEW_HALF) / 2
        int shared_ptr = q_offset_shared_half2_ptr + warpId * 40 + laneId;

        int q_offset_global_half2_ptr = i * 512; // num_warps * WARP_SIZE
        int global_ptr = q_offset_global_half2_ptr + tid;

        float global_rowsum_value = global_rowsum[shared_ptr * 2 / (head_size + SKEW_HALF)]; 

        half2 half2_val = res_vec[shared_ptr];
        half2_val.x = __hdiv(half2_val.x, __float2half(global_rowsum_value));
        half2_val.y = __hdiv(half2_val.y, __float2half(global_rowsum_value));

        q_vec[(offset_common + offset_BLOCK_M * stride_2)/2 + global_ptr] = half2_val;
    }   
}


void launch_block_attn_mask(
    __half *q, __half *k, __half *v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int batch_size, const int seq_len, const int head_num, const int head_size)
{
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);

    const int skew_size_q_half = BLOCK_M * (head_size + SKEW_HALF);
    const int skew_size_kv_half = BLOCK_N * (head_size + SKEW_HALF);
    const int skew_size_acc_half = BLOCK_M * (BLOCK_N + SKEW_HALF);

    size_t shared_mem_per_block = devProp.sharedMemPerBlock;
    size_t required_shared_mem =
        (skew_size_q_half   * sizeof(__half) +      // Q
         skew_size_kv_half  * sizeof(__half) +      // K V
         skew_size_acc_half * sizeof(__half) +      // acc
         skew_size_q_half   * sizeof(__half) +      // res
         4 * BLOCK_M * sizeof(float)                // rowmax  rowsum 
        );
    
    
    // printf(">>>      num_warps: %d, BLOCK_M: %d, BLOCK_N: %d\n", num_warps, BLOCK_M, BLOCK_N);
    // printf(">>>      Shared Memory Require: %.2f KB, and Device Config %.2f KB\n\n", required_shared_mem / 1024.0, shared_mem_per_block/1024.0);
    if (required_shared_mem > shared_mem_per_block)
    {
        printf(">>>     [ERROR] Shared Memory is not enough !\n");
        printf(">>>      BLOCK_M: %d, BLOCK_N: %d\n", BLOCK_M, BLOCK_N);
        printf(">>>      Shared Memory Require: %.2f KB, and Device Config %.2f KB\n\n", required_shared_mem / 1024.0, shared_mem_per_block/1024.0);
        return;
    }
    

    dim3 gridSize(seq_len / BLOCK_M, batch_size, head_num); 

    if(num_warps == 1)
    {
        dim3 blockSize(WARP_SIZE);

        if(BLOCK_N == 16) // mxxxn16
        {
            block_attn_mask_1warp_n16_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if(BLOCK_N == 32) // m32n32 ...
        {
            block_attn_mask_1warp_n32_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if(BLOCK_N == 64) // m32n64 ...
        {
            block_attn_mask_1warp_n64_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if(BLOCK_N == 128) // m32n64 ...
        {
            block_attn_mask_1warp_n128_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
    }
    else if(num_warps == 2)
    {
        dim3 blockSize(2 * WARP_SIZE);
        if(BLOCK_N == 16) //  m32n16 m64n16
        {
            block_attn_mask_2warp_n16_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if(BLOCK_N == 32) // m16n32 m32n32 m64n32
        {
            block_attn_mask_2warp_n32_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if(BLOCK_N == 64) //  m16n64 m32n64 m64n64
        {
            block_attn_mask_2warp_n64_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if(BLOCK_N == 128) // m16n128 m32n128
        {
            block_attn_mask_2warp_n128_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
    }


    else if(num_warps == 4)   //m16n64 m64n16  |  m32n32 m64n32  |  m32n64  m64n64
    {
        dim3 blockSize(4 * WARP_SIZE);
        if(BLOCK_N == 16) // m64n16
        {
            block_attn_mask_4warp_n16_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }  
        else if(BLOCK_N == 32) //  m32n32 m64n32 
        {
            block_attn_mask_4warp_n32_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if(BLOCK_M != 16 && BLOCK_N == 64)  // m32n64 m64n64
        { 
            block_attn_mask_4warp_n64_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if(BLOCK_M == 16 && BLOCK_N == 64) // m16n64
        {
            block_attn_mask_4warp_m16n64_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if (BLOCK_N == 128) // m32n128 m16n128
        {
            if (BLOCK_M == 16) //  m16n128
            {
                block_attn_mask_4warp_m16n128_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                    q, k, v,
                    full_row_ptr, full_col_idx,
                    part_row_ptr, part_col_idx, part_block_mask,
                    load_row_ptr, load_col_idx,
                    BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                    stride_0, stride_1, stride_2,
                    head_size);
            }
            else if (BLOCK_M == 32) // m32n128
            {
                block_attn_mask_4warp_m32n128_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                    q, k, v,
                    full_row_ptr, full_col_idx,
                    part_row_ptr, part_col_idx, part_block_mask,
                    load_row_ptr, load_col_idx,
                    BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                    stride_0, stride_1, stride_2,
                    head_size);
            }
        }
        
    }
    else if(num_warps == 8) //  m32n32 m64n32  |  m64n64 | m16n128 | m32n128
    {
        dim3 blockSize(8 * WARP_SIZE);

        if(BLOCK_N == 32) //  m64n32 
        {
            block_attn_mask_8warp_n32_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if(BLOCK_N == 64)  // m64n64 m32n64
        { 
            block_attn_mask_8warp_n64_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if(BLOCK_N == 128)  //  m16n128 m32n128
        { 
            if(BLOCK_M == 16){ //  m16n128 
                block_attn_mask_8warp_m16n128_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                    q, k, v,
                    full_row_ptr, full_col_idx,
                    part_row_ptr, part_col_idx, part_block_mask,
                    load_row_ptr, load_col_idx,
                    BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                    stride_0, stride_1, stride_2,
                    head_size);
            }
            else { //  m32n128 
                block_attn_mask_8warp_m32n128_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                    q, k, v,
                    full_row_ptr, full_col_idx,
                    part_row_ptr, part_col_idx, part_block_mask,
                    load_row_ptr, load_col_idx,
                    BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                    stride_0, stride_1, stride_2,
                    head_size);
            }
        }
    }   
    else if(num_warps == 16) // m64n64  m32n128
    {
        dim3 blockSize(16 * WARP_SIZE);
        
        if (BLOCK_N == 64) // m64n64
        {
            block_attn_mask_16warp_n64_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
        else if (BLOCK_N == 128) // m32n128
        {
            block_attn_mask_16warp_n128_kernel<<<gridSize, blockSize, required_shared_mem>>>(
                q, k, v,
                full_row_ptr, full_col_idx,
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps, softmax_scale,
                stride_0, stride_1, stride_2,
                head_size);
        }
    }



  

    
}