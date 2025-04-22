#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>

#include <iostream>
#include <cuda_fp16.h>

void launch_block_attn_mask(
    __half* q, __half* k, __half* v,
    const int* full_row_ptr, const int* full_col_idx,
    const int* part_row_ptr, const int* part_col_idx, __half* part_block_mask,
    const int* load_row_ptr, const int* load_col_idx,
    const int BLOCK_M, const int BLOCK_N, const int num_warps, const float softmax_scale,
    const int stride_0, const int stride_1, const int stride_2,
    const int batch_size, const int seq_len, const int head_num, const int head_size);

void block_attn_mask_gpu(
    at::Tensor q, at::Tensor k, at::Tensor v, 
    at::Tensor full_row_ptr, at::Tensor full_col_idx,
    at::Tensor part_row_ptr, at::Tensor part_col_idx, at::Tensor part_block_mask,
    at::Tensor load_row_ptr, at::Tensor load_col_idx,
    int BLOCK_M, int BLOCK_N, int num_warps)
{
    const auto batch_size = q.size(0);
    const auto head_num = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_size = q.size(3);

    const auto stride_0 = q.stride(0);
    const auto stride_1 = q.stride(1);
    const auto stride_2 = q.stride(2);

    const float scaled_value = 1.0f / sqrtf(head_size);

    launch_block_attn_mask(
        reinterpret_cast<__half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v.data_ptr<at::Half>()),
        full_row_ptr.data_ptr<int>(),
        full_col_idx.data_ptr<int>(),
        part_row_ptr.data_ptr<int>(),
        part_col_idx.data_ptr<int>(),
        reinterpret_cast< __half*>(part_block_mask.data_ptr<at::Half>()),
        load_row_ptr.data_ptr<int>(),
        load_col_idx.data_ptr<int>(),
        BLOCK_M, BLOCK_N, num_warps, scaled_value,
        stride_0, stride_1, stride_2,
        batch_size, seq_len, head_num, head_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "fused_bolock_attn: Test for SC25, varity of blockSize";
    m.def("forward", &block_attn_mask_gpu, "launch_fused_block_attn with sparse mask"); 
} 