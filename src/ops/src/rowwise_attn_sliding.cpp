#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>

#include <iostream>
#include <cuda_fp16.h>


void launcher_rowwise_attn_sliding(
    const __half* q, const __half* k, const __half* v, bool is_causal, __half* result,
    const int stride_0, const int stride_1, const int stride_2, const int band_width,
    const int batch_size, const int head_num, const int seq_len, const int head_size);

void rowwise_attn_sliding_gpu(at::Tensor q, at::Tensor k, at::Tensor v, bool is_causal, at::Tensor result, int band_width)
{
    const auto batch_size = q.size(0);
    const auto head_num   = q.size(1);
    const auto seq_len    = q.size(2);
    const auto head_size  = q.size(3);

    const auto stride_0 = q.stride(0);
    const auto stride_1 = q.stride(1);
    const auto stride_2 = q.stride(2);

    launcher_rowwise_attn_sliding(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        is_causal,
        reinterpret_cast< __half*>(result.data_ptr<at::Half>()),
        stride_0, stride_1, stride_2,
        band_width,
        batch_size, head_num, seq_len, head_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "My_fused_attention: Test for SC25";
    m.def("forward", &rowwise_attn_sliding_gpu, "rowwise_attn_band op for high sparsity"); 
} 