#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h> 



void launcher_bytetr_attn(
    __half *qkv, __half *qkv_bias_ptr,
    __half* mask, __half* attention_output,
    const int batch_size, const int head_num, const int seq_len, const int head_size,
    cudaStream_t stream);

void bytetr_attn_gpu(at::Tensor qkv, at::Tensor qkv_bias_ptr,at::Tensor mask,at::Tensor attention_output,int num_head)
{
    const auto batch_size = qkv.size(0);  
    const auto head_num   = num_head;
    const auto seq_len    = qkv.size(1);  
    const auto head_size  = qkv.size(2)/3/num_head;  

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launcher_bytetr_attn(
        reinterpret_cast< __half*>(qkv.data_ptr<at::Half>()),
        reinterpret_cast< __half*>(qkv_bias_ptr.data_ptr<at::Half>()),
        reinterpret_cast< __half*>(mask.data_ptr<at::Half>()),
        reinterpret_cast< __half*>(attention_output.data_ptr<at::Half>()),
        batch_size, head_num, seq_len, head_size,
        stream);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Byte_Attn: Test for SC25";
    m.def("forward", &bytetr_attn_gpu, "bytetr_attn op for high sparsity"); 
}
