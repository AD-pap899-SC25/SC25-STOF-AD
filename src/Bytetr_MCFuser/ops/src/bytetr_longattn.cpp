#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

// typedef struct {
//     int *batch_idx;
//     int *word_idx;
//     int valid_word_num;
//   } ET_Paramm;

// static unsigned long long make_align(unsigned long long val) {
//     return ((val + 15) >> 4) << 4;
//   }

void launcher_bytetr_longattn(void *buf,
    const int batch_size,const int seq_len,const int head_num,const int head_size,
    // cudaStream_t stream,ET_Paramm et_param,
    cudaStream_t stream,
    unsigned long long input_tensor_size,
    unsigned long long qk_outputt,
    unsigned long long partial_softmax,
    const bool is_remove_padding_,int multi_processor_count_,
    __half *qkv, __half *attr_bias_QKV,
    __half *atten_mask,float tao,
    __half *attention_output);

void bytetr_longattn_gpu(at::Tensor qkv, at::Tensor qkv_bias_ptr,at::Tensor atten_mask,at::Tensor attention_output,int num_head)
{
    const auto batch_size = qkv.size(0);  
    const auto head_num   = num_head;
    const auto seq_len    = qkv.size(1);  
    const auto head_size  = qkv.size(2)/3/num_head;
   
    auto make_align = [](unsigned long long val) { 
        return ((val + 15) >> 4) << 4; 
    };

    float      tao        = 1.0f;
    unsigned long long input_tensor_size = make_align(1ULL * batch_size * seq_len * head_num * head_size) * sizeof(__half);
    unsigned long long qk_outputt = make_align(1ULL * batch_size * head_num * seq_len * seq_len) * sizeof(__half);
    unsigned long long partial_softmax = make_align(1ULL * batch_size * seq_len * head_num * 32) * sizeof(float) * 2;
    const bool is_remove_padding_=false;
    int multi_processor_count_=108;
        
    const unsigned long long total_buf_size = 
    input_tensor_size * 3 +  // query + key + value
    qk_outputt + 
    partial_softmax * 2;     // partial_softmax_buf + softmax_reduced_buf


    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(qkv.device());
    at::Tensor buf_tensor = at::empty({(long)total_buf_size}, options);
    void* buf = buf_tensor.data_ptr();
    

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launcher_bytetr_longattn(
        buf,
        batch_size, seq_len, head_num, head_size, 
        stream,
        input_tensor_size,
        qk_outputt,
        partial_softmax,
        is_remove_padding_, multi_processor_count_,
        reinterpret_cast< __half*>(qkv.data_ptr<at::Half>()),
        reinterpret_cast< __half*>(qkv_bias_ptr.data_ptr<at::Half>()),
        reinterpret_cast< __half*>(atten_mask.data_ptr<at::Half>()),
        tao,
        reinterpret_cast< __half*>(attention_output.data_ptr<at::Half>())
        );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Byte_Longattn: Test for SC25";
    m.def("forward", &bytetr_longattn_gpu, "bytetr_longattn op for high sparsity"); 
}
