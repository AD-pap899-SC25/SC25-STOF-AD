#include "include/common.h"
#include "include/cutlass_attention.h"
#include "include/cutlass_attention_defs.h"
#include "include/attention_nofused_utils.h"
#include "include/cutlass_contrib/include/cutlass/contrib/args_pack_def.h"
#include "cassert"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <stdio.h>
#include <string.h>

using namespace bytetransformer;
using namespace cutlass_ops;


__global__ void add_QKV_bias(const __half *QKV, const __half *bias_QKV, __half *q_buf,
  __half *k_buf, __half *v_buf, const int batch_size,
  const int seq_len, const int head_num,
  const int half_head_size, const bool is_roformer) {
int batch_id = blockIdx.y;
int seq_id = blockIdx.x;
int head_id = threadIdx.x / half_head_size;
int id = threadIdx.x % half_head_size;
int src_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * 3) + threadIdx.x;
int trt_id = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_head_size + id;
half2 q_value =
__hadd2(((const half2 *)QKV)[src_id], __ldg(&((const half2 *)bias_QKV)[threadIdx.x]));
half2 k_value = __hadd2(((const half2 *)QKV)[src_id + blockDim.x],
__ldg(&((const half2 *)bias_QKV)[threadIdx.x + blockDim.x]));
half2 v_value = __hadd2(((const half2 *)QKV)[src_id + blockDim.x * 2],
__ldg(&((const half2 *)bias_QKV)[threadIdx.x + blockDim.x * 2]));

if (is_roformer) {
half2 ro_q = half2(__hneg(q_value.y), q_value.x);
half2 ro_k = half2(__hneg(k_value.y), k_value.x);
float position_enc = __fdividef(seq_id, __powf(10000.0f, __fdividef(id, half_head_size)));
half2 sin_pos = __float2half2_rn(__sinf(position_enc));
half2 cos_pos = __float2half2_rn(__cosf(position_enc));
q_value = __hadd2(__hmul2(q_value, cos_pos), __hmul2(ro_q, sin_pos));
k_value = __hadd2(__hmul2(k_value, cos_pos), __hmul2(ro_k, sin_pos));
}

((half2 *)q_buf)[trt_id] = q_value;
((half2 *)k_buf)[trt_id] = k_value;
((half2 *)v_buf)[trt_id] = v_value;
}


template <typename Gemm, typename DataType, int kMaxThreadblockNumInRow>
void gemm0_and_softmax_reduce_kernel_launcher(DataType *query, DataType *key, DataType *atten_mask,
                                              DataType *qk_output, float *partial_softmax_buf,
                                              float *softmax_reduced_buf, int *seqlen_offsets,
                                              const int batch_size, const int seq_len,
                                              const int head_num, const int head_size,
                                              const float tao, const bool is_remove_padding,
                                              const int sm_count, cudaStream_t stream) {
  using GemmKernel = typename Gemm::GemmKernel;
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;
  using ElementD = typename GemmKernel::ElementD;

  using Shape = typename GemmKernel::ThreadblockShape;
  const auto grid_shape = cutlass::gemm::GemmCoord(
      (seq_len + Shape::kM - 1) / Shape::kM, (seq_len + Shape::kN - 1) / Shape::kN, head_num);
  if (grid_shape.n() >= kMaxThreadblockNumInRow) {
    throw std::runtime_error("grid_shape.n(): " + std::to_string(grid_shape.n()) +
                             " exceeds maximum: " + std::to_string(kMaxThreadblockNumInRow));
  }
  static const int max_active_blocks = Gemm::maximum_active_blocks();
  const int tile_count = grid_shape.m() * grid_shape.n() * grid_shape.k();
  const int cta_count = std::min(tile_count, max_active_blocks * sm_count);

  typename GemmKernel::ParamsDef::ProblemSizeOperator::Params problem_size_op{seqlen_offsets,
                                                                              seq_len};
  typename GemmKernel::ParamsDef::BatchCountOperator::Params batch_count_op{head_num};
  typename AttentionTensorParamGeneratorOp<ElementA>::Params param_A_op{
      reinterpret_cast<ElementA *>(query), head_size, batch_size * seq_len * head_size,
      seq_len * head_size, nullptr};
  typename AttentionTensorParamGeneratorOp<ElementB>::Params param_B_op{
      reinterpret_cast<ElementB *>(key), head_size, batch_size * seq_len * head_size,
      seq_len * head_size, nullptr};
  typename AttentionTensorParamGeneratorOp<ElementC>::Params param_C_op{
      reinterpret_cast<ElementC *>(atten_mask), seq_len, 0, seq_len * seq_len, nullptr};
  typename AttentionTensorParamGeneratorOp<ElementD>::Params param_D_op{
      reinterpret_cast<ElementD *>(qk_output), seq_len, seq_len * seq_len,
      seq_len * seq_len * head_num, nullptr};

  ElementC alpha(1.0f / sqrt(head_size * 1.0f) / tao);
  ElementC beta(1.0);
  typename GemmKernel::EpilogueOutputOp::Params epilogue{
      alpha, beta, head_num, seq_len, grid_shape.n(), partial_softmax_buf};
  const int problem_count = batch_size;

  auto args = typename Gemm::Arguments(cutlass::gemm::GemmUniversalMode::kBatched, problem_count,
                                       cta_count, problem_size_op, batch_count_op,
                                       {},        // prologue_A, identity
                                       {},        // prolgoue_B, identity
                                       epilogue,  // partial softmax epilogue
                                       param_A_op, param_B_op, param_C_op, param_D_op);
  // launch kernel
  auto gemm = Gemm();
  auto status = gemm.initialize(args, nullptr, stream);
  CUTLASS_CHECK(status);
  status = gemm(stream);
  CUTLASS_CHECK(status);

  // lightweight kernel to calculate the final reduction for softmax
  softmax_reduction_kernel_launcher<float, Shape::kN>(
      partial_softmax_buf, seqlen_offsets, softmax_reduced_buf, batch_size, head_num, seq_len,
      grid_shape.n(), is_remove_padding, stream);
}

template <typename Gemm, typename DataType>
void gemm1_kernel_launcher(DataType *qk_output, DataType *value, DataType *attention_output,
                           float *softmax_reduced_buf, int const *seqlen_offsets,
                           const int batch_size, const int seq_len, const int head_num,
                           const int head_size, const bool is_remove_padding,
                           const int sm_count, cudaStream_t stream) {
  using GemmKernel = typename Gemm::GemmKernel;
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;
  using ElementD = typename GemmKernel::ElementD;

  using Shape = typename GemmKernel::ThreadblockShape;
  const auto grid_shape = cutlass::gemm::GemmCoord(
      (seq_len + Shape::kM - 1) / Shape::kM, (seq_len + Shape::kN - 1) / Shape::kN, head_num);
  const int tile_count = grid_shape.m() * grid_shape.n() * grid_shape.k();
  static const int max_active_blocks = Gemm::maximum_active_blocks();
  const int cta_count = std::min(tile_count, max_active_blocks * sm_count);
  const int hidden_dim = head_num * head_size;

  typename GemmKernel::ParamsDef::ProblemSizeOperator::Params problem_size_op{seqlen_offsets,
                                                                              seq_len};
  typename GemmKernel::ParamsDef::BatchCountOperator::Params batch_count_op{head_num};

  typename AttentionTensorParamGeneratorOp<ElementA>::Params param_A_op{
      reinterpret_cast<ElementA *>(qk_output), seq_len, seq_len * seq_len,
      seq_len * seq_len * head_num, nullptr};

  typename AttentionTensorParamGeneratorOp<ElementB>::Params param_B_op{
      reinterpret_cast<ElementB *>(value), head_size, batch_size * seq_len * head_size,
      seq_len * head_size, nullptr};
  typename AttentionTensorParamGeneratorOp<ElementC>::Params param_C_op{nullptr, 0, 0, 0};
  typename AttentionTensorParamGeneratorOp<ElementD>::Params param_D_op{
      reinterpret_cast<ElementD *>(attention_output), hidden_dim, head_size,
      is_remove_padding ? hidden_dim : seq_len * hidden_dim, seqlen_offsets};

  const int problem_count = batch_size;
  typename GemmKernel::PrologueDefA::Operator::Params prologue_A{head_num, seq_len,
                                                                 softmax_reduced_buf};
  auto args = typename Gemm::Arguments(cutlass::gemm::GemmUniversalMode::kBatched, problem_count,
                                       cta_count, problem_size_op, batch_count_op,
                                       prologue_A,                      // partial softmax prologue
                                       {},                              // identity
                                       {ElementC(1.0), ElementC(0.0)},  // epilogue
                                       param_A_op, param_B_op, param_C_op, param_D_op);
  // launch kernel
  auto gemm = Gemm();
  auto status = gemm.initialize(args, nullptr, stream);
  CUTLASS_CHECK(status);
  status = gemm(stream);
  CUTLASS_CHECK(status);
}

typedef struct {
    int *batch_idx;
    int *word_idx;
    int valid_word_num;
  } ET_Paramm;


  template <typename CutlassAttentionCore>
  void bytetr_longattn_kernel(void *buf,
      const int batch_size, const int seq_len, const int head_num, const int head_size,
      cudaStream_t stream,
      unsigned long long input_tensor_size,
      unsigned long long qk_outputt,
      unsigned long long partial_softmax,
      const bool is_remove_padding_, int multi_processor_count_,
      __half *qkv, __half *attr_bias_QKV,
      const __half *atten_mask, float tao,
      __half *attention_output){

  // calc buf pointers
  auto query = (__half *)((uint8_t *)buf);
  auto key = (__half *)((uint8_t *)query + input_tensor_size);
  auto value = (__half *)((uint8_t *)key + input_tensor_size);
  auto qk_output = (__half *)((uint8_t *)value + input_tensor_size);
  auto partial_softmax_buf = (float *)((uint8_t *)qk_output + qk_outputt);
  auto softmax_reduced_buf =(float *)((uint8_t *)partial_softmax_buf + partial_softmax);

  const bool is_roformer = false;
  // add bias
  if constexpr (true) {
    dim3 grid, block;
    const int head_size_half = head_size / 2; 
    
    // [batch_size, seq_len, hidden_dim] -> [head_num, batch_size, seq_len, head_size]
    grid.x = seq_len;
    grid.y = batch_size;
    block.x = head_num * head_size_half;
    // if (is_remove_padding_) {
    //   add_QKV_bias_padding<<<grid, block, 0, stream>>>(  // restore & clean zero for batch_gemm
    //       qkv, attr_bias_QKV, query, key, value, batch_size, seq_len,
    //       head_num, head_size_half, is_roformer, et_param.batch_idx, et_param.word_idx);
    // } else {
      add_QKV_bias<<<grid, block, 0, stream>>>(qkv,attr_bias_QKV, query,
                                               key, value, batch_size, seq_len, head_num,
                                               head_size_half, is_roformer);
    // }
    // printf(query)
  }

  __half *atten_mask_noconst = const_cast<__half *>(atten_mask);
  // int *seqlen_offsets = is_remove_padding_ ? et_param.batch_idx : nullptr;
  int *seqlen_offsets = nullptr;

  gemm0_and_softmax_reduce_kernel_launcher<typename CutlassAttentionCore::Gemm0, __half,32>(
      query, key, atten_mask_noconst, qk_output, partial_softmax_buf, softmax_reduced_buf,
      seqlen_offsets, batch_size, seq_len, head_num, head_size, tao,
      is_remove_padding_, multi_processor_count_, stream);

  gemm1_kernel_launcher<typename CutlassAttentionCore::Gemm1, __half>(
      qk_output, value,attention_output, softmax_reduced_buf, seqlen_offsets,
      batch_size, seq_len, head_num, head_size, is_remove_padding_,
      multi_processor_count_, stream);
}


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
  __half *attention_output) {

  int cur_seq_len =seq_len;
  // printf("%d",cur_seq_len);
  using CutlassAttentionCoree = bytetransformer::cutlass_ops::CutlassAttentionCore<1024, 64, cutlass::arch::Sm80, 2, ModelType::Bert>;
  if (cur_seq_len <= 1024) {
    // do_infer<1024, 64>(buf,
    bytetr_longattn_kernel<CutlassAttentionCoree>(buf,
        batch_size, seq_len, head_num, head_size,
        // stream, et_param,
        stream,
        input_tensor_size,
        qk_outputt, 
        partial_softmax,
        is_remove_padding_, multi_processor_count_,
        qkv, attr_bias_QKV,
        atten_mask,tao,
        attention_output);
        return;
  }
  printf("[ERROR][exec] unsupport seq_len!\n");
  exit(-1);
}

