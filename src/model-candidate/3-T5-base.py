# 
# python 3-T5-base.py
import torch
import numpy as np
import random
import torch.nn.functional as F
from utils.utils import transpose_for_scores, torch_cuda_identify, set_dtype, seqlen_to_mask, time_stamp_cudasync
from utils.masks import generate_triangle_mask, generate_strided_mask, generate_fixed_mask
import config

def T5_base_fwd_std():
    encoder_hidden_states = input_from_tensor
    
    # Encoder ---------------------------------------------------
    for layer in range(layer_num):
        input_tensor = encoder_hidden_states
        qkv = torch.matmul(encoder_hidden_states, qkv_kernel[layer])
        qkv = qkv + qkv_bias[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q = transpose_for_scores(q, head_num, head_size)
        k = transpose_for_scores(k, head_num, head_size)
        v = transpose_for_scores(v, head_num, head_size)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
        probs = F.softmax(scores, dim=-1)
        h = torch.matmul(probs, v)
        
        h = h.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        encoder_hidden_states = h.view(new_context_layer_shape)
        #attention output projection GEMM + bias
        encoder_hidden_states = torch.matmul(encoder_hidden_states, attr_output_kernel[layer])
        encoder_hidden_states = encoder_hidden_states + attr_output_bias[layer]
        encoder_hidden_states = encoder_hidden_states + input_tensor 
        # layer_Norm
        encoder_hidden_states = F.layer_norm(encoder_hidden_states, (hidden_dim, ), weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
        encoder_residual = encoder_hidden_states   
        #FFN GEMM 1 + add bias 
        encoder_hidden_states = torch.matmul(encoder_hidden_states, inter_kernel[layer])
        encoder_hidden_states = encoder_hidden_states + inter_bias[layer] 
        # T5 relu
        encoder_hidden_states = F.relu(encoder_hidden_states)  
        #FFN GEMM 2 + add bias
        encoder_hidden_states = torch.matmul(encoder_hidden_states, output_kernel[layer]) 
        encoder_hidden_states = encoder_hidden_states + output_bias[layer]  
        encoder_hidden_states = encoder_hidden_states + encoder_residual  
        # layer_Norm
        encoder_hidden_states = F.layer_norm(encoder_hidden_states, (hidden_dim, ), weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
        
        
    # Ready QK for Decoder
    Encoder_qkv = torch.matmul(encoder_hidden_states, qkv_kernel[layer])
    Encoder_qkv = Encoder_qkv + qkv_bias[layer]
    encoder_q, encoder_k, encoder_v = Encoder_qkv.chunk(3, dim=-1)
    encoder_q = transpose_for_scores(encoder_q, head_num, head_size)
    encoder_k = transpose_for_scores(encoder_k, head_num, head_size)
    
        
    # Decoder ---------------------------------------------------
    for layer in range(layer_num):
        
        qkv = torch.matmul(output_from_tensor, qkv_kernel_2[layer])
        qkv = qkv + qkv_bias_2[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q = transpose_for_scores(q, head_num, head_size)
        k = transpose_for_scores(k, head_num, head_size)
        v = transpose_for_scores(v, head_num, head_size)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
        scores -= 10000.0 * (1.0 - mask.unsqueeze(1)) 
        probs = F.softmax(scores, dim=-1)
        h = torch.matmul(probs, v)
        
        h = h.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        decoder_hidden_states = h.view(new_context_layer_shape)
        #attention output projection GEMM + bias
        decoder_hidden_states = torch.matmul(decoder_hidden_states, attr_output_kernel_2[layer])
        decoder_hidden_states = decoder_hidden_states + attr_output_bias_2[layer]
        decoder_hidden_states = decoder_hidden_states + output_from_tensor 
        # layer_Norm
        decoder_hidden_states = F.layer_norm(decoder_hidden_states, (hidden_dim, ), weight=attn_lynorm_gamma_2[layer], bias=attn_lynorm_beta_2[layer])
        decoder_residual = decoder_hidden_states       
        
        
        # cross-Attention
        qkv = torch.matmul(decoder_hidden_states, qkv_kernel_3[layer])
        qkv = qkv + qkv_bias_3[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        v = transpose_for_scores(v, head_num, head_size)
        
        scores = torch.matmul(encoder_q, encoder_k.transpose(-2, -1)) / (head_size ** .5)
        probs = F.softmax(scores, dim=-1)
        h = torch.matmul(probs, v)
        
        h = h.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        decoder_hidden_states = h.view(new_context_layer_shape)
        #attention output projection GEMM + bias
        decoder_hidden_states = torch.matmul(decoder_hidden_states, crattr_output_kernel_2[layer])
        decoder_hidden_states = decoder_hidden_states + crattr_output_bias_2[layer]
        decoder_hidden_states = decoder_hidden_states + decoder_residual
        # layer_Norm
        decoder_hidden_states = F.layer_norm(decoder_hidden_states, (hidden_dim, ), weight=crattn_lynorm_gamma_2[layer], bias=crattn_lynorm_beta_2[layer])
        decoder_residual = decoder_hidden_states
        
        # FFN GEMM 1 + add bias 
        decoder_hidden_states = torch.matmul(decoder_hidden_states, inter_kernel_2[layer])
        decoder_hidden_states = decoder_hidden_states + inter_bias_2[layer] 
        # T5  relu
        decoder_hidden_states = F.relu(decoder_hidden_states)  
        # FFN GEMM 2 + add bias
        decoder_hidden_states = torch.matmul(decoder_hidden_states, output_kernel_2[layer]) 
        decoder_hidden_states = decoder_hidden_states + output_bias_2[layer]  
        decoder_hidden_states = decoder_hidden_states + decoder_residual
        # layer_Norm
        encoder_hidden_states = F.layer_norm(decoder_hidden_states, (hidden_dim, ), weight=lynorm_gamma_2[layer], bias=lynorm_beta_2[layer])  
        
        transformer_output[layer] = encoder_hidden_states
        



if __name__ == '__main__':
    torch.manual_seed(0)
    torch_cuda_identify(True)
    
    batch_size = 8 # 16, 32, 48
    head_size = 64 # 768 / 12 = 64
    head_num = 12 
    layer_num = 12
    hidden_dim = head_num * head_size # 768
    seq_len = config.SEQ_LEN  
    mask_id = config.MASK_ID

    
    warmup_iters = config.WARMUP_TIME
    running_iters = config.RUNNING_TIME
    dtype = config.DATA_TYPE
    

    avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), "fp16")
    
    lower_triangle_mask = generate_triangle_mask(attr_mask).cuda()
    strided_mask = generate_strided_mask(attr_mask).cuda()
    fixed_mask = generate_fixed_mask(attr_mask).cuda()
    
    if(mask_id == 0):
        mask_name = 'Lower_triangle_mask'
        mask = lower_triangle_mask
    elif(mask_id == 1): 
        mask_name = 'Strided_mask'
        mask = strided_mask
    else:
        mask_name = 'Fixed_mask'
        mask = fixed_mask
    
    input_from_tensor = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
    output_from_tensor = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
    qkv_kernel = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_bias = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_kernel = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_bias = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_gamma = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_beta  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_kernel  = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_bias    = [set_dtype(torch.zeros(hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_kernel = [set_dtype(torch.zeros(hidden_dim * 4, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_bias            = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_gamma = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_beta  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    
    qkv_kernel_2 = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_bias_2 = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_kernel_3 = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_bias_3 = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    
    attr_output_kernel_2 = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_bias_2 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    crattr_output_kernel_2 = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    crattr_output_bias_2 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    
    attn_lynorm_gamma_2 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attn_lynorm_beta_2  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    crattn_lynorm_gamma_2 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    crattn_lynorm_beta_2  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    
    inter_kernel_2 = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_bias_2   = [set_dtype(torch.zeros(hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_kernel_2 = [set_dtype(torch.zeros(hidden_dim * 4, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_bias_2   = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    lynorm_gamma_2 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    lynorm_beta_2  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]

    transformer_output = [None for _ in range(layer_num)]
    
    # T5_base_fwd_std()
    # result = transformer_output[-1]
    # print(result)
    
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t0_start = time_stamp_cudasync()
        T5_base_fwd_std()
    t0_end = time_stamp_cudasync()
    base_time = (t0_end - t0_start) * 1000 / running_iters
    print("T5-base  |  bs:{}  |  seq:{}  |  Torch Naive    : {:.3f} ms / iter".format(batch_size, seq_len, base_time)) 