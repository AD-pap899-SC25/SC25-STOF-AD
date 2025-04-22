#  python correct_verify_end2end.py

import sys
import os
import math
import torch._dynamo


from ops.package_op import block_attn_mask_op
from ops.package_op import rowwise_attn_full_op
from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, transpose_for_scores
from util.masks import generate_causal_mask, get_sparse_storage
from util.masks import flex_causal_mask

import argparse
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def new_gelu(input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

def check_tensor(other_output, torch_output):
    max_diff = (other_output - torch_output).abs().max().item()
    mean_diff = (other_output - torch_output).abs().mean().item()
    return max_diff, mean_diff


# Bert-small | Bert-base | Bert-large ----------------------------------------
# ----------------------------------------------------------------------------
def bert_fwd_std(mask):
    with torch.no_grad():
        hidden_states = input_from_tensor
        for layer in range(layer_num):
            input_tensor = hidden_states

            qkv = qkv_kernel[layer]  + qkv_bias[layer]
            q, k, v = qkv.chunk(3, dim=-1)
            q = transpose_for_scores(q, head_num, head_size)
            k = transpose_for_scores(k, head_num, head_size)
            v = transpose_for_scores(v, head_num, head_size)

            # ------------------------------------------------------------- Attention start
            if attention_type=="torch_attention":
                scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
                scores -= 10000.0 * (1.0 - mask.unsqueeze(1))
                probs = F.softmax(scores, dim=-1)
                h = torch.matmul(probs, v)
                
                h = h.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                hidden_states = h.view(new_context_layer_shape)
                
            elif (attention_type == 'ByteTransformer'):
                mask=mask.half()
                if seq_len<=256:
                    result = bytetr_attn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
                else:
                    result = bytetr_longattn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
                h = result.view(batch_size, seq_len, head_num, head_size).permute(0, 2, 1, 3).contiguous()
                
                h=h.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                hidden_states = h.view(new_context_layer_shape)
                
            
            elif (attention_type == 'STOF_attention'):
                if seq_len<=256:
                    hidden_states = rowwise_attn_full_op(q, k, v, is_causal)
                elif (seq_len == 4096):
                    query1 = q.clone()
                    h = block_attn_mask_op(query1, k, v,
                                    full_row_ptr, full_col_idx, 
                                    part_row_ptr, part_col_idx, part_block_mask,
                                    load_row_ptr, load_col_idx,
                                    BLOCK_M, BLOCK_N, num_warps)
                    h = h.permute(0, 2, 1, 3).contiguous()
                    new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                    hidden_states = h.view(new_context_layer_shape)
                else:
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        FA2_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                    h = FA2_output.permute(0, 2, 1, 3).contiguous()
                    new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                    hidden_states = h.view(new_context_layer_shape)
                    
            # ------------------------------------------------------------ Attention End
            #attention output projection GEMM
            hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
            hidden_states = hidden_states + input_tensor  

            
            # layer_Norm
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                        weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
            
            residual = hidden_states  
            #FFN GEMM 1 + add bias 
            hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer] 
            hidden_states = F.gelu(hidden_states) 
            #FFN GEMM 2 + add bias
            hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]  
            hidden_states = hidden_states + residual
            # layer_Norm
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),  
                                        weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
            transformer_output[layer] = hidden_states


# GPT2 ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
def gpt_base_fwd_std(mask):
    hidden_states = input_from_tensor
    for layer in range(layer_num):
        input_tensor = hidden_states

        qkv = qkv_kernel[layer]  + qkv_bias[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q = transpose_for_scores(q, head_num, head_size)
        k = transpose_for_scores(k, head_num, head_size)
        v = transpose_for_scores(v, head_num, head_size)

        # ------------------------------------------------------------- Attention start
        if attention_type=="torch_attention":
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
            scores -= 10000.0 * (1.0 - mask.unsqueeze(1))
            probs = F.softmax(scores, dim=-1)
            h = torch.matmul(probs, v)
            
            h = h.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            hidden_states = h.view(new_context_layer_shape)
            
        elif (attention_type == 'ByteTransformer'):
            mask=mask.half()
            if seq_len<=256:
                result = bytetr_attn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            else:
                result = bytetr_longattn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            h = result.view(batch_size, seq_len, head_num, head_size).permute(0, 2, 1, 3).contiguous()
            
            h=h.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            hidden_states = h.view(new_context_layer_shape)
            
        
        elif (attention_type == 'STOF_attention'):
            if seq_len<=256:
                hidden_states = rowwise_attn_full_op(q, k, v, is_causal)
            elif (seq_len == 4096):
                query1 = q.clone()
                h = block_attn_mask_op(query1, k, v,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, part_block_mask,
                                load_row_ptr, load_col_idx,
                                BLOCK_M, BLOCK_N, num_warps)
                h = h.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                hidden_states = h.view(new_context_layer_shape)
            else:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    FA2_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                h = FA2_output.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                hidden_states = h.view(new_context_layer_shape)
                
        # ------------------------------------------------------------ Attention End
        
        #GEMM + bias + residual
        hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer])
        hidden_states = hidden_states + attr_output_bias[layer]
        hidden_states = hidden_states + input_tensor
        
        # layer_Norm
        hidden_states = F.layer_norm(hidden_states, (hidden_dim, ), weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
        residual = hidden_states 
        
        #FFN GEMM 1 + add bias 
        hidden_states = torch.matmul(hidden_states, inter_kernel[layer])
        hidden_states = hidden_states + inter_bias[layer] 
        
        #new_gelu
        hidden_states = new_gelu(hidden_states)  
        
        #FFN GEMM 2 + add bias + residual
        hidden_states = torch.matmul(hidden_states, output_kernel[layer]) 
        hidden_states = hidden_states + output_bias[layer]  
        hidden_states = hidden_states + residual

        # layer_Norm
        hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),  weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
        transformer_output[layer] = hidden_states


# T5 ------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def T5_base_fwd_std(mask):
    encoder_hidden_states = input_from_tensor
    
    # Encoder ---------------------------------------------------
    for layer in range(layer_num):
        input_tensor = encoder_hidden_states
        qkv = torch.matmul(encoder_hidden_states, qkv_kernel_raw[layer])
        qkv = qkv + qkv_bias[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q = transpose_for_scores(q, head_num, head_size)
        k = transpose_for_scores(k, head_num, head_size)
        v = transpose_for_scores(v, head_num, head_size)
        
        # ------------------------------------------------------------- Attention start
        if attention_type=="torch_attention":
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
            scores -= 10000.0 * (1.0 - mask.unsqueeze(1))
            probs = F.softmax(scores, dim=-1)
            h = torch.matmul(probs, v)
            
            h = h.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            encoder_hidden_states = h.view(new_context_layer_shape)
            
        elif (attention_type == 'ByteTransformer'):
            mask=mask.half()
            if seq_len<=256:
                result = bytetr_attn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            else:
                result = bytetr_longattn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            h = result.view(batch_size, seq_len, head_num, head_size).permute(0, 2, 1, 3).contiguous()
            
            h=h.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            encoder_hidden_states = h.view(new_context_layer_shape)
            
        
        elif (attention_type == 'STOF_attention'):
            if seq_len<=256:
                encoder_hidden_states = rowwise_attn_full_op(q, k, v, is_causal)
            elif (seq_len == 4096):
                query1 = q.clone()
                h = block_attn_mask_op(query1, k, v,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, part_block_mask,
                                load_row_ptr, load_col_idx,
                                BLOCK_M, BLOCK_N, num_warps)
                h = h.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                encoder_hidden_states = h.view(new_context_layer_shape)
            else:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    FA2_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                h = FA2_output.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                encoder_hidden_states = h.view(new_context_layer_shape)
        # ------------------------------------------------------------ Attention End
        
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
    Encoder_qkv = torch.matmul(encoder_hidden_states, qkv_kernel_raw[layer])
    Encoder_qkv = Encoder_qkv + qkv_bias[layer]
    encoder_q, encoder_k, encoder_v = Encoder_qkv.chunk(3, dim=-1)
    encoder_q = transpose_for_scores(encoder_q, head_num, head_size)
    encoder_k = transpose_for_scores(encoder_k, head_num, head_size)
    
        
    # Decoder ---------------------------------------------------
    for layer in range(layer_num):
        
        qkv = qkv_kernel[layer]  + qkv_bias[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q = transpose_for_scores(q, head_num, head_size)
        k = transpose_for_scores(k, head_num, head_size)
        v = transpose_for_scores(v, head_num, head_size)

        # ------------------------------------------------------------- Attention start
        if attention_type=="torch_attention":
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
            scores -= 10000.0 * (1.0 - mask.unsqueeze(1))
            probs = F.softmax(scores, dim=-1)
            h = torch.matmul(probs, v)
            
            h = h.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            decoder_hidden_states = h.view(new_context_layer_shape)
            
        elif (attention_type == 'ByteTransformer'):
            mask=mask.half()
            if seq_len<=256:
                result = bytetr_attn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            else:
                result = bytetr_longattn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            h = result.view(batch_size, seq_len, head_num, head_size).permute(0, 2, 1, 3).contiguous()
            
            h=h.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            decoder_hidden_states = h.view(new_context_layer_shape)
            
        
        elif (attention_type == 'STOF_attention'):
            if seq_len<=256:
                decoder_hidden_states = rowwise_attn_full_op(q, k, v, is_causal)
            elif (seq_len == 4096):
                query1 = q.clone()
                h = block_attn_mask_op(query1, k, v,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, part_block_mask,
                                load_row_ptr, load_col_idx,
                                BLOCK_M, BLOCK_N, num_warps)
                h = h.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                decoder_hidden_states = h.view(new_context_layer_shape)
            else:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    FA2_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                h = FA2_output.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                decoder_hidden_states = h.view(new_context_layer_shape)
                
        # ------------------------------------------------------------ Attention End

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
        # print("Shape of h:", h.shape)
        
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
        # T5 relu
        decoder_hidden_states = F.relu(decoder_hidden_states)  
        # FFN GEMM 2 + add bias
        decoder_hidden_states = torch.matmul(decoder_hidden_states, output_kernel_2[layer]) 
        decoder_hidden_states = decoder_hidden_states + output_bias_2[layer]  
        decoder_hidden_states = decoder_hidden_states + decoder_residual
        # layer_Norm
        encoder_hidden_states = F.layer_norm(decoder_hidden_states, (hidden_dim, ), weight=lynorm_gamma_2[layer], bias=lynorm_beta_2[layer])  
        
        transformer_output[layer] = encoder_hidden_states




if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    device = torch_cuda_identify(print_info = True)
    torch._dynamo.config.cache_size_limit = 64
    
    is_4080_laptop = False
    is_4090 = False
    is_A100 = False
    gpu_name = torch.cuda.get_device_name()
    if "NVIDIA GeForce RTX 4080 Laptop GPU" in gpu_name:
        is_4080_laptop = True
    if "NVIDIA GeForce RTX 4090" in gpu_name:
        is_4090 = True
    if "NVIDIA A100-PCIE-40GB" in gpu_name:
        is_A100 = True
        
    # torch.set_printoptions(profile="default")
    # torch.set_printoptions(precision=3, sci_mode=False) 
    
    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 1)')
    parser.add_argument('--head_num', type=int, default=12, help='Number of heads (default: 12)')
    parser.add_argument('--head_size', type=int, default=64, help='Head size (default: 64)')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length (default: 256)')
    parser.add_argument('--mask_id', type=int, default=0, help='Mask type: 0-Casual | 1-Sliding | 2-Longformer | 3-BigBird (default: 0)')
    parser.add_argument('--block_m', type=int, default=32, help='Block Size of M (default:32)')
    parser.add_argument('--block_n', type=int, default=32, help='Block Size of N (default:32)')
    parser.add_argument('--num_warps', type=int, default=4, help='Warp Num to launch (default:4)')
    parser.add_argument('--layer_num', type=int, default=12, help='Sequence length (default: 1)')
    parser.add_argument('--model_selection', type=str, default="bert", help='Sequence length (default: 1)')
    args = parser.parse_args() 
    
    batch_size = args.batch_size
    head_num   = args.head_num
    head_size  = args.head_size
    seq_len    = args.seq_len
    mask_id    = args.mask_id
    BLOCK_M    = args.block_m
    BLOCK_N    = args.block_n
    num_warps  = args.num_warps
    layer_num  = args.layer_num
    hidden_dim = head_num * head_size 
    model_selection=args.model_selection
    data_type  = torch.float16
    dtype = "fp16"
    running_device = "cuda"
    attention_type="torch_attention"
    

    if(num_warps > (BLOCK_M/16) * (BLOCK_N/16)):
        print(f"num_warps: {num_warps}, (BLOCK_M/16) * (BLOCK_N/16): {int(BLOCK_M / 16) * int(BLOCK_N / 16)}")
        print("Error! Here should be: num_warps <= (BLOCK_M/16) * (BLOCK_N/16) !")
        exit(0)
    

    # for model_selection in ['bert_small','bert_base','bert_large','gpt','t5']:
    for model_selection in ['bert_base']:
        
        if model_selection == "bert_small":
            inference_model=bert_fwd_std
            head_num=8
            layer_num=6
        elif model_selection == "bert_base":
            inference_model=bert_fwd_std
            head_num=12
            layer_num=12
        elif model_selection == "bert_large":
            inference_model=bert_fwd_std   
            head_num=16
            layer_num=24     
        elif model_selection == "gpt":
            inference_model=gpt_base_fwd_std
            head_num=12
            layer_num=12
        elif model_selection == "t5":
            inference_model= T5_base_fwd_std
            head_num=12
            layer_num=12
        
        hidden_dim = head_num * head_size
        test_Torch           = True
        test_ByteTransformer = True
        test_Torch_Compile   = True
        test_STOF_end2end    = True
        
        if(is_4080_laptop):
            test_Torch_Compile   = False
        
        if seq_len > 1024:
            test_ByteTransformer = False
            print("Unsupport by ByteTransformer!")
        if seq_len==4096 and (batch_size==8 or batch_size==16):
            continue
        
        avg_seq_len = seq_len
        low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
        input_lens = torch.randint(low=low, high=high, size=(batch_size,))
        seqlen_mask = seqlen_to_mask(input_lens, seq_len)
        attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), "fp16")
        
        mask_mod = None
        score_mod = None
        mask=None

        if(mask_id == 0):
            is_causal = True
            mask_name = 'Causal_Mask'
            mask_mod = flex_causal_mask
            mask = generate_causal_mask(attr_mask).cuda()
            
        nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, part_block_mask, load_row_ptr, load_col_idx = get_sparse_storage(mask, BLOCK_M, BLOCK_N)

        print(f'[model] Unified bench test for {model_selection}') 
        

        input_from_tensor           = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
        qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        qkv_kernel                  = [set_dtype(torch.zeros(batch_size, seq_len, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        qkv_kernel_raw              = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        attr_output_kernel          = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        attr_output_bias            = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        attr_output_layernorm_gamma = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        attr_output_layernorm_beta  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        inter_kernel                = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        inter_bias                  = [set_dtype(torch.zeros(hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        output_kernel               = [set_dtype(torch.zeros(hidden_dim * 4, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        output_bias                 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        output_layernorm_gamma      = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        output_layernorm_beta       = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        
        output_from_tensor = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
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
        
    
        # PyTorch Naive -----------------------------------------
        if(test_Torch):
            attention_type="torch_attention"
            inference_model(mask)
            torch_output = transformer_output[-1]
        
        
        # Torch Compile -----------------------------------------
        if(test_Torch_Compile):
            torch._dynamo.reset()
            torch_compiled_bert_std = torch.compile(inference_model, mode='default')
            torch_compiled_bert_std(mask)
            torch_compiled_output = transformer_output[-1]
            max_diff, mean_diff = check_tensor(torch_compiled_output, torch_output)
            print(f"model {model_selection}    [CHECK] Torch Compile\t   max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )
        
        
        # ByteTransformer ---------------------------------------
        if(test_ByteTransformer):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, os.path.join(current_dir, "Bytetr_MCFuser"))
            from Bytetr_MCFuser.ops.package_op import bytetr_attn_op, bytetr_longattn_op
    
            attention_type="ByteTransformer"
            inference_model(mask)
            ByteTransformer_output=transformer_output[-1]
            max_diff, mean_diff = check_tensor(ByteTransformer_output, torch_output)
            print(f"model {model_selection}    [CHECK] ByteTransformer   max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )
            
            
        # STOF ----------------------------------------------------
        if(test_STOF_end2end):
            attention_type="STOF_attention"
            inference_model(mask)
            STOF_output=transformer_output[-1]
            max_diff, mean_diff = check_tensor(STOF_output, torch_output)
            print(f"model {model_selection}    [CHECK] STOF              max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )        



