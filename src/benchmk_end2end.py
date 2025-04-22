
import sys
import os

import argparse
import torch
import torch._dynamo
import math
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel
from ops.package_op import block_attn_mask_op
from ops.package_op import rowwise_attn_sliding_op
from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, time_stamp_cudasync, transpose_for_scores
from util.masks import generate_bigbird_mask, get_sparse_storage
from util.masks import flex_bigbird_mask
from triton_template.gemm_add_layernorm import triton_matmul_bias_layernorm 
from triton_template.gemm_gemm import triton_matmul_batch


import warnings
from torch.jit import TracerWarning
warnings.filterwarnings("ignore", category=TracerWarning)

import csv
from typing import Dict, Any

def load_config_table(filepath: str) -> list:
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        return [row for row in reader]

def get_replacement_flags(config: Dict[str, Any], csv_file: str = "../data/triton_perf.csv") -> tuple:

    supported_hardware = ['A100', '4090', '4080_laptop']
    table = load_config_table(csv_file)
    
    hardware = next((k for k in supported_hardware if config.get(f"is_{k}", False)), None)
    if not hardware:
        raise ValueError("Unsupported hardware type")
    
    query = {
        'hardware': hardware,
        'batch_size': str(config['batch_size']),
        'seq_length': str(config['seq_len']),
        'hid_dim': str(config['hidden_dim'])
    }
    
    for row in table:
        if all(row[k] == v for k, v in query.items()):
            return (
                row['GEMM+Layernorm(large)'] == '1',
                row['GEMM+Layernorm(small)'] == '1',
                row['GEMM+GEMM'] == '1'
            )
    
    raise LookupError(f"No matching config found for {query}")


def new_gelu(input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


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
                
                
            elif (attention_type == 'Compile_attention'):
                scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
                # scores -= 10000.0 * (1.0 - mask.unsqueeze(1))
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
                    hidden_states = rowwise_attn_sliding_op(query, key, value, is_causal, int(sqrt_seq_len/8))
                else:
                    query1 = query.clone()
                    h = block_attn_mask_op(query1, key, value,
                                    full_row_ptr, full_col_idx, 
                                    part_row_ptr, part_col_idx, part_block_mask,
                                    load_row_ptr, load_col_idx,
                                    BLOCK_M, BLOCK_N, num_warps)
                    h = h.permute(0, 2, 1, 3).contiguous()
                    new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                    hidden_states = h.view(new_context_layer_shape)

                    
            # ------------------------------------------------------------ Attention End
            #attention output projection GEMM 
            if(replace_gemm_layernorm_small == True and head_num != 12):
                attr_output_kernel_temp = attr_output_kernel[layer].unsqueeze(0).expand(batch_size, -1, -1)
                hidden_states = triton_matmul_bias_layernorm(hidden_states, attr_output_kernel_temp, attr_output_bias[layer] + input_tensor)
                hidden_states += attr_output_layernorm_beta[layer]
            else:
                hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
                hidden_states = hidden_states + input_tensor
                hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                            weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
            
            residual = hidden_states       
            #FFN GEMM 1 + add bias 
            hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer] 
            hidden_states = F.gelu(hidden_states) 
            
            #FFN GEMM 2 + add bias
            if(replace_gemm_layernorm_large == True and head_num != 12):
                output_kernel_temp = output_kernel[layer].unsqueeze(0).expand(batch_size, -1, -1)
                hidden_states = triton_matmul_bias_layernorm(hidden_states, output_kernel_temp, output_bias[layer] + residual)
                hidden_states += output_layernorm_beta[layer]
            else:
                hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]  
                hidden_states = hidden_states + residual
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
        
        elif (attention_type == 'Compile_attention'):
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
                hidden_states = rowwise_attn_sliding_op(query, key, value, is_causal, int(sqrt_seq_len/8))
            else:
                query1 = query.clone()
                h = block_attn_mask_op(query1, key, value,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, part_block_mask,
                                load_row_ptr, load_col_idx,
                                BLOCK_M, BLOCK_N, num_warps)
                h = h.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                hidden_states = h.view(new_context_layer_shape)
                
        # ------------------------------------------------------------ Attention End
                
        if(replace_gemm_layernorm_small == True and head_num != 12):
                attr_output_kernel_temp = attr_output_kernel[layer].unsqueeze(0).expand(batch_size, -1, -1)
                hidden_states = triton_matmul_bias_layernorm(hidden_states, attr_output_kernel_temp, attr_output_bias[layer] + input_tensor)
                hidden_states += attr_output_layernorm_beta[layer]
        else:
            hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
            hidden_states = hidden_states + input_tensor
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                        weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
            
        
        residual = hidden_states 
        
        #FFN GEMM 1 + add bias 
        hidden_states = torch.matmul(hidden_states, inter_kernel[layer])
        hidden_states = hidden_states + inter_bias[layer] 
        
        #new_gelu
        hidden_states = new_gelu(hidden_states)  
        
        
        #FFN GEMM 2 + add bias
        if(replace_gemm_layernorm_large == True and head_num != 12):
            output_kernel_temp = output_kernel[layer].unsqueeze(0).expand(batch_size, -1, -1)
            hidden_states = triton_matmul_bias_layernorm(hidden_states, output_kernel_temp, output_bias[layer] + residual)
            hidden_states += output_layernorm_beta[layer]
        else:
            hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]  
            hidden_states = hidden_states + residual
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),  
                                            weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
        
        transformer_output[layer] = hidden_states


# T5 ------------------------------------------------------------------------
# ----------------------------------------------------------------------------
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
            
        elif (attention_type == 'Compile_attention'):
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
                encoder_hidden_states = rowwise_attn_sliding_op(query, key, value, is_causal, int(sqrt_seq_len/8))
            else:
                query1 = query.clone()
                h = block_attn_mask_op(query1, key, value,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, part_block_mask,
                                load_row_ptr, load_col_idx,
                                BLOCK_M, BLOCK_N, num_warps)
                h = h.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                encoder_hidden_states = h.view(new_context_layer_shape)
          
        # ------------------------------------------------------------ Attention End
        
        
        
        if(replace_gemm_layernorm_small == True and head_num != 12):
            attr_output_kernel_temp = attr_output_kernel[layer].unsqueeze(0).expand(batch_size, -1, -1)
            hidden_states = triton_matmul_bias_layernorm(encoder_hidden_states, attr_output_kernel_temp, attr_output_bias[layer] + input_tensor)
            hidden_states += attr_output_layernorm_beta[layer]
        else:
            encoder_hidden_states = torch.matmul(encoder_hidden_states, attr_output_kernel[layer])
            encoder_hidden_states = encoder_hidden_states + attr_output_bias[layer]
            encoder_hidden_states = encoder_hidden_states + input_tensor  
            encoder_hidden_states = F.layer_norm(encoder_hidden_states, (hidden_dim, ), weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
            
        
        
        encoder_residual = encoder_hidden_states
        #FFN GEMM 1 + add bias 
        encoder_hidden_states = torch.matmul(encoder_hidden_states, inter_kernel[layer])
        encoder_hidden_states = encoder_hidden_states + inter_bias[layer] 
        # T5 relu
        encoder_hidden_states = F.relu(encoder_hidden_states)  
        
        
        if(replace_gemm_layernorm_large == True and head_num != 12):
            output_kernel_temp = output_kernel[layer].unsqueeze(0).expand(batch_size, -1, -1)
            hidden_states = triton_matmul_bias_layernorm(encoder_hidden_states, output_kernel_temp, output_bias[layer] + encoder_residual)
            hidden_states += output_layernorm_beta[layer]
        else:
            encoder_hidden_states = torch.matmul(encoder_hidden_states, output_kernel[layer]) 
            encoder_hidden_states = encoder_hidden_states + output_bias[layer]  
            encoder_hidden_states = encoder_hidden_states + encoder_residual
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
            
        elif (attention_type == 'Compile_attention'):
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
                decoder_hidden_states = rowwise_attn_sliding_op(query, key, value, is_causal, int(sqrt_seq_len/8))
            elif (seq_len == 4096):
                query1 = query.clone()
                h = block_attn_mask_op(query1, key, value,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, part_block_mask,
                                load_row_ptr, load_col_idx,
                                BLOCK_M, BLOCK_N, num_warps)
                h = h.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
                decoder_hidden_states = h.view(new_context_layer_shape)
            else:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    FA2_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
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
            
        
        if(replace_gemm_layernorm_small == True and head_num != 12):
            attr_output_kernel_temp = crattr_output_kernel_2[layer].unsqueeze(0).expand(batch_size, -1, -1)
            hidden_states = triton_matmul_bias_layernorm(decoder_hidden_states, attr_output_kernel_temp, decoder_hidden_states + crattr_output_bias_2[layer])
            hidden_states += crattn_lynorm_beta_2[layer]
        else:
            decoder_hidden_states = torch.matmul(decoder_hidden_states, crattr_output_kernel_2[layer])
            decoder_hidden_states = decoder_hidden_states + crattr_output_bias_2[layer]
            decoder_hidden_states = decoder_hidden_states + decoder_residual 
            decoder_hidden_states = F.layer_norm(decoder_hidden_states, (hidden_dim, ), weight=crattn_lynorm_gamma_2[layer], bias=crattn_lynorm_beta_2[layer])
        
        decoder_residual = decoder_hidden_states  
        
        # FFN GEMM 1 + add bias 
        decoder_hidden_states = torch.matmul(decoder_hidden_states, inter_kernel_2[layer])
        decoder_hidden_states = decoder_hidden_states + inter_bias_2[layer] 
        # T5 relu
        decoder_hidden_states = F.relu(decoder_hidden_states)  
        
        
        #FFN GEMM 2 + add bias
        if(replace_gemm_layernorm_large == True and head_num != 12):
            output_kernel_temp = output_kernel_2[layer].unsqueeze(0).expand(batch_size, -1, -1)
            hidden_states = triton_matmul_bias_layernorm(hidden_states, output_kernel_temp, output_bias_2[layer] + decoder_residual)
            hidden_states += lynorm_beta_2[layer]
        else:
            decoder_hidden_states = torch.matmul(decoder_hidden_states, output_kernel_2[layer]) 
            decoder_hidden_states = decoder_hidden_states + output_bias_2[layer]  
            decoder_hidden_states = decoder_hidden_states + decoder_residual
            encoder_hidden_states = F.layer_norm(decoder_hidden_states, (hidden_dim, ), weight=lynorm_gamma_2[layer], bias=lynorm_beta_2[layer])  
        
        transformer_output[layer] = encoder_hidden_states



if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    device = torch_cuda_identify(print_info = False)
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
    
    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--mask_id', type=int, default=3, help='Mask type: 1-Sliding | 2-Longformer | 3-BigBird (default: 0)')
    parser.add_argument('--block_m', type=int, default=16, help='Block Size of M (default:32)')
    parser.add_argument('--block_n', type=int, default=16, help='Block Size of N (default:32)')
    parser.add_argument('--num_warps', type=int, default=1, help='Warp Num to launch (default:4)')
    
    parser.add_argument('--method', type=str, default="TorchNative", help='TorchNative, TorchCompile, ByteTrans, STOF')
    parser.add_argument('--model', type=str, default="bert", help='Sequence length (default: 1)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 1)')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length (default: 256)')
    args = parser.parse_args() 

    mask_id    = args.mask_id
    BLOCK_M    = args.block_m
    BLOCK_N    = args.block_n
    num_warps  = args.num_warps
    model_selection  = args.model
    method_selection = args.method
    
    head_size = 64
    seq_len   = args.seq_len
    batch_size = args.batch_size
    
    data_type  = torch.float16
    dtype = "fp16"
    running_device = "cuda"
    attention_type="torch_attention"
    
    
    if(num_warps > (BLOCK_M/16) * (BLOCK_N/16)):
        print(f"num_warps: {num_warps}, (BLOCK_M/16) * (BLOCK_N/16): {int(BLOCK_M / 16) * int(BLOCK_N / 16)}")
        print("Error! Here should be: num_warps <= (BLOCK_M/16) * (BLOCK_N/16) !")
        exit(0)

                
    data_type  = torch.float16
    running_device = "cuda"
    sqrt_seq_len = int(math.sqrt(seq_len))
    fill_rate    = 0.1
    
    warmup_iters = 10
    running_iters = 20        

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
    
    
    
    
    test_Torch           = False
    test_ByteTransformer = False
    test_Torch_Compile   = False
    test_STOF            = False
    test_STOF_Compile    = False
    
    
    if method_selection == "TorchNative":
        test_Torch           = True
    elif(method_selection == "TorchCompile"):
        test_Torch_Compile   = True
        if(is_4080_laptop): 
            test_Torch_Compile = False
    elif(method_selection == "ByteTrans"):
        test_ByteTransformer = True
        if seq_len > 1024:
            print("ByteTransformer unsupported for seq_len > 1024")
            test_ByteTransformer = False
    elif(method_selection == "STOF"):
        test_STOF    = True
        test_STOF_Compile = True
    

        

    avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), "fp16")
    mask_mod = None
    score_mod = None
    mask=None
    
    if(mask_id == 3):
        is_causal = False
        mask_name = 'BigBird_Mask'
        mask_mod = flex_bigbird_mask
        mask = generate_bigbird_mask(attr_mask, globalwidth=BLOCK_M, bandwidth=BLOCK_N, fill_rate=fill_rate, is_cauasl=is_causal).cuda()

    # print(f' [model] Unified bench test for {model_selection}')
    
    
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
    
    # for STOF 
    query = torch.randn(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)
    key = torch.randn(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)
    value = torch.randn(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)
    

    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, part_block_mask, load_row_ptr, load_col_idx, = get_sparse_storage(mask, BLOCK_M, BLOCK_N)
    
    replace_gemm_layernorm_large = False
    replace_gemm_layernorm_small = False
    replace_gemm_gemm = False
    
    # PyTorch Naive  ---------------------------------------
    if(test_Torch):
        attention_type="torch_attention"
        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:    
                t0_start = time_stamp_cudasync()

            inference_model(mask)
            torch_output = transformer_output[-1]
            
        t0_end = time_stamp_cudasync()
        base_time = (t0_end - t0_start) * 1000 / running_iters
        print("e2e {} | bs:{} | seq:{}  |  Torch Native    : {:.3f} ms / iter".format(model_selection, batch_size, seq_len, base_time)) 


    # PyTorch Compile  ---------------------------------------
    if(test_Torch_Compile):
        attention_type="torch_attention"
        # attention_type="Compile_attention"
        torch_compiled_model_std = torch.compile(inference_model, mode='default', backend='inductor')
        
        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:    
                t1_start = time_stamp_cudasync()
                
            torch_compiled_model_std(mask)
            torch_compiled_output = transformer_output[-1]
            
        t1_end = time_stamp_cudasync()
        torch_compiled_time = (t1_end - t1_start) * 1000 / running_iters   
        print("e2e {} | bs:{} | seq:{}  |  Torch Compile   : {:.3f} ms / iter".format(model_selection, batch_size, seq_len, torch_compiled_time)) 


    # ByteTransformer ------------------------------------
    if(test_ByteTransformer):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(current_dir, "Bytetr_MCFuser"))
        from Bytetr_MCFuser.ops.package_op import bytetr_attn_op, bytetr_longattn_op
                
        attention_type="ByteTransformer"
        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:    
                t2_start = time_stamp_cudasync()
                
            inference_model(mask)
            ByteTransformer_output = transformer_output[-1]
            
        t2_end = time_stamp_cudasync()
        byteattn_time = (t2_end - t2_start) * 1000 / running_iters
        print("e2e {} | bs:{} | seq:{}  |  ByteTransformer: {:.3f} ms / iter".format(model_selection, batch_size, seq_len, byteattn_time)) 


    #  STOF_attention ------------------------------------
    if(test_STOF):    
        attention_type="STOF_attention"
        
        if(hidden_dim != 768 and is_4080_laptop == False):
            config = {
                "is_4080_laptop": is_4080_laptop, 
                "is_4090": is_4090,
                "is_A100": is_A100,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim
            }
            
            supported_hardware = ['A100', '4090', '4080_laptop']
            hardware = next((k for k in supported_hardware if config.get(f"is_{k}", False)), None)
            
            replace_gemm_layernorm_large, replace_gemm_layernorm_small, replace_gemm_gemm = get_replacement_flags(config)
        
        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:    
                t3_start = time_stamp_cudasync()
        
            inference_model(mask)
            STOF_output=transformer_output[-1]
            
        t3_end = time_stamp_cudasync()
        STOF_time1 = (t3_end - t3_start) * 1000 / running_iters
        
        
        STOF_compiled_model_std = torch.compile(inference_model, mode='default', backend='inductor')
        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:    
                t4_start = time_stamp_cudasync()
        
            STOF_compiled_model_std(mask)
            STOF_output=transformer_output[-1]
        
        t4_end = time_stamp_cudasync()
        STOF_compiled_time = (t4_end - t4_start) * 1000 / running_iters


        STOF_time = min(STOF_time1, STOF_compiled_time)
        print("e2e {} | bs:{} | seq:{}  |  STOF           : {:.3f} ms / iter".format(model_selection, batch_size, seq_len, STOF_time)) 
        
        
        
        
        
