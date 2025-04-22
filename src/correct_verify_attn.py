# python correct_verify_attn.py --batch_size 1 --head_num 1 --head_size 64 --seq_len 256

import sys
import os
import argparse
import torch
import math
from ops.package_op import block_attn_mask_op                 # Our kernel
from ops.package_op import rowwise_attn_full_op, rowwise_attn_sliding_op, rowwise_attn_mask_op
from torch.nn.attention import SDPBackend, sdpa_kernel        # FlashAttn2
# import ct_mcfuser_mask                                        # MCFuser
from torch.nn.attention.flex_attention import flex_attention  # FlexAttn

from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, transpose_for_scores
from util.masks import generate_causal_mask, get_sparse_storage
from util.masks import create_block_mask_cached, flex_causal_mask


def torch_attn_std(q, k, v, mask=None):
    # Q(B, H, S, W) @ K^T(B, H, W, S) = (B, H, S, S)
    kt = k.transpose(-2, -1)
    scores = torch.matmul(q, kt)
    scores /= (q.shape[-1] ** .5)
    
    if mask != None:
        scores -= 10000.0 * (1.0 - mask.unsqueeze(1))
        
    probs = torch.nn.functional.softmax(scores, dim=-1)
    h = torch.matmul(probs, v)
    return h


def check_tensor(other_output, torch_output):
    max_diff = (other_output - torch_output).abs().max().item()
    mean_diff = (other_output - torch_output).abs().mean().item()
    return max_diff, mean_diff


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
   
    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--head_num', type=int, default=12, help='Number of heads (default: 12)')
    parser.add_argument('--head_size', type=int, default=64, help='Head size (default: 64)')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length (default: 256)')
    parser.add_argument('--mask_id', type=int, default=0, help='Mask type: 0-Casual (default: 0)')
    parser.add_argument('--block_m', type=int, default=32, help='Block Size of M (default:32)')
    parser.add_argument('--block_n', type=int, default=32, help='Block Size of N (default:32)')
    parser.add_argument('--num_warps', type=int, default=1, help='Warp Num to launch (default:1)')
    args = parser.parse_args() 
    
    batch_size = args.batch_size
    head_num   = args.head_num
    head_size  = args.head_size
    seq_len    = args.seq_len
    mask_id    = args.mask_id
    BLOCK_M    = args.block_m
    BLOCK_N    = args.block_n
    num_warps  = args.num_warps
    
    if(num_warps > (BLOCK_M/16) * (BLOCK_N/16)):
        print(f"num_warps: {num_warps}, (BLOCK_M/16) * (BLOCK_N/16): {int(BLOCK_M / 16) * int(BLOCK_N / 16)}")
        print("Error! Here should be: num_warps <= (BLOCK_M/16) * (BLOCK_N/16) !")
        exit(0)
    
    data_type  = torch.float16
    running_device = "cuda"
    hidden_dim=head_num*head_size
    dtype="fp16"
    
    test_FlexAttn  = True
    test_FlashAttn = True
    test_Torch     = True
    test_ByteTrans = True
    
    if(is_4080_laptop):
        test_FlexAttn = False
    
    if(test_ByteTrans):
        hidden_dim=head_num*head_size
        dtype="fp16"
        hidden_states  = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
        qkv          = set_dtype(torch.zeros(batch_size, seq_len, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) 
        qkv_bias       = set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) 
        qkv2 = qkv  + qkv_bias
        query_byte, key_byte, value_byte = qkv2.chunk(3, dim=-1)
        query_byte = transpose_for_scores(query_byte, head_num, head_size)
        key_byte = transpose_for_scores(key_byte, head_num, head_size)
        value_byte = transpose_for_scores(value_byte, head_num, head_size)
    
    query = torch.randn(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)
    key = torch.randn(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)
    value = torch.randn(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)
    
    
    avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), "fp16")
    
    mask_mod = None
    score_mod = None
    
    if(mask_id == 0):
        is_causal = True
        mask_name = 'Causal_Mask'
        mask_mod = flex_causal_mask
        mask = generate_causal_mask(attr_mask).cuda()
        
    
    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, part_block_mask, load_row_ptr, load_col_idx = get_sparse_storage(mask, BLOCK_M, BLOCK_N)
    
    print(f"Correctness Verification for Attention with Mask ({mask_name}) ... ...")
    
    # PyTorch Naive  ---------------------------------------
    torch_output = torch_attn_std(query, key, value, mask)
    
    # FlashAttn2   Note: Torch >= 2.2.0
    if(test_FlashAttn):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            FA2_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        max_diff, mean_diff = check_tensor(FA2_output, torch_output)
        print(f"[CHECK]  FlashAttn2\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )
    
    
    # FlexAttn  ---------------------------------------
    if(test_FlexAttn):
        compiled_flex_attention = torch.compile(flex_attention, mode="default", dynamic=False)
        block_mask = create_block_mask_cached(mask_mod, 1, 1, seq_len, seq_len, device=query.device)
        flex_output = compiled_flex_attention(query, key, value, score_mod=score_mod, block_mask=block_mask)
        max_diff, mean_diff = check_tensor(flex_output, torch_output)
        print(f"[CHECK]  FlexAttn\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )
            
    # ByteTransformer --------------------------------------- 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, "Bytetr_MCFuser"))
    from Bytetr_MCFuser.ops.package_op import bytetr_attn_op, bytetr_longattn_op
    
    torch_output_byte = torch_attn_std(query_byte, key_byte, value_byte, mask)
    
    if seq_len<=256:
        mask=mask.half()
        ByteTransformer_output = bytetr_attn_op(qkv,qkv_bias,mask,head_num)
        ByteTransformer_output_4d =  ByteTransformer_output.view(batch_size, seq_len, head_num, head_size).permute(0, 2, 1, 3).contiguous()
        max_diff, mean_diff = check_tensor(ByteTransformer_output_4d,torch_output_byte)
        print(f"[CHECK]  ByteTransformer  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}")
    elif seq_len<=1024:
        mask=mask.half()
        ByteTransformer_output = bytetr_longattn_op(qkv,qkv_bias,mask,head_num)
        ByteTransformer_output_4d = ByteTransformer_output.view(batch_size, seq_len, head_num, head_size).permute(0, 2, 1, 3).contiguous()
        max_diff, mean_diff = check_tensor(ByteTransformer_output_4d,torch_output_byte)
        print(f"[CHECK]  ByteTransformer  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}")
    else:
        print("ByteTransformer unsurpported for seq_len > 1024 !")


    # STOF Row-Wise Kernel -----------------------------------------------
    STOF_rowwise_output = rowwise_attn_full_op(query, key, value, is_causal)
    
    h = torch_output.permute(0, 2, 1, 3).contiguous() 
    new_context_layer_shape = h.size()[:-2] + (query.shape[1]*query.shape[3], )
    hidden_states0 = h.view(new_context_layer_shape) 
    max_diff, mean_diff = check_tensor(STOF_rowwise_output, hidden_states0)
    print(f"[CHECK]  STOF Row-Wise\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )
    

    # STOF Block-wise Kernel -----------------------------------------------
    query1 = query.clone()
    STOF_blockwise_output = block_attn_mask_op(query1, key, value,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, part_block_mask, 
                                load_row_ptr, load_col_idx,
                                BLOCK_M, BLOCK_N, num_warps)
    max_diff, mean_diff = check_tensor(STOF_blockwise_output, torch_output)
    print(f"[CHECK]  STOF Blk-Wise\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )
    
        
   

