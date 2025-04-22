import torch
from torch.autograd import Function
import rowwise_attn_sliding, rowwise_attn_full, rowwise_attn_mask
import block_attn_mask


__all__ = ['rowwise_attn_full_op', 'rowwise_attn_sliding_op', 'rowwise_attn_mask_op', 'block_attn_mask_op']


# result.shape = {batch_size, seq_len, head_num * head_size}
class RowWise_Attn_Full(Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):  
        batch_size = q.size(0)
        hidden_dim = q.size(1) * q.size(-1)
        seq_len = q.size(-2)
        result = torch.zeros((batch_size, seq_len, hidden_dim), device=q.device, dtype=q.dtype)
        
        rowwise_attn_full.forward(q.contiguous(), k.contiguous(), v.contiguous(), is_causal, result)
        ctx.mark_non_differentiable(result)
        
        return result

class RowWise_Attn_Sliding(Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal, bandwidth):
        batch_size = q.size(0)
        hidden_dim = q.size(1) * q.size(-1)
        seq_len = q.size(-2)
        result = torch.zeros((batch_size, seq_len, hidden_dim), device=q.device, dtype=q.dtype)
        
        rowwise_attn_sliding.forward(q.contiguous(), k.contiguous(), v.contiguous(), is_causal, result, bandwidth)
        ctx.mark_non_differentiable(result)
        
        return result
    
class RowWise_Attn_Mask(Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal, row_mask):
        batch_size = q.size(0)
        hidden_dim = q.size(1) * q.size(-1)
        seq_len = q.size(-2)
        result = torch.zeros((batch_size, seq_len, hidden_dim), device=q.device, dtype=q.dtype)
        
        rowwise_attn_mask.forward(q.contiguous(), k.contiguous(), v.contiguous(), is_causal, result, row_mask.contiguous())
        ctx.mark_non_differentiable(result)
        
        return result
        
class Block_Attn_Mask(Function):
    @staticmethod
    def forward(ctx, q, k, v, 
                full_row_ptr, full_col_idx, 
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps):  
        
        block_attn_mask.forward(
            q.contiguous(), 
            k.contiguous(), 
            v.contiguous(),
            full_row_ptr.contiguous(), full_col_idx.contiguous(), 
            part_row_ptr.contiguous(), part_col_idx.contiguous(), part_block_mask.contiguous(), 
            load_row_ptr.contiguous(), load_col_idx.contiguous(),
            BLOCK_M, BLOCK_N, num_warps)

        ctx.mark_non_differentiable(q)
        return q

rowwise_attn_full_op = RowWise_Attn_Full.apply
rowwise_attn_sliding_op = RowWise_Attn_Sliding.apply
rowwise_attn_mask_op = RowWise_Attn_Mask.apply

block_attn_mask_op =  Block_Attn_Mask.apply