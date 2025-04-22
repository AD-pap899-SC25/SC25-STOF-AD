import torch
from torch.autograd import Function
import bytetr_attn
import bytetr_longattn


__all__ = ['bytetr_attn_op','bytetr_longattn_op']

class Byte_Attn(Function):
    @staticmethod
    def forward(ctx, 
                qkv,qkv_bias_ptr,mask,
                num_head):  
        batch_size = qkv.size(0)
        head_num =num_head
        seq_len = qkv.size(1)
        head_size = qkv.size(2)//3//num_head
        hidden_dim =  qkv.size(2)//3
        attention_output = torch.zeros((batch_size, seq_len, hidden_dim), device=qkv.device, dtype=torch.float16)

        bytetr_attn.forward(
            qkv.contiguous(),qkv_bias_ptr.contiguous(),
            mask.contiguous(),attention_output,
            head_num)
            
        ctx.mark_non_differentiable(attention_output)
        return attention_output


class Byte_Longattn(Function):
    @staticmethod
    def forward(ctx, 
                qkv,qkv_bias_ptr,mask,
                num_head):  
        batch_size = qkv.size(0)
        head_num =num_head
        seq_len = qkv.size(1)
        head_size = qkv.size(2)//3//num_head
        # hidden_dim =  qkv.size(2)//3
        attention_output = torch.zeros((batch_size, seq_len, head_num * head_size), device=qkv.device, dtype=torch.float16)

        bytetr_longattn.forward(
            qkv.contiguous(),qkv_bias_ptr.contiguous(),
            mask.contiguous(),attention_output,
            head_num)
            
        ctx.mark_non_differentiable(attention_output)
        return attention_output

bytetr_attn_op=Byte_Attn.apply
bytetr_longattn_op=Byte_Longattn.apply