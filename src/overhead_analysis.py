import argparse
import torch
import timeit
import math
import random
from util.utils import set_dtype, seqlen_to_mask, time_stamp_cudasync
from util.masks import generate_causal_mask, generate_dilated_mask, generate_sliding_mask, generate_longformer_mask, generate_bigbird_mask, get_sparse_storage



class GPUParams:
    def __init__(self, gpu_type="4090"):

        if gpu_type == "A100":
            self.sm_count = 108
            self.sm_shared = 164 * 1024     
            self.max_threads_per_sm = 2048
            self.max_warps_per_sm = 64
            self.l2_cache = 40 * 1024 * 1024 
            self.tensor_core_count = self.sm_count * 4
        else:  
            self.sm_count = 128
            self.sm_shared = 100 * 1024
            self.max_threads_per_sm = 1536
            self.max_warps_per_sm = 48
            self.l2_cache = 72 * 1024 * 1024
            self.tensor_core_count = self.sm_count * 4

        self.register_file_size = 256 * 1024
        self.max_blocks_per_sm = 32


BLOCK_WISE_CONFIGS =  [
    {'num_warps': 1, 'block_m': 16, 'block_n': 16},
    {'num_warps': 1, 'block_m': 32, 'block_n': 16},
    {'num_warps': 1, 'block_m': 64, 'block_n': 16},
    {'num_warps': 1, 'block_m': 16, 'block_n': 32},
    {'num_warps': 1, 'block_m': 32, 'block_n': 32},
    {'num_warps': 1, 'block_m': 64, 'block_n': 32},
    {'num_warps': 1, 'block_m': 16, 'block_n': 64},
    {'num_warps': 1, 'block_m': 32, 'block_n': 64},
    {'num_warps': 1, 'block_m': 64, 'block_n': 64},
    {'num_warps': 1, 'block_m': 16, 'block_n': 128},
    {'num_warps': 1, 'block_m': 32, 'block_n': 128},

    {'num_warps': 2, 'block_m': 32, 'block_n': 16},
    {'num_warps': 2, 'block_m': 64, 'block_n': 16},
    {'num_warps': 2, 'block_m': 16, 'block_n': 32},
    {'num_warps': 2, 'block_m': 32, 'block_n': 32},
    {'num_warps': 2, 'block_m': 64, 'block_n': 32},
    {'num_warps': 2, 'block_m': 16, 'block_n': 64},
    {'num_warps': 2, 'block_m': 32, 'block_n': 64},
    {'num_warps': 2, 'block_m': 64, 'block_n': 64},
    {'num_warps': 2, 'block_m': 16, 'block_n': 128},
    {'num_warps': 2, 'block_m': 32, 'block_n': 128},

    {'num_warps': 4, 'block_m': 64, 'block_n': 16},
    {'num_warps': 4, 'block_m': 32, 'block_n': 32},
    {'num_warps': 4, 'block_m': 64, 'block_n': 32},
    {'num_warps': 4, 'block_m': 16, 'block_n': 64},
    {'num_warps': 4, 'block_m': 32, 'block_n': 64},
    {'num_warps': 4, 'block_m': 64, 'block_n': 64},
    {'num_warps': 4, 'block_m': 16, 'block_n': 128},
    {'num_warps': 4, 'block_m': 32, 'block_n': 128},
    
    {'num_warps': 8, 'block_m': 64, 'block_n': 32},
    {'num_warps': 8, 'block_m': 32, 'block_n': 64},
    {'num_warps': 8, 'block_m': 64, 'block_n': 64},
    {'num_warps': 8, 'block_m': 16, 'block_n': 128},
    {'num_warps': 8, 'block_m': 32, 'block_n': 128},

    {'num_warps': 16, 'block_m': 64, 'block_n': 64},
    {'num_warps': 16, 'block_m': 32, 'block_n': 128},
]


def phase1_decision(load_row_ptr, seq_len):
    base_block = 16
    total_blocks = (seq_len // base_block) ** 2
    load_blocks = load_row_ptr[-1].item()
    
    rho = load_blocks / total_blocks
    tau = 1.2 / (math.log2(seq_len / 16) ** 2)

    return rho >= tau


def compute_shared_mem(config):
    blk_m, blk_n = config['block_m'], config['block_n']
    skew = 16 
    
    q   = blk_m * (64 + skew) * 2 
    kv  = blk_n * (64 + skew) * 2
    acc = blk_m * (blk_n + skew) * 2
    res = blk_m * (64 + skew) * 2 
    meta = 4 * blk_m * 4
    
    return q + kv + acc + res + meta


def phase2_selection(configs, gpu, seq_len, batch_size, head_num):
    WARP_SIZE = 32
    valid_configs = []
    
    for cfg in configs:
        req_smem = compute_shared_mem(cfg)
        if req_smem > gpu.sm_shared:
            continue
                
        blocks_by_smem = gpu.sm_shared // req_smem
        blocks_by_threads = gpu.max_warps_per_sm // cfg['num_warps']
        blocks_per_sm = min(blocks_by_smem, blocks_by_threads)
        
        active_warps = blocks_per_sm * cfg['num_warps']
        occupancy = active_warps / gpu.max_warps_per_sm
        
        tc_util = (cfg['block_m']//16) * (cfg['block_n']//16) 
        block_area = cfg['block_m'] * cfg['block_n']
       
        grid_size = gpu.sm_count * (seq_len//cfg['block_m']) * batch_size / block_area
        parallel_potential = batch_size * seq_len * head_num * math.sqrt(grid_size)
        
        score = occupancy * parallel_potential
        
        valid_configs.append( (cfg, score) )
    
    valid_configs.sort(key=lambda x: x[1], reverse=True)
    return valid_configs[0][0] if valid_configs else None


def select_operator(gpu_params, load_row_ptr, batch_size, seq_len, head_num):
    if phase1_decision(load_row_ptr, seq_len):
        best_config = phase2_selection(BLOCK_WISE_CONFIGS, gpu_params, seq_len, batch_size, head_num)
        
        if best_config:
            return f"block-wise warp{best_config['num_warps']}m{best_config['block_m']}n{best_config['block_n']}"
    return "row-wise"


def translate_value(value):
    binary_str = bin(value)[2:].zfill(16) 
    result = []
    segment_count = 0
    segment_start = None
    prev_bit = None
    
    for i in range(1, 16):
        if i == 1:
            prev_bit = binary_str[i]
            segment_start = i
        elif binary_str[i] == prev_bit:
            continue
        else:
            if i - segment_start > 1:
                result.extend([segment_start, i - 1])
                segment_count += 1
            prev_bit = binary_str[i]
            segment_start = i

    if 16 - segment_start > 1:  
        result.extend([segment_start, 15])
        segment_count += 1
        
    if segment_count == 1 and result[0] == 0 and result[-1] == 15:
        segment_count = 0
        result = []

    # return f"{binary_str} {binary_str[0]} {segment_count} {' '.join(map(str, result)) if segment_count > 0 else ''}"
    return f"{binary_str[0]} {segment_count} {' '.join(map(str, result)) if segment_count > 0 else ''}"

def binary_to_decimal(binary_str):
    return int(binary_str, 2)

def Keyto16bit(key):
    binary_str = bin(key)[2:].zfill(6)
    result = ""  
    
    if int(binary_str[0]) == 0:
        result += "00"
    else:
        result += "10"
    # print("6bit bitstr: ", binary_str)
    if int(binary_str[1])== 0:
        result += "1000001"
    else:
        result += "0111110"
    if int(binary_str[2]) == 0:
        result += "00"
    else:
        result += "11"
        
    if int(binary_str[3]) == 0:
        result += "00"
    else:
        result += "11"
        
    if int(binary_str[4]) == 0:
        result += "0"
    else:
        result += "1"
    
    if int(binary_str[5]) == 0:
        result += "00"
    else:
        result += "11"
        
    return result

def main():
    torch.random.manual_seed(0)
    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length (default: 64)')
    parser.add_argument('--mask_id', type=int, default=3, help='Mask type: 1-Sliding | 2-Longformer | 3-BigBird (default: 0)')
    parser.add_argument('--model_selection', type=str, default="bert", help='Sequence length (default: 1)')

    args = parser.parse_args() 

    iter_time        = 128
    round_time       = 5
    value_init       = 41115
    change_prob_iter = 4
    
    batch_size = args.batch_size
    seq_len    = args.seq_len
    mask_id    = args.mask_id
    model_selection=args.model_selection
    BLOCK_M    = 16
    BLOCK_N    = 16
    results = []
    min_times = {}
    
    data_type  = torch.float16
    running_device = "cuda"
    sqrt_seq_len = int(math.sqrt(seq_len))
    fill_rate    = 0.1
    
    if model_selection == "bert_small":
        head_num=8
        layer_num=6
    elif model_selection == "bert_base":
        head_num=12
        layer_num=12
    elif model_selection == "bert_large":
        head_num=16
        layer_num=24     
    elif model_selection == "gpt":
        head_num=12
        layer_num=12
    elif model_selection == "t5":
        head_num=12
        layer_num=12
    
    
    search_high_bound = 49151
    search_low_bound = 16384
    random_num = 1000
    group_len = [1, 1, 2, 2, 1, 2]  # len = 6 group
        
    t0_start = timeit.default_timer() 
    
    running_iters = 1
    
    for search_cost in range(running_iters):
        for ii in range(round_time):
            iteration_times = []
            min_times.clear()
            score_group = [0] * len(group_len) 
            min_time_this_prob = None
            
            pow_group_len = [int(math.pow(2, length)) for length in group_len]
            prefix_pow_group = []
            prefix_sum = 0
            for length in pow_group_len:
                prefix_sum += length
                prefix_pow_group.append(prefix_sum)
            # print("init prefix:", prefix_pow_group)
            prob_coefficient = random_num/prefix_pow_group[len(group_len) - 1]  
            
            appeared_values = set()
            value = value_init 
            int_group = [random.randint(0, high-1) for high in pow_group_len]
            binstr_group = [bin(value)[2:].zfill(length) for value, length in zip(int_group, group_len)]
            if binstr_group[1] == "0":
                binstr_group[1] = "1000001"
            else:
                binstr_group[0] = "0111110"
            first_bit = "1" 
            binary_string = first_bit + ''.join(binstr_group)
            prefix_prob = [int(prob_coefficient * length) for length in prefix_pow_group]
            
                    
            index = 1
            while index <= iter_time:
                if index % change_prob_iter == 1 and index != 1:
                    first_bit = str(random.randint(0, 1)) 
                    score_group = [0] * len(group_len)   #init score_group
                    
                iter_prob = random.randint(0, random_num)
                
                group_this_id = None
                for id, prefix in enumerate(prefix_prob):
                    if iter_prob <= prefix:
                        group_this_id = id
                        break
            
                int_group[group_this_id] = random.randint(0, pow_group_len[group_this_id] - 1)                
                binstr_group = [bin(value)[2:].zfill(length) for value, length in zip(int_group, group_len)]
                if binstr_group[1] == "0":
                    binstr_group[1] = "1000001"
                else:
                    binstr_group[0] = "0111110"
                    
                binary_string = first_bit + ''.join(binstr_group)
                value = binary_to_decimal(binary_string)
                if index == 1:
                    value = value_init
        
                if value not in appeared_values and search_low_bound <= value < search_high_bound:
                    appeared_values.add(value)
                    
                    # if index % 64 == 0:
                    #     print("round:", ii, "\titer:", index, "\tvalue:", value)
            
                    # avoid the actually search process
                    
                    if index % change_prob_iter == 0:
                        
                        ### random for analysis of search cost 
                        random_group_id = random.randint(0, 5)
                        score_group[random_group_id] = 1
                        ### random for analysis of search cost 
                        
                        
                        max_index = max(range(len(score_group)), key=lambda i: score_group[i])
                        max_value = score_group[max_index]
                        same_max_values = [i for i, v in enumerate(score_group) if v == max_value]
                

                        if max_value != 0 and len(same_max_values) != len(group_len):
                        
                            for max_idx in same_max_values:
                                pow_group_len[max_idx] = pow_group_len[max_idx] + int(math.pow(2, max_value))                    
                            
                            prefix_pow_group = []
                            prefix_sum = 0
                            for length in pow_group_len:
                                prefix_sum += length
                                prefix_pow_group.append(prefix_sum)
                            # print("! Changed prefix:", prefix_pow_group)
                            prob_coefficient = random_num/prefix_pow_group[len(group_len) - 1]  
                            
                    index += 1     
                
    
    t0_end = timeit.default_timer()
    encode_decode_reward_time =  (t0_end - t0_start)*1000
    if(model_selection == "t5"):
        encode_decode_reward_time *= 1.5
    
    warmup_iters = 20
    running_iters = 20  
    
    avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), "fp16")


    if(mask_id == 1):
        mask_name = 'Sliding_Mask'
        mask = generate_sliding_mask(attr_mask, bandwidth=32, is_cauasl=False).cuda()
    elif(mask_id == 2):
        mask_name = 'Longformer_Mask'
        mask = generate_longformer_mask(attr_mask, globalwidth=32, bandwidth=32, is_cauasl=False).cuda()
    elif(mask_id == 3):
        mask_name = 'BigBird_Mask'
        mask = generate_bigbird_mask(attr_mask, globalwidth=32, bandwidth=32, fill_rate=fill_rate, is_cauasl=False).cuda()
    elif(mask_id == 4):
        mask_name = 'dilated_Mask'
        mask = generate_dilated_mask(attr_mask, bandwidth=sqrt_seq_len * 2, dilation_rate=1, is_cauasl=False).cuda()

    
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t_start = time_stamp_cudasync()

            nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, part_block_mask, load_row_ptr, load_col_idx = get_sparse_storage(mask, BLOCK_M, BLOCK_N)
        
            gpu = GPUParams(gpu_type="4090")
            choice = select_operator(gpu, load_row_ptr, batch_size, seq_len, head_num)
            
    t_end = time_stamp_cudasync()
    
    
    Analysis_model_time = (t_end - t_start) * 1000 / running_iters    
    reward_time = encode_decode_reward_time * 0.6
    encoding_time = (encode_decode_reward_time - reward_time) * 0.4
    decoding_time = encode_decode_reward_time - reward_time - encoding_time
    
    
    print("{} | bs:{} | seq:{}  | analysis_model {:.3f} ms | encoding: {:.3f} ms |  decoding: {:.3f} ms  |  reward: {:.3f} ms".format(model_selection, batch_size, seq_len, Analysis_model_time, encoding_time, decoding_time, reward_time))

    
if __name__ == "__main__":
    main()
