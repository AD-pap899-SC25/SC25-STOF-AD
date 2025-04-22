import torch
import triton
import triton.language as tl
import time

@triton.jit
def main_kernel1(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    ax_4_0 = pid_x # ax_4_0 -> ['pid_x'] + __root
    offset_11 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_11 -> ['ax_4_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_5 = tl.arange(0, 128) * 1 # offset_5 -> ['__root'] + __root
    offset_6 = tl.arange(0, 128) * 1 # offset_6 -> ['__root'] + __root
    offset_8 = tl.arange(0, 128) * 1 # offset_8 -> ['__root'] + __root
    offset_9 = tl.arange(0, 128) * 1 # offset_9 -> ['__root'] + __root
    mask_1_data = tl.load(mask_1 + 0 * 196608 + offset * 16384 + offset_8[:,None] * 128 + offset_9[None,:] * 1) # mask_1_data -> ['offset', 'offset_8', 'offset_9'] + __root
    offset_12 = tl.arange(0, 128) * 1 # offset_12 -> ['__root'] + __root
    v_1_data = tl.load(v_1 + 0 * 98304 + offset * 8192 + offset_12[:,None] * 64 + offset_11[None,:] * 1) # v_1_data -> ['offset', 'offset_12', 'offset_11'] + __root
    offset_13 = tl.arange(0, 128) * 1 # offset_13 -> ['__root'] + __root
    offset_14 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_14 -> ['ax_4_0'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 16], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 1): # ax_5_0 -> ['__root'] + __root
        T_einsum_1_data = tl.zeros([128, 128], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_7 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_7 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + 0 * 98304 + offset * 8192 + offset_5[:,None] * 64 + offset_7[None,:] * 1) # q_1_data -> ['offset', 'offset_5', 'offset_7'] + ax_6_0
            k_1_data = tl.load(k_1 + 0 * 98304 + offset * 8192 + offset_6[:,None] * 64 + offset_7[None,:] * 1) # k_1_data -> ['offset', 'offset_6', 'offset_7'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['k_1_data', 'T_einsum_1_data', 'q_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['mask_1_data', 'T_einsum_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_red_1_data', 'compute_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['T_einsum_2_data', 'compute_1_data', 'v_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + 0 * 98304 + offset * 8192 + offset_13[:,None] * 64 + offset_14[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_13', 'offset_14'] + __root

@triton.jit
def main_kernel2(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_5 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_5 -> ['ax_3_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_6 = tl.arange(0, 256) * 1 # offset_6 -> ['__root'] + __root
    offset_8 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_8 -> ['ax_3_0'] + __root
    offset_9 = tl.arange(0, 256) * 1 # offset_9 -> ['__root'] + __root
    mask_1_data = tl.load(mask_1 + 0 * 786432 + offset * 65536 + offset_8[:,None] * 256 + offset_9[None,:] * 1) # mask_1_data -> ['offset', 'offset_8', 'offset_9'] + __root
    offset_12 = tl.arange(0, 256) * 1 # offset_12 -> ['__root'] + __root
    offset_11 = tl.arange(0, 64) * 1 # offset_11 -> ['__root'] + __root
    v_1_data = tl.load(v_1 + 0 * 196608 + offset * 16384 + offset_12[:,None] * 64 + offset_11[None,:] * 1) # v_1_data -> ['offset', 'offset_12', 'offset_11'] + __root
    offset_13 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_13 -> ['ax_3_0'] + __root
    offset_14 = tl.arange(0, 64) * 1 # offset_14 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([64, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([64, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 1): # ax_5_0 -> ['__root'] + __root
        T_einsum_1_data = tl.zeros([64, 256], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_7 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_7 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + 0 * 196608 + offset * 16384 + offset_5[:,None] * 64 + offset_7[None,:] * 1) # q_1_data -> ['offset', 'offset_5', 'offset_7'] + ax_6_0
            k_1_data = tl.load(k_1 + 0 * 196608 + offset * 16384 + offset_6[:,None] * 64 + offset_7[None,:] * 1) # k_1_data -> ['offset', 'offset_6', 'offset_7'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['T_einsum_1_data', 'q_1_data', 'k_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_red_1_data', 'compute_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['T_einsum_2_data', 'compute_1_data', 'v_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + 0 * 196608 + offset * 16384 + offset_13[:,None] * 64 + offset_14[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_13', 'offset_14'] + __root

@triton.jit
def main_kernel3(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_5 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_5 -> ['ax_3_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_8 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_8 -> ['ax_3_0'] + __root
    offset_11 = tl.arange(0, 64) * 1 # offset_11 -> ['__root'] + __root
    offset_13 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_13 -> ['ax_3_0'] + __root
    offset_14 = tl.arange(0, 64) * 1 # offset_14 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([64, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([64, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 2): # ax_5_0 -> ['__root'] + __root
        offset_6 = ax_5_0 * 256 + tl.arange(0, 256) * 1 # offset_6 -> ['ax_5_0'] + ax_5_0
        offset_9 = ax_5_0 * 256 + tl.arange(0, 256) * 1 # offset_9 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + 0 * 3145728 + offset * 262144 + offset_8[:,None] * 512 + offset_9[None,:] * 1) # mask_1_data -> ['offset', 'offset_8', 'offset_9'] + ax_5_0
        offset_12 = ax_5_0 * 256 + tl.arange(0, 256) * 1 # offset_12 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + 0 * 393216 + offset * 32768 + offset_12[:,None] * 64 + offset_11[None,:] * 1) # v_1_data -> ['offset', 'offset_12', 'offset_11'] + ax_5_0
        T_einsum_1_data = tl.zeros([64, 256], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_7 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_7 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + 0 * 393216 + offset * 32768 + offset_5[:,None] * 64 + offset_7[None,:] * 1) # q_1_data -> ['offset', 'offset_5', 'offset_7'] + ax_6_0
            k_1_data = tl.load(k_1 + 0 * 393216 + offset * 32768 + offset_6[:,None] * 64 + offset_7[None,:] * 1) # k_1_data -> ['offset', 'offset_6', 'offset_7'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['T_einsum_1_data', 'k_1_data', 'q_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_1_data', 'compute_red_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['T_einsum_2_data', 'compute_1_data', 'v_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + 0 * 393216 + offset * 32768 + offset_13[:,None] * 64 + offset_14[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_13', 'offset_14'] + __root

@triton.jit
def main_kernel4(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_5 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_5 -> ['ax_3_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_7 = tl.arange(0, 64) * 1 # offset_7 -> ['__root'] + __root
    q_1_data = tl.load(q_1 + 0 * 786432 + offset * 65536 + offset_5[:,None] * 64 + offset_7[None,:] * 1) # q_1_data -> ['offset', 'offset_5', 'offset_7'] + __root
    offset_8 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_8 -> ['ax_3_0'] + __root
    offset_11 = tl.arange(0, 64) * 1 # offset_11 -> ['__root'] + __root
    offset_13 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_13 -> ['ax_3_0'] + __root
    offset_14 = tl.arange(0, 64) * 1 # offset_14 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([64, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([64, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 16): # ax_5_0 -> ['__root'] + __root
        offset_6 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_6 -> ['ax_5_0'] + ax_5_0
        k_1_data = tl.load(k_1 + 0 * 786432 + offset * 65536 + offset_6[:,None] * 64 + offset_7[None,:] * 1) # k_1_data -> ['offset', 'offset_6', 'offset_7'] + ax_5_0
        offset_9 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_9 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + 0 * 12582912 + offset * 1048576 + offset_8[:,None] * 1024 + offset_9[None,:] * 1) # mask_1_data -> ['offset', 'offset_8', 'offset_9'] + ax_5_0
        offset_12 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_12 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + 0 * 786432 + offset * 65536 + offset_12[:,None] * 64 + offset_11[None,:] * 1) # v_1_data -> ['offset', 'offset_12', 'offset_11'] + ax_5_0
        T_einsum_1_data = tl.zeros([64, 64], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 1): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['T_einsum_1_data', 'q_1_data', 'k_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_1_data', 'compute_red_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['v_1_data', 'compute_1_data', 'T_einsum_2_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + 0 * 786432 + offset * 65536 + offset_13[:,None] * 64 + offset_14[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_13', 'offset_14'] + __root

@triton.jit
def main_kernel5(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    ax_4_0 = pid_x # ax_4_0 -> ['pid_x'] + __root
    offset_11 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_11 -> ['ax_4_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_5 = ax_3_0 * 16 + tl.arange(0, 16) * 1 # offset_5 -> ['ax_3_0'] + __root
    offset_8 = ax_3_0 * 16 + tl.arange(0, 16) * 1 # offset_8 -> ['ax_3_0'] + __root
    offset_13 = ax_3_0 * 16 + tl.arange(0, 16) * 1 # offset_13 -> ['ax_3_0'] + __root
    offset_14 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_14 -> ['ax_4_0'] + __root
    compute_red_1_data = tl.zeros([16, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([16, 16], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 4): # ax_5_0 -> ['__root'] + __root
        offset_6 = ax_5_0 * 512 + tl.arange(0, 512) * 1 # offset_6 -> ['ax_5_0'] + ax_5_0
        offset_9 = ax_5_0 * 512 + tl.arange(0, 512) * 1 # offset_9 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + 0 * 50331648 + offset * 4194304 + offset_8[:,None] * 2048 + offset_9[None,:] * 1) # mask_1_data -> ['offset', 'offset_8', 'offset_9'] + ax_5_0
        offset_12 = ax_5_0 * 512 + tl.arange(0, 512) * 1 # offset_12 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + 0 * 1572864 + offset * 131072 + offset_12[:,None] * 64 + offset_11[None,:] * 1) # v_1_data -> ['offset', 'offset_12', 'offset_11'] + ax_5_0
        T_einsum_1_data = tl.zeros([16, 512], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_7 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_7 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + 0 * 1572864 + offset * 131072 + offset_5[:,None] * 64 + offset_7[None,:] * 1) # q_1_data -> ['offset', 'offset_5', 'offset_7'] + ax_6_0
            k_1_data = tl.load(k_1 + 0 * 1572864 + offset * 131072 + offset_6[:,None] * 64 + offset_7[None,:] * 1) # k_1_data -> ['offset', 'offset_6', 'offset_7'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['k_1_data', 'q_1_data', 'T_einsum_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['mask_1_data', 'T_einsum_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_red_1_data', 'compute_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['T_einsum_2_data', 'compute_1_data', 'v_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + 0 * 1572864 + offset * 131072 + offset_13[:,None] * 64 + offset_14[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_13', 'offset_14'] + __root

@triton.jit
def main_kernel6(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    ax_4_0 = pid_x # ax_4_0 -> ['pid_x'] + __root
    offset_11 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_11 -> ['ax_4_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_5 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_5 -> ['ax_3_0'] + __root
    offset_8 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_8 -> ['ax_3_0'] + __root
    offset_13 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_13 -> ['ax_3_0'] + __root
    offset_14 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_14 -> ['ax_4_0'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 16], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 32): # ax_5_0 -> ['__root'] + __root
        offset_6 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_6 -> ['ax_5_0'] + ax_5_0
        offset_9 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_9 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + 0 * 201326592 + offset * 16777216 + offset_8[:,None] * 4096 + offset_9[None,:] * 1) # mask_1_data -> ['offset', 'offset_8', 'offset_9'] + ax_5_0
        offset_12 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_12 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + 0 * 3145728 + offset * 262144 + offset_12[:,None] * 64 + offset_11[None,:] * 1) # v_1_data -> ['offset', 'offset_12', 'offset_11'] + ax_5_0
        T_einsum_1_data = tl.zeros([128, 128], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_7 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_7 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + 0 * 3145728 + offset * 262144 + offset_5[:,None] * 64 + offset_7[None,:] * 1) # q_1_data -> ['offset', 'offset_5', 'offset_7'] + ax_6_0
            k_1_data = tl.load(k_1 + 0 * 3145728 + offset * 262144 + offset_6[:,None] * 64 + offset_7[None,:] * 1) # k_1_data -> ['offset', 'offset_6', 'offset_7'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['q_1_data', 'k_1_data', 'T_einsum_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['mask_1_data', 'T_einsum_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_1_data', 'compute_red_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['T_einsum_2_data', 'compute_1_data', 'v_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + 0 * 3145728 + offset * 262144 + offset_13[:,None] * 64 + offset_14[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_13', 'offset_14'] + __root


@triton.jit
def main_kernel7(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 16, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 16, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    offset_6 = tl.arange(0, 128) * 1 # offset_6 -> ['__root'] + __root
    offset_7 = tl.arange(0, 128) * 1 # offset_7 -> ['__root'] + __root
    offset_9 = tl.arange(0, 128) * 1 # offset_9 -> ['__root'] + __root
    offset_10 = tl.arange(0, 128) * 1 # offset_10 -> ['__root'] + __root
    # mask_1_data = tl.load(mask_1 + offset * 196608 + offset_1 * 16384 + offset_9[:,None] * 128 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + __root
    mask_1_data = tl.load(mask_1 + 0 * 196608 + offset_1 * 16384 + offset_9[:,None] * 128 + offset_10[None,:] * 1)
    offset_13 = tl.arange(0, 128) * 1 # offset_13 -> ['__root'] + __root
    offset_12 = tl.arange(0, 64) * 1 # offset_12 -> ['__root'] + __root
    v_1_data = tl.load(v_1 + offset *98304 + offset_1 * 8192 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + __root
    offset_14 = tl.arange(0, 128) * 1 # offset_14 -> ['__root'] + __root
    offset_15 = tl.arange(0, 64) * 1 # offset_15 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 1): # ax_5_0 -> ['__root'] + __root
        T_einsum_1_data = tl.zeros([128, 128], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_8 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_8 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + offset * 98304 + offset_1 * 8192 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + ax_6_0
            k_1_data = tl.load(k_1 + offset * 98304 + offset_1 * 8192 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data+ tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['q_1_data', 'k_1_data', 'T_einsum_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_1_data', 'compute_red_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['v_1_data', 'compute_1_data', 'T_einsum_2_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + offset *98304 + offset_1 * 8192 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root


@triton.jit
def main_kernel8(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_6 -> ['ax_3_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 16, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 16, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    offset_9 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_12 = tl.arange(0, 64) * 1 # offset_12 -> ['__root'] + __root
    offset_14 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = tl.arange(0, 64) * 1 # offset_15 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 4): # ax_5_0 -> ['__root'] + __root
        offset_7 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_7 -> ['ax_5_0'] + ax_5_0
        offset_10 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_10 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + offset * 786432 + offset_1 * 65536 + offset_9[:,None] * 256 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + ax_5_0
        offset_13 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_13 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + offset *196608+ offset_1 * 16384 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + ax_5_0
        T_einsum_1_data = tl.zeros([128, 128], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 2): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_8 = ax_6_0 * 32 + tl.arange(0, 32) * 1 # offset_8 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + offset * 196608+ offset_1 * 16384 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + ax_6_0
            k_1_data = tl.load(k_1 + offset * 196608+ offset_1 * 16384 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['T_einsum_1_data', 'q_1_data', 'k_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_1_data', 'compute_red_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['compute_1_data', 'T_einsum_2_data', 'v_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + offset * 196608 + offset_1 * 16384+ offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root



@triton.jit
def main_kernel9(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_6 -> ['ax_3_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 16, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 16, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    offset_9 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_12 = tl.arange(0, 64) * 1 # offset_12 -> ['__root'] + __root
    offset_14 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = tl.arange(0, 64) * 1 # offset_15 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 4): # ax_5_0 -> ['__root'] + __root
        offset_7 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_7 -> ['ax_5_0'] + ax_5_0
        offset_10 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_10 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + offset * 3145728 + offset_1 * 262144 + offset_9[:,None] * 512 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + ax_5_0
        offset_13 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_13 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + offset *393216 + offset_1 * 32768 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + ax_5_0
        T_einsum_1_data = tl.zeros([128, 128], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 2): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_8 = ax_6_0 * 32 + tl.arange(0, 32) * 1 # offset_8 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + offset * 393216 + offset_1 * 32768 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + ax_6_0
            k_1_data = tl.load(k_1 + offset * 393216 + offset_1 * 32768 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['T_einsum_1_data', 'q_1_data', 'k_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_1_data', 'compute_red_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['compute_1_data', 'T_einsum_2_data', 'v_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + offset * 393216 + offset_1 * 32768 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root

@triton.jit
def main_kernel10(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_6 -> ['ax_3_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 8, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 8, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    offset_9 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_12 = tl.arange(0, 64) * 1 # offset_12 -> ['__root'] + __root
    offset_14 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = tl.arange(0, 64) * 1 # offset_15 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 8): # ax_5_0 -> ['__root'] + __root
        offset_7 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_7 -> ['ax_5_0'] + ax_5_0
        offset_10 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_10 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + offset * 12582912 + offset_1 * 1048576 + offset_9[:,None] * 1024 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + ax_5_0
        offset_13 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_13 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + offset * 786432 + offset_1 * 65536 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + ax_5_0
        T_einsum_1_data = tl.zeros([128, 128], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_8 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_8 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + offset * 786432 + offset_1 * 65536 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + ax_6_0
            k_1_data = tl.load(k_1 + offset * 786432 + offset_1 * 65536 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['k_1_data', 'T_einsum_1_data', 'q_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['mask_1_data', 'T_einsum_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_1_data', 'compute_red_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['T_einsum_2_data', 'compute_1_data', 'v_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + offset * 786432 + offset_1 * 65536 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root


@triton.jit
def main_kernel11(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    ax_4_0 = pid_x # ax_4_0 -> ['pid_x'] + __root
    offset_12 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_12 -> ['ax_4_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 8, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 8, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_6 -> ['ax_3_0'] + __root
    offset_9 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_14 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_15 -> ['ax_4_0'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 16], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 16): # ax_5_0 -> ['__root'] + __root
        offset_7 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_7 -> ['ax_5_0'] + ax_5_0
        offset_10 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_10 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + offset * 50331648 + offset_1 * 4194304 + offset_9[:,None] * 2048 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + ax_5_0
        offset_13 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_13 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + offset * 1572864 + offset_1 * 131072 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + ax_5_0
        T_einsum_1_data = tl.zeros([128, 128], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_8 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_8 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + offset * 1572864 + offset_1 * 131072 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + ax_6_0
            k_1_data = tl.load(k_1 + offset * 1572864 + offset_1 * 131072 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['k_1_data', 'q_1_data', 'T_einsum_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['mask_1_data', 'T_einsum_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_red_1_data', 'compute_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['v_1_data', 'T_einsum_2_data', 'compute_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['compute_red_1_data', 'T_einsum_2_data'] + __root
    _ = tl.store(T_divide_1 + offset * 1572864 + offset_1 * 131072 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root

@triton.jit
def main_kernel12(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    ax_4_0 = pid_x # ax_4_0 -> ['pid_x'] + __root
    offset_12 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_12 -> ['ax_4_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 8, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 8, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_6 -> ['ax_3_0'] + __root
    offset_9 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_14 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_15 -> ['ax_4_0'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 16], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 8): # ax_5_0 -> ['__root'] + __root
        offset_7 = ax_5_0 * 512 + tl.arange(0, 512) * 1 # offset_7 -> ['ax_5_0'] + ax_5_0
        offset_10 = ax_5_0 * 512 + tl.arange(0, 512) * 1 # offset_10 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + offset * 201326592 + offset_1 * 16777216 + offset_9[:,None] * 4096 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + ax_5_0
        offset_13 = ax_5_0 * 512 + tl.arange(0, 512) * 1 # offset_13 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + offset * 3145728 + offset_1 * 262144 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + ax_5_0
        T_einsum_1_data = tl.zeros([128, 512], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_8 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_8 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + offset * 3145728 + offset_1 * 262144 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + ax_6_0
            k_1_data = tl.load(k_1 + offset * 3145728 + offset_1 * 262144 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['k_1_data', 'T_einsum_1_data', 'q_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_1_data', 'compute_red_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['T_einsum_2_data', 'v_1_data', 'compute_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + offset * 3145728 + offset_1 * 262144 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root


@triton.jit
def main_kernel13(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    ax_4_0 = pid_x # ax_4_0 -> ['pid_x'] + __root
    offset_12 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_12 -> ['ax_4_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 16, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 16, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 16 + tl.arange(0, 16) * 1 # offset_6 -> ['ax_3_0'] + __root
    offset_7 = tl.arange(0, 128) * 1 # offset_7 -> ['__root'] + __root
    offset_9 = ax_3_0 * 16 + tl.arange(0, 16) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_10 = tl.arange(0, 128) * 1 # offset_10 -> ['__root'] + __root
    mask_1_data = tl.load(mask_1 + offset * 196608 + offset_1 * 16384 + offset_9[:,None] * 128 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + __root
    offset_13 = tl.arange(0, 128) * 1 # offset_13 -> ['__root'] + __root
    v_1_data = tl.load(v_1 + offset * 98304 + offset_1 * 8192 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + __root
    offset_14 = ax_3_0 * 16 + tl.arange(0, 16) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_15 -> ['ax_4_0'] + __root
    compute_red_1_data = tl.zeros([16, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([16, 16], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 1): # ax_5_0 -> ['__root'] + __root
        T_einsum_1_data = tl.zeros([16, 128], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_8 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_8 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + offset * 98304 + offset_1 * 8192 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + ax_6_0
            k_1_data = tl.load(k_1 + offset * 98304 + offset_1 * 8192 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['q_1_data', 'k_1_data', 'T_einsum_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['mask_1_data', 'T_einsum_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_1_data', 'compute_red_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['compute_1_data', 'v_1_data', 'T_einsum_2_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['compute_red_1_data', 'T_einsum_2_data'] + __root
    _ = tl.store(T_divide_1 + offset * 98304 + offset_1 * 8192 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root

@triton.jit
def main_kernel14(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_6 -> ['ax_3_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 8, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 8, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    offset_8 = tl.arange(0, 64) * 1 # offset_8 -> ['__root'] + __root
    q_1_data = tl.load(q_1 + offset * 196608 + offset_1 * 16384 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + __root
    offset_9 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_12 = tl.arange(0, 64) * 1 # offset_12 -> ['__root'] + __root
    offset_14 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = tl.arange(0, 64) * 1 # offset_15 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([64, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([64, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 8): # ax_5_0 -> ['__root'] + __root
        offset_7 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_7 -> ['ax_5_0'] + ax_5_0
        k_1_data = tl.load(k_1 + offset *196608 + offset_1 * 16384+ offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_5_0
        offset_10 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_10 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + offset * 786432 + offset_1 * 65536 + offset_9[:,None] *256 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + ax_5_0
        offset_13 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_13 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + offset *196608 + offset_1 * 16384 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + ax_5_0
        T_einsum_1_data = tl.zeros([64, 64], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 1): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['k_1_data', 'q_1_data', 'T_einsum_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_red_1_data', 'compute_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['T_einsum_2_data', 'compute_1_data', 'v_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + offset * 196608 + offset_1 * 16384 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root



@triton.jit
def main_kernel15(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_6 -> ['ax_3_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 8, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 8, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    offset_8 = tl.arange(0, 64) * 1 # offset_8 -> ['__root'] + __root
    q_1_data = tl.load(q_1 + offset * 393216 + offset_1 * 32768 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + __root
    offset_9 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_12 = tl.arange(0, 64) * 1 # offset_12 -> ['__root'] + __root
    offset_14 = ax_3_0 * 64 + tl.arange(0, 64) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = tl.arange(0, 64) * 1 # offset_15 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([64, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([64, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 8): # ax_5_0 -> ['__root'] + __root
        offset_7 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_7 -> ['ax_5_0'] + ax_5_0
        k_1_data = tl.load(k_1 + offset * 393216 + offset_1 * 32768 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_5_0
        offset_10 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_10 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + offset * 3145728 + offset_1 * 262144 + offset_9[:,None] * 512 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + ax_5_0
        offset_13 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_13 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + offset * 393216 + offset_1 * 32768 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + ax_5_0
        T_einsum_1_data = tl.zeros([64, 64], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 1): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['k_1_data', 'q_1_data', 'T_einsum_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_red_1_data', 'compute_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['T_einsum_2_data', 'compute_1_data', 'v_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + offset * 393216 + offset_1 * 32768 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root

@triton.jit
def main_kernel16(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_6 -> ['ax_3_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 16, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 16, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    offset_9 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_12 = tl.arange(0, 64) * 1 # offset_12 -> ['__root'] + __root
    offset_14 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = tl.arange(0, 64) * 1 # offset_15 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 32): # ax_5_0 -> ['__root'] + __root
        offset_7 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_7 -> ['ax_5_0'] + ax_5_0
        offset_10 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_10 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + offset * 12582912 + offset_1 * 1048576 + offset_9[:,None] * 1024 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + ax_5_0
        offset_13 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_13 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + offset * 786432 + offset_1 * 65536 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + ax_5_0
        T_einsum_1_data = tl.zeros([128, 64], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 2): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_8 = ax_6_0 * 32 + tl.arange(0, 32) * 1 # offset_8 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + offset * 786432 + offset_1 * 65536 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + ax_6_0
            k_1_data = tl.load(k_1 + offset * 786432 + offset_1 * 65536 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['q_1_data', 'T_einsum_1_data', 'k_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_red_1_data', 'compute_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['v_1_data', 'compute_1_data', 'T_einsum_2_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['compute_red_1_data', 'T_einsum_2_data'] + __root
    _ = tl.store(T_divide_1 + offset * 786432 + offset_1 * 65536 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root



@triton.jit
def main_kernel17(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_6 -> ['ax_3_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 16, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 16, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    offset_9 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_12 = tl.arange(0, 64) * 1 # offset_12 -> ['__root'] + __root
    offset_14 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = tl.arange(0, 64) * 1 # offset_15 -> ['__root'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 64], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 32): # ax_5_0 -> ['__root'] + __root
        offset_7 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_7 -> ['ax_5_0'] + ax_5_0
        offset_10 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_10 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + offset * 50331648 + offset_1 * 4194304 + offset_9[:,None] * 2048 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + ax_5_0
        offset_13 = ax_5_0 * 64 + tl.arange(0, 64) * 1 # offset_13 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + offset * 1572864 + offset_1 * 131072 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + ax_5_0
        T_einsum_1_data = tl.zeros([128, 64], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 2): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_8 = ax_6_0 * 32 + tl.arange(0, 32) * 1 # offset_8 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + offset * 1572864 + offset_1 * 131072 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + ax_6_0
            k_1_data = tl.load(k_1 + offset * 1572864 + offset_1 * 131072 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['q_1_data', 'T_einsum_1_data', 'k_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['T_einsum_1_data', 'mask_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_red_1_data', 'compute_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['v_1_data', 'compute_1_data', 'T_einsum_2_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['compute_red_1_data', 'T_einsum_2_data'] + __root
    _ = tl.store(T_divide_1 + offset * 1572864 + offset_1 * 131072 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root

@triton.jit
def main_kernel18(T_divide_1, k_1, mask_1, q_1, v_1):
    pid_x = tl.program_id(axis=0) # pid_x -> ['__root'] + __root
    ax_4_0 = pid_x # ax_4_0 -> ['pid_x'] + __root
    offset_12 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_12 -> ['ax_4_0'] + __root
    pid_z = tl.program_id(axis=2) # pid_z -> ['__root'] + __root
    ax_1_ax_2_fused = pid_z # ax_1_ax_2_fused -> ['pid_z'] + __root
    offset = tl.broadcast_to(ax_1_ax_2_fused // 8, (1,)) # offset -> ['ax_1_ax_2_fused'] + __root
    offset_1 = tl.broadcast_to(ax_1_ax_2_fused % 8, (1,)) # offset_1 -> ['ax_1_ax_2_fused'] + __root
    pid_y = tl.program_id(axis=1) # pid_y -> ['__root'] + __root
    ax_3_0 = pid_y # ax_3_0 -> ['pid_y'] + __root
    offset_6 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_6 -> ['ax_3_0'] + __root
    offset_9 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_9 -> ['ax_3_0'] + __root
    offset_14 = ax_3_0 * 128 + tl.arange(0, 128) * 1 # offset_14 -> ['ax_3_0'] + __root
    offset_15 = ax_4_0 * 16 + tl.arange(0, 16) * 1 # offset_15 -> ['ax_4_0'] + __root
    compute_red_1_data = tl.zeros([128, 1], dtype=tl.float32) # compute_red_1_data -> ['__root'] + __root
    T_einsum_2_data = tl.zeros([128, 16], dtype=tl.float32) # T_einsum_2_data -> ['__root'] + __root
    for ax_5_0 in range(0, 32): # ax_5_0 -> ['__root'] + __root
        offset_7 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_7 -> ['ax_5_0'] + ax_5_0
        offset_10 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_10 -> ['ax_5_0'] + ax_5_0
        mask_1_data = tl.load(mask_1 + offset * 201326592 + offset_1 * 16777216 + offset_9[:,None] * 4096 + offset_10[None,:] * 1) # mask_1_data -> ['offset', 'offset_1', 'offset_9', 'offset_10'] + ax_5_0
        offset_13 = ax_5_0 * 128 + tl.arange(0, 128) * 1 # offset_13 -> ['ax_5_0'] + ax_5_0
        v_1_data = tl.load(v_1 + offset * 3145728 + offset_1 * 262144 + offset_13[:,None] * 64 + offset_12[None,:] * 1) # v_1_data -> ['offset', 'offset_1', 'offset_13', 'offset_12'] + ax_5_0
        T_einsum_1_data = tl.zeros([128, 128], dtype=tl.float32) # T_einsum_1_data -> ['ax_5_0'] + ax_5_0
        for ax_6_0 in range(0, 4): # ax_6_0 -> ['ax_5_0'] + ax_5_0
            offset_8 = ax_6_0 * 16 + tl.arange(0, 16) * 1 # offset_8 -> ['ax_6_0'] + ax_6_0
            q_1_data = tl.load(q_1 + offset * 3145728 + offset_1 * 262144 + offset_6[:,None] * 64 + offset_8[None,:] * 1) # q_1_data -> ['offset', 'offset_1', 'offset_6', 'offset_8'] + ax_6_0
            k_1_data = tl.load(k_1 + offset * 3145728 + offset_1 * 262144 + offset_7[:,None] * 64 + offset_8[None,:] * 1) # k_1_data -> ['offset', 'offset_1', 'offset_7', 'offset_8'] + ax_6_0
            T_einsum_1_data = (T_einsum_1_data + tl.dot(q_1_data, k_1_data.T)) # T_einsum_1_data_1 -> ['k_1_data', 'q_1_data', 'T_einsum_1_data'] + ax_6_0
        compute_1_data = tl.exp(((T_einsum_1_data * 0.125) + mask_1_data)) # compute_1_data -> ['mask_1_data', 'T_einsum_1_data'] + ax_5_0
        compute_red_1_data = (compute_red_1_data + tl.reshape(tl.sum(compute_1_data, axis=1), compute_red_1_data.shape)) # compute_red_1_data_1 -> ['compute_red_1_data', 'compute_1_data'] + ax_5_0
        T_einsum_2_data = (T_einsum_2_data + tl.dot(compute_1_data.to(tl.float16), v_1_data)) # T_einsum_2_data_1 -> ['T_einsum_2_data', 'v_1_data', 'compute_1_data'] + ax_5_0
    T_divide_1_data = (T_einsum_2_data / compute_red_1_data) # T_divide_1_data -> ['T_einsum_2_data', 'compute_red_1_data'] + __root
    _ = tl.store(T_divide_1 + offset * 3145728 + offset_1 * 262144 + offset_14[:,None] * 64 + offset_15[None,:] * 1, T_divide_1_data) # _ -> ['T_divide_1_data', 'offset', 'offset_1', 'offset_14', 'offset_15'] + __root


def mcfuser_attn(batch_size,head_num,seq_len,head_size,mask):
    # torch.cuda.empty_cache()
    torch.manual_seed(0)
    # 定义矩阵尺寸
    Q = torch.randn((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
    K = torch.randn((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
    V = torch.randn((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
    T= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
    T2= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
    if batch_size==1 and seq_len==128:
       
        main_kernel1[(4,1,12)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==1 and seq_len==256:
       
        main_kernel2[(1,4,12)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==1 and seq_len==512:
       
        main_kernel3[(1,8,12)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==1 and seq_len==1024:
       
        main_kernel4[(1,16,12)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==1 and (seq_len==2048):
       
        # main_kernel5[(4,32,12)](T,K,mask,Q,V)
        main_kernel5[(4,128,12)](T,K,mask,Q,V)
              
        return T
    
    elif batch_size==1 and seq_len==4096:
       
        main_kernel6[(4,32,12)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==8 and seq_len==128:
       
        main_kernel7[(1,2,128)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==8 and seq_len==256:
       
        main_kernel8[(1,2,128)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==8 and seq_len==512:
       
        main_kernel9[(1,4,128)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==8 and seq_len==1024:
       
        main_kernel10[(1,8,64)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==8 and seq_len==2048:
       
        main_kernel11[(4,16,64)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==8 and seq_len==4096:           #！！！！！！！！！！！！！！！！！！！！！
       
        main_kernel12[(4,32,64)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==16 and seq_len==128:
       
        main_kernel13[(4,8,256)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==16 and seq_len==256:
       
        main_kernel14[(1,8,128)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==16 and seq_len==512:
       
        main_kernel15[(1,8,128)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==16 and seq_len==1024:
       
        main_kernel16[(4,8,128)](T,K,mask,Q,V)
      
        return T
    
    elif batch_size==16 and seq_len==2048:    
       
        main_kernel17[(1,16,128)](T,K,mask,Q,V)
      
        return T
    
    else:  # batch_size==16 and seq_len==4096:         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!error
        print("Out of Memory on 4090")
       
        main_kernel18[(4,32,128)](T,K,mask,Q,V)
      
        return T   

   