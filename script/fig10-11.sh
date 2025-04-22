#!/bin/bash
cd ../src

script -c "python benchmk_attn_unified.py \
    --mask_id=1 --head_num=12 --head_size=64 --print_flag=True" MHA_result.txt