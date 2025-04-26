#!/bin/bash
cd ../src

mask_ids=(1 2 3 4)
for mask_id in "${mask_ids[@]}"; do
    script -c "python benchmk_attn_unified.py --mask_id=$mask_id"   \
        ../data/MHA_Performance/fig_10_11_mask_${mask_id}.txt
done

cd ../plot/fig10-11
python fig10-11.py --file_path1="../../data/MHA_Performance/fig_10_11_mask_1.txt" \
                    --file_path2="../../data/MHA_Performance/fig_10_11_mask_4.txt" \
                    --file_path3="../../data/MHA_Performance/fig_10_11_mask_2.txt" \
                    --file_path4="../../data/MHA_Performance/fig_10_11_mask_3.txt"