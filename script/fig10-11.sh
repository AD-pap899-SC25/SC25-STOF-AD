#!/bin/bash
cd ../src

mask_ids=(1 2 3 4)
for mask_id in "${mask_ids[@]}"; do
    python benchmk_attn_unified.py --mask_id="$mask_id"
done

cd ../plot/fig10-11
python fig10-11.py