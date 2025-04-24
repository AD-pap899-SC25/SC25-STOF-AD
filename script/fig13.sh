#!/bin/bash

cd ../src

models=("bert_small" "bert_base" "bert_large" "gpt" "t5")


for model in "${models[@]}"; do
    python ablation_onlyMHA.py \
        --batch_size=1 --seq_len=128 \
        --model="$model" \
        --method="STOF"
done


for model in "${models[@]}"; do
    python ablation_onlyMHA.py \
        --batch_size=8 --seq_len=512 \
        --model="$model" \
        --method="STOF"
done


for model in "${models[@]}"; do
    python ablation_onlyMHA.py \
        --batch_size=16 --seq_len=2048 \
        --model="$model" \
        --method="STOF"
done


for model in "${models[@]}"; do
    python ablation_onlyfusion.py \
        --batch_size=1 --seq_len=128 \
        --model="$model" \
        --method="STOF-Compiled"
done

for model in "${models[@]}"; do
    python ablation_onlyfusion.py \
        --batch_size=8 --seq_len=512 \
        --model="$model" \
        --method="STOF-Compiled"
done

for model in "${models[@]}"; do
    python ablation_onlyfusion.py \
        --batch_size=16 --seq_len=2048 \
        --model="$model" \
        --method="STOF-Compiled"
done

cd ../plot/fig13
python fig13.py