#!/bin/bash

models=("bert_small" "bert_base" "bert_large" "gpt" "t5")

cd ../src

for model in "${models[@]}"; do
    python overhead_analysis.py \
        --batch_size=1 \
        --seq_len=128 \
        --model="$model"
done

for model in "${models[@]}"; do
    python overhead_analysis.py \
        --batch_size=8 \
        --seq_len=512 \
        --model="$model"
done

for model in "${models[@]}"; do
    python overhead_analysis.py \
        --batch_size=16 \
        --seq_len=2048 \
        --model="$model"
done

cd ../plot/fig14
python fig14.py