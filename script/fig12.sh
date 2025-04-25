#!/bin/bash
cd ../src


models=("bert_small" "bert_base" "bert_large" "gpt" "t5")
methods=("TorchNative" "TorchCompile" "ByteTrans" "STOF")


for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        python benchmk_end2end.py \
            --batch_size=1 \
            --seq_len=128 \
            --model="$model" \
            --method="$method"
    done
done

for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        python benchmk_end2end.py \
            --batch_size=8 \
            --seq_len=512 \
            --model="$model" \
            --method="$method"
    done
done

for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        python benchmk_end2end.py \
            --batch_size=16 \
            --seq_len=2048 \
            --model="$model" \
            --method="$method"
    done
done


cd ../plot/fig12
python fig12.py