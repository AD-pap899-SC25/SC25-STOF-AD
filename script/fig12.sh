#!/bin/bash
cd ../src


batch_sizes=(1)
seq_lens=(256)
# batch_sizes=(1 8 16)
# seq_lens=(128 256 512 1024 2048 4096)
models=("bert_small" "bert_base" "bert_large" "gpt" "t5")
methods=("TorchNative" "TorchCompile" "ByteTrans" "STOF")


for bs in "${batch_sizes[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        for model in "${models[@]}"; do
            for method in "${methods[@]}"; do
                python benchmk_end2end.py \
                    --batch_size="$bs" \
                    --seq_len="$seq_len" \
                    --model="$model" \
                    --method="$method"
            done
        done
    done
done

cd ../plot/fig12
python fig12.py