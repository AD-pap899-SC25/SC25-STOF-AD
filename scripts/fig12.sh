#!/bin/bash
cd ../src


models=("bert_small" "bert_base" "bert_large" "gpt" "t5")
methods=("TorchNative" "TorchCompile" "ByteTrans")
batch_seq_pairs=("1 128" "8 512" "16 2048")

for bs_seq in "${batch_seq_pairs[@]}"; do
    bs=$(echo $bs_seq | cut -d' ' -f1)
    seq=$(echo $bs_seq | cut -d' ' -f2)
    
    for model in "${models[@]}"; do
        for method in "${methods[@]}"; do
            script -a -c "python benchmk_end2end.py \
                --batch_size=$bs \
                --seq_len=$seq \
                --model=$model \
                --method=$method" \
                "../data/End2End_Performance/fig12_stof_other.txt"
        done

        script -a -c "python benchmk_end2end.py \
            --batch_size=$bs \
            --seq_len=$seq \
            --model=$model \
            --method=STOF" \
            "../data/End2End_Performance/fig12_stof.txt"

    done
done
    

cd ../plot/fig12
python fig12_single_device.py \
    --stof_other_device="../../data/End2End_Performance/fig12_stof_other.txt" \
    --stof_device="../../data/End2End_Performance/fig12_stof.txt" \

