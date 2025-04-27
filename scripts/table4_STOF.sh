
models=("bert_small" "bert_base" "bert_large" "gpt" "t5")

cd ../src

for model in "${models[@]}"; do
    python tuning_STOF_cost.py \
        --method="STOF" \
        --batch_size=1 \
        --seq_len=128 \
        --model="$model"
done

for model in "${models[@]}"; do
    python tuning_STOF_cost.py \
        --method="STOF" \
        --batch_size=8 \
        --seq_len=512 \
        --model="$model"
done

for model in "${models[@]}"; do
    python tuning_STOF_cost.py \
        --method="STOF" \
        --batch_size=16 \
        --seq_len=2048 \
        --model="$model"
done