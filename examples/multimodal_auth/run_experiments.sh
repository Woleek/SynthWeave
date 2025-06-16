#!/bin/bash

trap "echo 'Interrupted. Exiting.'; exit 1" SIGINT

FUSIONS=("CFF" "AFF" "IAFF" "CAFF" "GFF" "MMD")
DATASETS=("SWAN_DF" "DeepSpeak_v1_1")
declare -A DATA_DIRS=(
    [DeepSpeak_v1_1]="./encoded_data/DeepSpeak_v1_1"
    [SWAN_DF]="./encoded_data/SWAN_DF_320"
)
declare -A MAX_EPOCHS=(
    [DeepSpeak_v1_1]=100
    [SWAN_DF]=25
)

COMMON_ARGS="--encoded --batch_size 512 --task binary --emb_dim 512 --proj_dim 256 --dropout 0.3 --lr 1e-5 --iil_mode whitening --train_strategy FRADE"

# Train on each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Running experiments for dataset: $DATASET"

    # Train with each fusion method
    for FUSION in "${FUSIONS[@]}"; do
        echo "Running training with fusion: $FUSION"
        uv run train.py $COMMON_ARGS --fusion "$FUSION" --data_dir "${DATA_DIRS[$DATASET]}" --dataset "$DATASET" --max_epochs "${MAX_EPOCHS[$DATASET]}"

        # Evaluate on each dataset
        for EVAL_DATASET in "${DATASETS[@]}"; do
            echo "Running evaluation with fusion: $FUSION on dataset: $EVAL_DATASET"
            uv run evaluate.py --data_dir "${DATA_DIRS[$EVAL_DATASET]}" --dataset "$EVAL_DATASET" --task binary --fusion "$FUSION" --trained_on "$DATASET"
        done
    done
done