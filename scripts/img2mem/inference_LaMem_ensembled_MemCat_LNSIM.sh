#!/bin/bash

#cd ../..

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# custom config
DATA="/data/dingsd/img2emo/image_memoriability/datasets"
TRAINER=EnsembleMemCLIPVitLora
DATASET=clip_trivialaug_lnsim_cross_group
SEED=$1
BATCH_SIZE=$2

CFG=clip_lnsim_vit_lora_mse_srcc_ccc_ep40
# SHOTS=16


# I need the shell script to take a list of source domains.
# Define the list of strings
shift 2  # Remove the first two argument to get the source domains arguments
SOURCE_DOMAINS=("$@")
LEN_SOURCE_DOMAINS=${#SOURCE_DOMAINS[@]}
echo "BATCH_SIZE: ${BATCH_SIZE}"
# Join the elements of the list with "_" using parameter expansion
SOURCE_DOMAINS_STR="${SOURCE_DOMAINS[*]}"
SOURCE_DOMAINS_STR="${SOURCE_DOMAINS_STR// /_}"  # Replace spaces with underscores

DIR=output/${DATASET}/${TRAINER}/${CFG}/${SOURCE_DOMAINS_STR}/seed${SEED}

echo "Run this job and save the output to ${DIR}"

python3 train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/img2memoriability/${DATASET}.yml \
    --config-file configs/trainers/img2memoriablity_ensemble/${CFG}.yml \
    --source-domains ${SOURCE_DOMAINS[*]} \
    --output-dir ${DIR} \
    --eval-only \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH_SIZE} \
    TRAIN.GRADIENT_ACCUMULATION_STEPS 8