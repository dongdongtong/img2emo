#!/bin/bash

#cd ../..

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# custom config
DATA="/root/autodl-tmp"
TRAINER=EmoticVitLora
DATASET=va
SEED=$1
BATCH_SIZE=$2

CFG=emotic_vit_lora_ep20
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

DIR=output/${DATASET}/${TRAINER}/${CFG}/${SOURCE_DOMAINS_STR}/more_trivial_augment_seed${SEED}_method2
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python3 train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yml \
    --config-file configs/trainers/img2emo/${CFG}.yml \
    --source-domains ${SOURCE_DOMAINS[*]} \
    --output-dir ${DIR} \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH_SIZE} \
    INPUT.TRANSFORMS trivial_transform
fi