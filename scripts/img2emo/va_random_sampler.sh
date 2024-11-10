#!/bin/bash

#cd ../..

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# custom config
DATA="/data/dingsd/img2emo/data"
TRAINER=EmoticMaxVit
DATASET=va
SEED=$1

CFG=emotic_maxvit_ep20
# SHOTS=16

# I need the shell script to take a list of source domains.
# Define the list of strings
# SOURCE_DOMAINS=("EMOTIC")
shift 1  # Remove the first argument to get the source domains arguments
SOURCE_DOMAINS=("$@")
# Join the elements of the list with "_" using parameter expansion
SOURCE_DOMAINS_STR="${SOURCE_DOMAINS[*]}"
SOURCE_DOMAINS_STR="${SOURCE_DOMAINS_STR// /_}"  # Replace spaces with underscores

DIR=output/${DATASET}/${TRAINER}/${CFG}/${SOURCE_DOMAINS_STR}/seed${SEED}
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
    --output-dir ${DIR}
    # INPUT.TRANSFORMS trivial_transform
fi