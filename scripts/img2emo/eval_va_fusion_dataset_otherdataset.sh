#!/bin/bash

# # First get individual arguments
# param1=$1
# param2=$2
# shift 2  # Remove first two arguments

# # Remaining arguments become array
# args=("$@")

# echo "param1: $param1"
# echo "param2: $param2"
# echo "Array arguments: ${args[@]}"

# # Access array elements
# for i in "${!args[@]}"; do
#     echo "args[$i] = ${args[i]}"
# done



export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# custom config
DATA="/data/dingsd/img2emo/data"
TRAINER=EmoticMaxVit
DATASET=va
SEED=$1
TEST_DATASET=gaped

CFG=emotic_maxvit_ep250
# SHOTS=16

# I need the shell script to take a list of source domains.
# Define the list of strings
# SOURCE_DOMAINS=("EMOTIC" "NAPS_H" "OASIS" "EmoMadrid" "GAPED" "Emotion6")
shift 1  # Remove the first argument to get the source domains arguments
SOURCE_DOMAINS=("$@")
echo "SOURCE_DOMAINS: ${SOURCE_DOMAINS[@]}"
# Join the elements of the list with "_" using parameter expansion
SOURCE_DOMAINS_STR="${SOURCE_DOMAINS[*]}"
SOURCE_DOMAINS_STR="${SOURCE_DOMAINS_STR// /_}"  # Replace spaces with underscores

DIR=output/${DATASET}/${TRAINER}/${CFG}/${SOURCE_DOMAINS_STR}/random_domain_sampler_average_domain_eval_seed${SEED}

python3 train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${TEST_DATASET}.yml \
    --config-file configs/trainers/img2emo/${CFG}.yml \
    --source-domains GAPED \
    --output-dir ${DIR} \
    --model-dir ${DIR} \
    --eval-only