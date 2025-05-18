#!/bin/bash


# bash scripts/img2mem/mem_giantvit_lora.sh 1 32 LaMem
# bash scripts/img2mem/mem_giantvit_lora_srccloss.sh 1 32 LaMem

# bash scripts/img2mem/mem_giantvit_lora_srccloss_LNSIM.sh 1 32 LaMem


# Some inferences
# this is using pretrained resmem model directly on LNSIM.
# bash scripts/img2mem/inference_resmem.sh 1 32 LNSIM
# this is using our finetuned CLIP model (first LaMem, then MemCat) directly on LNSIM.
# bash scripts/img2mem/inference_finetuned_LaMem_MemCat_to_LNSIM.sh 1 32 LNSIM
bash scripts/img2mem/inference_LaMem_ensembled_MemCat_LNSIM.sh 1 1 LNSIM
# =============================================
# === Train Simple Baseline Model ============
# =============================================

# bash scripts/img2emo/va_random_sampler.sh 1 EMOTIC
# bash scripts/img2emo/va_random_sampler.sh 1 EMOTIC NAPS_H
# bash scripts/img2emo/va_random_sampler.sh 1 EMOTIC NAPS_H OASIS


# =============================================
# === Eval Simple & Fair Baseline Model ==============
# =============================================
# bash scripts/img2emo/eval_va_fusion_dataset.sh 1 EMOTIC
# bash scripts/img2emo/eval_va_fusion_dataset.sh 1 EMOTIC NAPS_H
# bash scripts/img2emo/eval_va_fusion_dataset.sh 1 EMOTIC NAPS_H OASIS


# ===============================================================
# === Scale up Maxvit_tiny to Giant Vit using lora ==============
# ===============================================================
# bash scripts/img2emo/va_giantvit_lora.sh 1 32 EMOTIC NAPS_H OASIS
# bash scripts/img2emo/va_giantvit_lora.sh 1 32 EMOTIC NAPS_H OASIS Emotion6


# ===============================================================
# === Eval Giant Vit using lora ==============
# ===============================================================
# bash scripts/img2emo/eval_va_fusion_dataset_loravit.sh 1 EMOTIC NAPS_H OASIS
