#!/bin/bash

# bash scripts/img2mem/clip_mem_giantvit_lora_srccloss_LaMem.sh 1 16 LaMem


# ====================================================================
# === Train Giant Vit using lora with mse+srcc+ccc loss ==============
# === Dataset: LaMem (60K) fold 1,2,3,4,5 ==================================
# ====================================================================
bash scripts/img2mem/clip_mem_giantvit_lora_mse_srcc_ccc_LaMem.sh 1 16 1 LaMem
# bash scripts/img2mem/clip_mem_giantvit_lora_mse_srcc_ccc_LaMem.sh 1 16 2 LaMem
# bash scripts/img2mem/clip_mem_giantvit_lora_mse_srcc_ccc_LaMem.sh 1 16 3 LaMem
# bash scripts/img2mem/clip_mem_giantvit_lora_mse_srcc_ccc_LaMem.sh 1 16 4 LaMem
# bash scripts/img2mem/clip_mem_giantvit_lora_mse_srcc_ccc_LaMem.sh 1 16 5 LaMem


# ====================================================================
# === Train Giant Vit using lora with mse+srcc+ccc loss ==============
# === Dataset: LNSIM (2,632) seed 1,2,3 ==============================
# ====================================================================
bash scripts/img2mem/clip_mem_giantvit_lora_mse_srcc_ccc_LNSIM.sh 1 16 LNSIM


# ====================================================================
# === Train Giant Vit using lora with mse+srcc+ccc loss ==============
# === Dataset: MemCat (10K) seed 1,2,3 ==============================
# ====================================================================
bash scripts/img2mem/clip_mem_giantvit_lora_mse_srcc_ccc_MemCat.sh 1 16 MemCat


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
