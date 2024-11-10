#!/bin/bash

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
