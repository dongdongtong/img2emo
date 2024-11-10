# Img2Emo Project

The Img2Emo project is designed to train and evaluate models for valence and arousal estimation from images. This guide will help you understand how to use the provided scripts to train and evaluate models using different datasets and configurations.

## Prerequisites

Ensure you have the necessary environment set up to run the scripts, including any dependencies and datasets required for the project.
Requirements:

- Python 3.10
- Pytorch >= 2.0

## Running the Scripts

#### Train Simple Baseline Model

To train a simple baseline model, you can use the following commands. Uncomment the desired line in the `run.sh` script to execute:

#### Train with EMOTIC dataset

```
bash scripts/img2emo/va_random_sampler.sh 1 EMOTIC
```

#### Train with EMOTIC and NAPS_H datasets

```
bash scripts/img2emo/va_random_sampler.sh 1 EMOTIC NAPS_H
```

#### Train with EMOTIC, NAPS_H, and OASIS datasets

```
bash scripts/img2emo/va_random_sampler.sh 1 EMOTIC NAPS_H OASIS
```


### Evaluate Simple & Fair Baseline Model

To evaluate the baseline model, use the following commands. Uncomment the desired line in the `run.sh` script to execute:

#### Evaluate with EMOTIC dataset

```
bash scripts/img2emo/eval_va_fusion_dataset.sh 1 EMOTIC
```

#### Evaluate with EMOTIC and NAPS_H datasets

```
bash scripts/img2emo/eval_va_fusion_dataset.sh 1 EMOTIC NAPS_H
```

#### Evaluate with EMOTIC, NAPS_H, and OASIS datasets

```
bash scripts/img2emo/eval_va_fusion_dataset.sh 1 EMOTIC NAPS_H OASIS
```


### Scale up Maxvit_tiny to Giant Vit using LoRA

To scale up the model using LoRA, use the following commands. Uncomment the desired line in the `run.sh` script to execute:


#### Scale up with EMOTIC, NAPS_H, and OASIS datasets

```
bash scripts/img2emo/va_giantvit_lora.sh 1 32 EMOTIC NAPS_H OASIS
```

#### Scale up with EMOTIC, NAPS_H, OASIS, and Emotion6 datasets

```
bash scripts/img2emo/va_giantvit_lora.sh 1 32 EMOTIC NAPS_H OASIS Emotion6
```



## Acknowledgements

This project is inspired by and builds upon various existing works in the field of emotion recognition and model scaling.

For more details, refer to the individual script files and their documentation.