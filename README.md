# AnyProject

AnyProject is a prototype project for medical image analysis, allowing you to build highly custom project quickly based on it.

This project allows for custom transforms for medical image using MONAI. Check the example data/transforms.

For now, it cannot support multi-gpu distributed parallel training.

It has contained an segmentation example.

## Add Custom Dataset
You need to define the structure of your dataset, read the available data from your original directory.
Some steps:

* create a dataset.py like `data/datasets/hematoma.py` to define the available data, including images, labels, segmentations.
* Use the `DATASET_REGISTRY` to register it.
* Define a dataset wrapper to help load the data like `HematomaSegWrapper` in `data/datasets/hematoma.py`.
* Use the custom dataset name and dataset wrapper name in the newly-created `configs/datasets/*.yml`.

## Add Custom Transforms
Just follow the same steps as `Add Custom Dataset`. 
But now, you need to create the custom transforms in the `data/transforms` like `data/transforms/hematoma_seg_augs.py`.

## Add Custom trainer
Refer to `trainers/segmentation/hematoma_seg.py`.

## Run the code.
For example, run `bash scripts/hematoma_seg/run.sh`.

# Acknowledgement
This project is heavily copied from [dassl](https://github.com/KaiyangZhou/Dassl.pytorch).