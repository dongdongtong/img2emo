# Created by: @luckydog on 2024-11-02
# This is a dataset for loading EMOTIC dataset.


"""Created by Dingsd on 11/01/2024 22:21
"""

from collections.abc import Sequence
import os
from os.path import basename, dirname, join

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split

from ..base_dataset import DatasetBase, ContinuousLabelDatum
from ..build import DATASET_REGISTRY, DATASET_WRAPPER_REGISTRY
from utils.tools import read_image
    
import os
import os.path as osp
import random
from copy import deepcopy

import pandas as pd

from typing import Dict, List

from utils import read_json, mkdir_if_missing, write_json


@DATASET_REGISTRY.register()
class MemCat(DatasetBase):
    # generate data list and saved!
    # make a data structure for hematoma segmentation
    # train val test split
    
    _lab2cname = {
        0: 'memoriability',
    }
    
    _classnames = ['memoriability',]
    
    _num_classes = 1
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        source_domain = cfg.DATASET.SOURCE_DOMAINS[0]
        
        # memcat's directory structure
        # MemCat images
        # ├── category
        # │   ├── subcategory
        # │   │   ├── image1.jpg
        # │   │   ├── image2.jpg
        # │   │   ├── ...
        
        
        image_dir = osp.join(cfg.DATASET.ROOT, source_domain, "MemCat_images", "MemCat")
        excel_path = osp.join(cfg.DATASET.ROOT, source_domain, "MemCat_data", "memcat_image_data.csv")
        
        # excel structure: image_file, category, subcategory, memorability_w_fa_correction
        
        # Read the Excel file
        df = pd.read_csv(excel_path)
        
        # Create train/val/test splits with stratification by subcategory
        train_x, test_val_x = self.create_stratified_splits(df, image_dir)
        valid_x, test_x = self.split_validation_test(test_val_x, image_dir)
        
        super().__init__(train_x=train_x, train_u=None, val=valid_x, test=test_x, num_classes=self._num_classes, lab2cname=self._lab2cname, classnames=self._classnames)
    
    def create_stratified_splits(self, df, image_dir):
        # First split: 70% train, 30% for val+test
        train_df, test_val_df = train_test_split(
            df, 
            test_size=0.3,  # 30% for val+test
            stratify=df['subcategory'],
            random_state=42
        )
        
        # Convert DataFrames to our data structure
        train_data = self.df_to_data_list(train_df, image_dir)
        test_val_data = self.df_to_data_list(test_val_df, image_dir)
        
        return train_data, test_val_data
    
    def split_validation_test(self, test_val_data, image_dir):
        # Create temporary DataFrame for second split
        print(f"test_val_data head: {test_val_data[:5]}, example path: {test_val_data[0].impath}, subcategory: {test_val_data[0].impath.split(os.sep)[-2]}")
        temp_df = pd.DataFrame([
            {
                'path': d.impath, 
                'memorability_w_fa_correction': d.label, 
                'subcategory': d.impath.split(os.sep)[-2], 
                'category': d.impath.split(os.sep)[-3], 
                'image_file': basename(d.impath)
            }
            for d in test_val_data
        ])
        
        # check if any subcategory's count is less than 2
        subcategory_counts = temp_df['subcategory'].value_counts()
        print(f"Subcategory counts: {subcategory_counts}")
        temp_split_df = temp_df[temp_df['subcategory'].isin(subcategory_counts[subcategory_counts >= 2].index)]
        remaining_df = temp_df[~temp_df['subcategory'].isin(subcategory_counts[subcategory_counts >= 2].index)]
        
        # Second split: 1:2 ratio for val:test within the 30% (so ~10% val, ~20% test of total)
        val_df, test_df = train_test_split(
            temp_split_df, 
            test_size=2/3,  # 2/3 of the 30% for test (20% of total)
            stratify=temp_split_df['subcategory'],
            random_state=42
        )
        test_df = pd.concat([test_df, remaining_df], ignore_index=True)
        
        # Convert back to our data structure
        val_data = self.df_to_data_list(val_df, image_dir)
        test_data = self.df_to_data_list(test_df, image_dir)
        
        return val_data, test_data
    
    def df_to_data_list(self, df, image_dir):
        data = []
        
        for _, row in df.iterrows():
            # Construct the file path using category, subcategory, and image_file
            img_path = osp.join(image_dir, row['category'], row['subcategory'], row['image_file'])
            memoriability = float(row['memorability_w_fa_correction'])
            
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            
            data.append(ContinuousLabelDatum(img_path, memoriability))
        
        return data