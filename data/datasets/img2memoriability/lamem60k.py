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
class LaMem(DatasetBase):
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
        
        image_dir = osp.join(cfg.DATASET.ROOT, source_domain, "lamem", "images")
        splits_dir = osp.join(cfg.DATASET.ROOT, source_domain, "lamem", "splits")
        
        fold = cfg.DATASET.FOLD
        train_split = osp.join(splits_dir, f"train_{fold}.txt")
        val_split = osp.join(splits_dir, f"val_{fold}.txt")
        test_split = osp.join(splits_dir, f"test_{fold}.txt")
        
        train_x = self.read_data(train_split, image_dir)
        valid_x = self.read_data(val_split, image_dir)
        test_x = self.read_data(test_split, image_dir)
        
        super().__init__(train_x=train_x, train_u=None, val=valid_x, test=test_x, num_classes=self._num_classes, lab2cname=self._lab2cname, classnames=self._classnames)
    
    def read_data(self, data_split: str, image_dir: str) -> List[ContinuousLabelDatum]:
        
        # read txt using pandas, note that the txt have no headers
        data_df = pd.read_csv(data_split, sep=" ", header=None)
        
        data = []
        
        for row in data_df.itertuples():
            # print(row)
            img_path = osp.join(image_dir, row._1)
            memoriability = float(row._2)
            
            data.append(ContinuousLabelDatum(img_path, memoriability))
        
        return data