# Created by: @luckydog on 2024-11-02
# This is a dataset for loading EMOTION6 dataset.


"""Created by Dingsd on 11/02/2024 22:21
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

from ..base_dataset import DatasetBase, PairDatum, MultiLabelDatum
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
class NAPS_H(DatasetBase):
    # generate data list and saved!
    # make a data structure for hematoma segmentation
    # train val test split
    
    _lab2cname = {
        0: 'valence',
        1: 'arousal',
    }
    
    _classnames = ['valence', 'arousal']
    
    _num_classes = 2
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        source_domain = cfg.DATASET.SOURCE_DOMAINS[0]
        
        data_gt_path = join(cfg.DATASET.ROOT, source_domain, "ground_truth_v_a.xlsx")
        data_df = pd.read_excel(data_gt_path, header=0, names=["image_filename", "Category", "Nr", "V/H", "Description", "valence", "arousal"])
        
        self.image_dir = join(cfg.DATASET.ROOT, source_domain, 'images')
        
        train_x = self.read_data(data_df, source_domain)
        
        super().__init__(train_x=train_x, train_u=None, val=train_x, test=train_x, num_classes=self._num_classes, lab2cname=self._lab2cname, classnames=self._classnames)
    
    def read_data(self, data_df: pd.DataFrame, domain: str) -> List[MultiLabelDatum]:
        data = []
        
        for row in data_df.itertuples():
            
            img_path = os.path.join(self.cfg.DATASET.ROOT, domain, "images", row.image_filename + ".jpg")
            
            valence = row.valence   # default value range 1-9
            arousal = row.arousal   # default value range 1-9
            
            # rescale value range into 1-10
            valence = (valence - 1) / 8 * 9 + 1
            arousal = (arousal - 1) / 8 * 9 + 1
            
            label = [valence, arousal]
            
            data.append(MultiLabelDatum(img_path, label, domain))
        
        return data