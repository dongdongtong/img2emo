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

from sklearn.model_selection import train_test_split

from ..base_dataset import DatasetBase, PairDatum, MultiLabelDatum
from ..build import DATASET_REGISTRY, DATASET_WRAPPER_REGISTRY
from utils.tools import read_image
    
import os
import os.path as osp
import random
from copy import deepcopy

from glob import glob

import pandas as pd

from typing import Dict, List

from utils import read_json, mkdir_if_missing, write_json


@DATASET_REGISTRY.register()
class Xiecheng(DatasetBase):
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
        
        random_seed = cfg.SEED
        
        source_domain = cfg.DATASET.SOURCE_DOMAINS[0]
        
        data = list(glob(join(cfg.DATASET.ROOT, source_domain, "*.png")))
        
        data_df = pd.read_csv(join(cfg.DATASET.ROOT, source_domain, 'ground_truth.txt'), sep='\t')
        data_df.columns = ['image_filename', 'valence', 'arousal', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
        
        emotion6_data = self.read_data(data_df, source_domain)
        
        emotion6_train_x, emotion6_val_x = train_test_split(emotion6_data, test_size=0.2, random_state=random_seed)
        print(f"emotion6_train_x: {len(emotion6_train_x)}, emotion6_val_x: {len(emotion6_val_x)}")
        
        super().__init__(train_x=emotion6_train_x, train_u=None, val=emotion6_val_x, test=emotion6_val_x, num_classes=self._num_classes, lab2cname=self._lab2cname, classnames=self._classnames)
    
    def read_data(self, data_df: pd.DataFrame, domain: str) -> List[MultiLabelDatum]:
        data = []
        
        for row in data_df.itertuples():
            
            img_path = os.path.join(self.cfg.DATASET.ROOT, domain, "images", row.image_filename)
            
            valence = row.valence  # default range 1~9
            arousal = row.arousal  # default range 1~9
            
            # rescale value range into 1-10
            valence = (valence - 1) / 8 * 9 + 1
            arousal = (arousal - 1) / 8 * 9 + 1
            
            label = [valence, arousal]
            
            data.append(MultiLabelDatum(img_path, label, domain))
        
        return data