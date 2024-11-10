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

import pandas as pd

from typing import Dict, List

from utils import read_json, mkdir_if_missing, write_json


@DATASET_REGISTRY.register()
class GAPED(DatasetBase):
    """GAPED dataset for emotion recognition."""
    
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
        
        gaped_A_df = pd.read_csv(join(cfg.DATASET.ROOT, source_domain, "A.txt"), sep='\t')
        gaped_A_df = gaped_A_df.iloc[:, [0, 1, 2]]
        gaped_A_df.columns = ['image_filename', 'valence', 'arousal']
        gaped_A_data = self.read_data(gaped_A_df, source_domain, "A")
        gaped_A_train_x, gaped_A_val_x = train_test_split(gaped_A_data, test_size=0.2, random_state=random_seed)
        
        gaped_H_df = pd.read_csv(join(cfg.DATASET.ROOT, source_domain, "H.txt"), sep='\t')
        gaped_H_df = gaped_H_df.iloc[:, [0, 1, 2]]
        gaped_H_df.columns = ['image_filename', 'valence', 'arousal']
        gaped_H_data = self.read_data(gaped_H_df, source_domain, "H")
        gaped_H_train_x, gaped_H_val_x = train_test_split(gaped_H_data, test_size=0.2, random_state=random_seed)
        
        gaped_N_df = pd.read_csv(join(cfg.DATASET.ROOT, source_domain, "N.txt"), sep='\t')
        gaped_N_df = gaped_N_df.iloc[:, [0, 1, 2]]
        gaped_N_df.columns = ['image_filename', 'valence', 'arousal']
        gaped_N_data = self.read_data(gaped_N_df, source_domain, "N")
        gaped_N_train_x, gaped_N_val_x = train_test_split(gaped_N_data, test_size=0.2, random_state=random_seed)
        
        gaped_P_df = pd.read_csv(join(cfg.DATASET.ROOT, source_domain, "P.txt"), sep='\t')
        gaped_P_df = gaped_P_df.iloc[:, [0, 1, 2]]
        gaped_P_df.columns = ['image_filename', 'valence', 'arousal']
        gaped_P_data = self.read_data(gaped_P_df, source_domain, "P")
        gaped_P_train_x, gaped_P_val_x = train_test_split(gaped_P_data, test_size=0.2, random_state=random_seed)
        
        gaped_Sn_df = pd.read_csv(join(cfg.DATASET.ROOT, source_domain, "Sn.txt"), sep='\t')
        gaped_Sn_df = gaped_Sn_df.iloc[:, [0, 1, 2]]
        gaped_Sn_df.columns = ['image_filename', 'valence', 'arousal']
        gaped_Sn_data = self.read_data(gaped_Sn_df, source_domain, "Sn")
        gaped_Sn_train_x, gaped_Sn_val_x = train_test_split(gaped_Sn_data, test_size=0.2, random_state=random_seed)
        
        gaped_Sp_df = pd.read_csv(join(cfg.DATASET.ROOT, source_domain, "Sp.txt"), sep='\t')
        gaped_Sp_df = gaped_Sp_df.iloc[:, [0, 1, 2]]
        gaped_Sp_df.columns = ['image_filename', 'valence', 'arousal']
        gaped_Sp_data = self.read_data(gaped_Sp_df, source_domain, "Sp")
        gaped_Sp_train_x, gaped_Sp_val_x = train_test_split(gaped_Sp_data, test_size=0.2, random_state=random_seed)
        
        gaped_train_x = gaped_A_train_x + gaped_H_train_x + gaped_N_train_x + gaped_P_train_x + gaped_Sn_train_x + gaped_Sp_train_x
        gaped_val_x = gaped_A_val_x + gaped_H_val_x + gaped_N_val_x + gaped_P_val_x + gaped_Sn_val_x + gaped_Sp_val_x
        print(f"gaped_train_x: {len(gaped_train_x)}, gaped_val_x: {len(gaped_val_x)}")
        
        super().__init__(train_x=gaped_train_x, train_u=None, val=gaped_val_x, test=gaped_val_x, num_classes=self._num_classes, lab2cname=self._lab2cname, classnames=self._classnames)
            
    def __len__(self):
        return len(self.indices)
    
    def read_data(self, data_df: pd.DataFrame, domain: str, image_dir_name:str) -> List[MultiLabelDatum]:
        data = []
        
        for row in data_df.itertuples():
            
            img_path = os.path.join(self.cfg.DATASET.ROOT, domain, image_dir_name, row.image_filename.split(".")[0] + ".bmp")
            
            valence = row.valence   # default value range 0~100
            arousal = row.arousal   # default value range 0~100
            
            # rescale value range into 1-10
            valence = valence / 100 * 9 + 1
            arousal = arousal / 100 * 9 + 1
            
            label = [valence, arousal]
            
            data.append(MultiLabelDatum(img_path, label, domain))
        
        return data
