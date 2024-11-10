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
class Emotic(DatasetBase):
    # generate data list and saved!
    # make a data structure for hematoma segmentation
    # train val test split
    
    _lab2cname = {
        0: 'valence',
        1: 'arousal',
    }
    
    _classnames = ['valence', 'arousal']
    
    _num_classes = 2
    
    dataset_train_csv = "annot_arrs_train.csv"
    dataset_val_csv = "annot_arrs_val.csv"
    dataset_test_csv = "annot_arrs_test.csv"
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        source_domain = cfg.DATASET.SOURCE_DOMAINS[0]
        
        train_df = pd.read_csv(osp.join(cfg.DATASET.ROOT, source_domain, "annots_arrs", self.dataset_train_csv))
        val_df = pd.read_csv(osp.join(cfg.DATASET.ROOT, source_domain, "annots_arrs", self.dataset_val_csv))
        test_df = pd.read_csv(osp.join(cfg.DATASET.ROOT, source_domain, "annots_arrs", self.dataset_test_csv))
        
        train_x = self.read_data(train_df, source_domain)
        valid_x = self.read_data(val_df, source_domain)
        test_x = self.read_data(test_df, source_domain)
        
        super().__init__(train_x=train_x, train_u=None, val=valid_x, test=test_x, num_classes=self._num_classes, lab2cname=self._lab2cname, classnames=self._classnames)
    
    def read_data(self, data_df: pd.DataFrame, domain: str) -> List[MultiLabelDatum]:
        data = []
        
        for row in data_df.itertuples():
            img_path = osp.join(self.cfg.DATASET.ROOT, domain, "img_arrs", f"{row.Crop_name}")
            valence = row.Valence
            arousal = row.Arousal
            
            label = [valence, arousal]
            
            data.append(MultiLabelDatum(img_path, label, domain))
        
        return data
        


@DATASET_WRAPPER_REGISTRY.register()
class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, base_transform=None, other_transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.base_transform = base_transform  # accept list (tuple) as input
        self.other_transform = other_transform  # accept list (tuple) as input
        self.is_train = is_train
        
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.other_transform is not None:
            if isinstance(self.other_transform, (list, tuple)):
                for i, tfm in enumerate(self.other_transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.other_transform, img0)
                output["img"] = img
        else:
            output["img"] = self.base_transform(img0)

        if self.return_img0:
            output["img0"] = self.base_transform(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img = tfm(img0)

        return img