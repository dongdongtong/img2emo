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
class VADataset(DatasetBase):
    # generate data list and saved!
    # fuse multiple dataset
    
    _lab2cname = {
        0: 'valence',
        1: 'arousal',
    }
    
    _classnames = ['valence', 'arousal']
    
    _num_classes = 2
    
    # EMOTIC
    dataset_train_csv = "annot_arrs_train.csv"
    dataset_val_csv = "annot_arrs_val.csv"
    dataset_test_csv = "annot_arrs_test.csv"
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        random_seed = cfg.SEED
        
        source_domain = cfg.DATASET.SOURCE_DOMAINS[0]
        source_domains = "_".join(cfg.DATASET.SOURCE_DOMAINS)
        
        train_x = []
        val_x = []
        
        if "EMOTIC" in source_domains:  # VA 1-10 default
            # EMOTIC data
            emotic_train_df = pd.read_csv(osp.join(cfg.DATASET.ROOT, "EMOTIC", "annots_arrs", self.dataset_train_csv))
            emotic_val_df = pd.read_csv(osp.join(cfg.DATASET.ROOT, "EMOTIC", "annots_arrs", self.dataset_val_csv))
            emotic_test_df = pd.read_csv(osp.join(cfg.DATASET.ROOT, "EMOTIC", "annots_arrs", self.dataset_test_csv))
            
            emotic_train_x = self.read_data_EMOTIC(emotic_train_df, "EMOTIC")
            emotic_val_x = self.read_data_EMOTIC(emotic_val_df, "EMOTIC")
            emotic_test_x = self.read_data_EMOTIC(emotic_test_df, "EMOTIC")
            print(f"emotic_train_x: {len(emotic_train_x)}, emotic_val_x: {len(emotic_val_x)}, emotic_test_x: {len(emotic_test_x)}")
            
            train_x += (emotic_train_x + emotic_val_x)
            val_x += emotic_test_x
        
        if "NAPS_H" in source_domains:   # VA 1-9 default
            naps_data_gt_path = join(cfg.DATASET.ROOT, "NAPS_H", "ground_truth_v_a.xlsx")
            naps_data_df = pd.read_excel(naps_data_gt_path, header=0, names=["image_filename", "Category", "Nr", "V/H", "Description", "valence", "arousal"])
            naps_data = self.read_data_NAP_S(naps_data_df, "NAPS_H")
            
            naps_train_x, naps_val_x = train_test_split(naps_data, test_size=0.2, random_state=random_seed)
            print(f"naps_train_x: {len(naps_train_x)}, naps_val_x: {len(naps_val_x)}")
            
            train_x += naps_train_x
            val_x += naps_val_x

        if "EmoMadrid" in source_domains:  # VA -2~2 default
            emomadrid_data_df = pd.read_excel(
                join(cfg.DATASET.ROOT, "EmoMadrid", "EMindex_18May_2023.xlsx"),
                header=1
                )
            emomadrid_data_df = emomadrid_data_df[["EM CODE", "Mean Valence", "Mean Arousal"]]
            emomadrid_data_df.columns = ["image_filename", "valence", "arousal"]
            emomadrid_data = self.read_data_EmoMadrid(emomadrid_data_df, "EmoMadrid")
            
            emomadrid_train_x, emomadrid_val_x = train_test_split(emomadrid_data, test_size=0.2, random_state=random_seed)
            print(f"emomadrid_train_x: {len(emomadrid_train_x)}, emomadrid_val_x: {len(emomadrid_val_x)}")
            
            train_x += emomadrid_train_x
            val_x += emomadrid_val_x
        
        if "GAPED" in source_domains:  # VA 0~100 default
            gaped_A_df = pd.read_csv(join(cfg.DATASET.ROOT, "GAPED", "A.txt"), sep='\t')
            gaped_A_df = gaped_A_df.iloc[:, [0, 1, 2]]
            gaped_A_df.columns = ['image_filename', 'valence', 'arousal']
            gaped_A_data = self.read_data_GAPED(gaped_A_df, "GAPED", "A")
            gaped_A_train_x, gaped_A_val_x = train_test_split(gaped_A_data, test_size=0.2, random_state=random_seed)
            
            gaped_H_df = pd.read_csv(join(cfg.DATASET.ROOT, "GAPED", "H.txt"), sep='\t')
            gaped_H_df = gaped_H_df.iloc[:, [0, 1, 2]]
            gaped_H_df.columns = ['image_filename', 'valence', 'arousal']
            gaped_H_data = self.read_data_GAPED(gaped_H_df, "GAPED", "H")
            gaped_H_train_x, gaped_H_val_x = train_test_split(gaped_H_data, test_size=0.2, random_state=random_seed)
            
            gaped_N_df = pd.read_csv(join(cfg.DATASET.ROOT, "GAPED", "N.txt"), sep='\t')
            gaped_N_df = gaped_N_df.iloc[:, [0, 1, 2]]
            gaped_N_df.columns = ['image_filename', 'valence', 'arousal']
            gaped_N_data = self.read_data_GAPED(gaped_N_df, "GAPED", "N")
            gaped_N_train_x, gaped_N_val_x = train_test_split(gaped_N_data, test_size=0.2, random_state=random_seed)
            
            gaped_P_df = pd.read_csv(join(cfg.DATASET.ROOT, "GAPED", "P.txt"), sep='\t')
            gaped_P_df = gaped_P_df.iloc[:, [0, 1, 2]]
            gaped_P_df.columns = ['image_filename', 'valence', 'arousal']
            gaped_P_data = self.read_data_GAPED(gaped_P_df, "GAPED", "P")
            gaped_P_train_x, gaped_P_val_x = train_test_split(gaped_P_data, test_size=0.2, random_state=random_seed)
            
            gaped_Sn_df = pd.read_csv(join(cfg.DATASET.ROOT, "GAPED", "Sn.txt"), sep='\t')
            gaped_Sn_df = gaped_Sn_df.iloc[:, [0, 1, 2]]
            gaped_Sn_df.columns = ['image_filename', 'valence', 'arousal']
            gaped_Sn_data = self.read_data_GAPED(gaped_Sn_df, "GAPED", "Sn")
            gaped_Sn_train_x, gaped_Sn_val_x = train_test_split(gaped_Sn_data, test_size=0.2, random_state=random_seed)
            
            gaped_Sp_df = pd.read_csv(join(cfg.DATASET.ROOT, "GAPED", "Sp.txt"), sep='\t')
            gaped_Sp_df = gaped_Sp_df.iloc[:, [0, 1, 2]]
            gaped_Sp_df.columns = ['image_filename', 'valence', 'arousal']
            gaped_Sp_data = self.read_data_GAPED(gaped_Sp_df, "GAPED", "Sp")
            gaped_Sp_train_x, gaped_Sp_val_x = train_test_split(gaped_Sp_data, test_size=0.2, random_state=random_seed)
            
            gaped_train_x = gaped_A_train_x + gaped_H_train_x + gaped_N_train_x + gaped_P_train_x + gaped_Sn_train_x + gaped_Sp_train_x
            gaped_val_x = gaped_A_val_x + gaped_H_val_x + gaped_N_val_x + gaped_P_val_x + gaped_Sn_val_x + gaped_Sp_val_x
            print(f"gaped_train_x: {len(gaped_train_x)}, gaped_val_x: {len(gaped_val_x)}")
            train_x += gaped_train_x
            val_x += gaped_val_x
        
        if "OASIS" in source_domains:  # VA 1~7 default
            # OASIS data
            oasis_data_df = pd.read_csv(join(cfg.DATASET.ROOT, "OASIS", 'OASIS.csv'))
            oasis_data_df.columns = ['image_no', 'image_filename', 'Category', 'Source', 'Valence_mean', 'Valence_SD', 'Valence_N', 'Arousal_mean', 'Arousal_SD', 'Arousal_N']
            oasis_data = self.read_data_OASIS(oasis_data_df, "OASIS")
            
            oasis_train_x, oasis_val_x = train_test_split(oasis_data, test_size=0.2, random_state=random_seed)
            print(f"oasis_train_x: {len(oasis_train_x)}, oasis_val_x: {len(oasis_val_x)}")

            train_x += oasis_train_x
            val_x += oasis_val_x
        
        # Emotion6 data
        emotion6_data_df = pd.read_csv(join(cfg.DATASET.ROOT, "Emotion6", 'ground_truth.txt'), sep='\t')
        emotion6_data_df.columns = ['image_filename', 'valence', 'arousal', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
        emotion6_data = self.read_data_Emotion6(emotion6_data_df, "Emotion6")
        
        emotion6_train_x, emotion6_val_x = train_test_split(emotion6_data, test_size=0.2, random_state=random_seed)
        print(f"emotion6_train_x: {len(emotion6_train_x)}, emotion6_val_x: {len(emotion6_val_x)}")
        
        if "Emotion6" in source_domains:
            train_x += emotion6_train_x
        
        test_x = emotion6_val_x
        
        super().__init__(train_x=train_x, train_u=None, val=val_x, test=test_x, num_classes=self._num_classes, lab2cname=self._lab2cname, classnames=self._classnames)
    
    def read_data_EMOTIC(self, data_df: pd.DataFrame, domain: str) -> List[MultiLabelDatum]:
        data = []
        
        for row in data_df.itertuples():
            img_path = osp.join(self.cfg.DATASET.ROOT, domain, "img_arrs", f"{row.Crop_name}")
            valence = row.Valence
            arousal = row.Arousal
            
            label = [valence, arousal]
            
            data.append(MultiLabelDatum(img_path, label, domain))
        
        return data
    
    def read_data_NAP_S(self, data_df: pd.DataFrame, domain: str) -> List[MultiLabelDatum]:
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
    
    def read_data_Emotion6(self, data_df: pd.DataFrame, domain: str) -> List[MultiLabelDatum]:
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
    
    def read_data_EmoMadrid(self, data_df: pd.DataFrame, domain: str) -> List[MultiLabelDatum]:
        data = []
        
        for row in data_df.itertuples():
            
            # notice EmoMadrid data EM0083.JPG has to be EM0083.jpg
            img_path = os.path.join(self.cfg.DATASET.ROOT, domain, row.image_filename.strip() + ".jpg")
            if not os.path.exists(img_path) and os.path.exists(img_path.replace(".jpg", ".JPG")):
                img_path = img_path.replace(".jpg", ".JPG")
            
            if not os.path.exists(img_path):
                print(f"img_path {img_path} does not exist")
                continue
            
            valence = row.valence   # default value range: -2~2
            arousal = row.arousal   # default value range: -2~2
            
            # rescale value range into 1~10
            valence = (valence + 2) / 4 * 9 + 1
            arousal = (arousal + 2) / 4 * 9 + 1
            
            label = [valence, arousal]
            
            data.append(MultiLabelDatum(img_path, label, domain))
        
        return data
    
    def read_data_GAPED(self, data_df: pd.DataFrame, domain: str, image_dir_name:str) -> List[MultiLabelDatum]:
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
    
    def read_data_OASIS(self, data_df: pd.DataFrame, domain: str) -> List[MultiLabelDatum]:
        data = []
        
        for row in data_df.itertuples():
            
            img_path = os.path.join(self.cfg.DATASET.ROOT, domain, "images", row.image_filename.strip() + ".jpg")
            
            valence = row.Valence_mean  # default value range 1~7
            arousal = row.Arousal_mean  # default value range 1~7
            
            # rescale value range into 1-10
            valence = (valence - 1) / 6 * 9 + 1
            arousal = (arousal - 1) / 6 * 9 + 1
            
            label = [valence, arousal]
            
            data.append(MultiLabelDatum(img_path, label, domain))
        
        return data