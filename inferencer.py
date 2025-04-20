import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from timm.models.resnet import resnet18, resnet50
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image
import os
import os.path as osp
from os.path import join, dirname, basename

from copy import deepcopy


import random
from tqdm import tqdm

import time

import multiprocessing

from torchvision.transforms import TrivialAugmentWide

import loralib as lora

from torchvision.models import maxvit_t
from models.vit.lora_timm_vit import vit_giant_patch14_clip_224_lora

from utils.scenes import scenes, template

import loralib as lora


def trainable_state_dict(model: nn.Module):
    my_state_dict = model.state_dict()

    trainable_state_dict = {k: v for (name, param), (k, v) in zip(model.named_parameters(), my_state_dict.items()) if param.requires_grad}

    return trainable_state_dict


def load_pretrained_model_lora_classhead(model: nn.Module, trainable_state_dict: dict):
    model_dict = model.state_dict()

    # update the model's state_dict with the trainable parameters
    model_dict.update(trainable_state_dict)

    model.load_state_dict(model_dict)

    return model


class VAModel(nn.Module):
    def __init__(self, model_name: str, lora_reduction: int, droprate: float):
        super().__init__()

        if model_name == "vit_giant_patch14_clip_224_lora":
            model = vit_giant_patch14_clip_224_lora(pretrained=True, pretrained_strict=False, lora_reduction=lora_reduction)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze base model parameters
        lora.mark_only_lora_as_trainable(model, bias='all')
        
        num_features = model.num_features
        self.class_head = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(droprate),
                nn.Linear(256, 2)
            )
        
        self.model = model
    
    def forward(self, x):
        features = self.model.forward_features(x)
        x = features[:, self.model.num_prefix_tokens:].mean(dim=1)
        x = self.class_head(x)
        return {
            "va": x,
        }
        

import open_clip
class CLIPModel(object):
    def __init__(self, device):
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
        
        print(f"Loading CLIP model and freezeing parameters")
        # Freeze the model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        scenes_text_feature_path = "/data/dingsd/img2emo/image_memoriability/datasets/scenes401_PlaceDataset.pt"
        if osp.exists(scenes_text_feature_path):
            print(f"Loading text features from {scenes_text_feature_path}")
            scenes_text_features = torch.load(scenes_text_feature_path, weights_only=False, map_location="cpu")
        else:
            print("Calculating text features...")
            tokenizer = open_clip.get_tokenizer('ViT-g-14')
            scenes_texts = tokenizer([template + scene for scene in scenes])
            scenes_text_features = model.encode_text(scenes_texts)  # shape (401, 1024)
            torch.save(scenes_text_features, scenes_text_feature_path)
            print(f"Text features saved to {scenes_text_feature_path}")
        
        self.visual = model.visual.to(device)
        self.visual.eval()
        self.scenes_text_features = scenes_text_features.to(device)

    @torch.no_grad()
    def forward(self, images):
        image_features = self.visual(images)  # shape (bs, 1024)
        text_features = self.scenes_text_features.clone()  # shape (401, 1024)
        
        # normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # print(f"Image features shape: {image_features.shape}, dtype: {image_features.dtype}")
        # print(f"Text features shape: {text_features.shape}, dtype: {text_features.dtype}")
        # calculate similarity
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # shape (bs, 401)
        
        return {
            "image_features": image_features,
            "text_features": text_features,
            "scene_probs": text_probs,
        }
        

class MemModel(nn.Module):
    def __init__(self, device, lora_reduction: int, droprate: float = 0.):
        super().__init__()
        
        model = vit_giant_patch14_clip_224_lora(pretrained=True, pretrained_strict=False, lora_reduction=lora_reduction)
        model.head = None
        
        # Freeze base model parameters
        lora.mark_only_lora_as_trainable(model, bias='all')
        
        num_features = model.num_features + 1024 + 401   # here 1024 means image features from clip, 401 means scene probas
        self.class_head = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(droprate),
                nn.Linear(256, 1)
                )
        
        self.model = model
        
        self.clipmodel = CLIPModel(device)
        self.device = device
    
    def forward(self, x):
        features = self.model.forward_features(x)
        clip_features = self.clipmodel.forward(x)
        x = features[:, self.model.num_prefix_tokens:].mean(dim=1)
        x = torch.cat((x, clip_features["image_features"], clip_features["scene_probs"]), dim=1)
        x = self.class_head(x)
        return x



class VAPredictor:
    def __init__(
        self, 
        pretrained_model="vit_giant_patch14_clip_224_lora", 
        lora_reduction=8, 
        model_path="output/EmoticVitLora/emotic_vit_lora_ep20/EMOTIC_NAPS_H_OASIS_Emotion6/model/model-best.pth.tar",
        droprate=0.,
    ):
        self.model = VAModel(pretrained_model, lora_reduction, droprate)
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.model = load_pretrained_model_lora_classhead(self.model, torch.load(model_path))

        self.model.eval()

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, input_image_path: str) -> dict:
        image = Image.open(input_image_path).convert('RGB')

        image_tensor = self.test_transform(image).unsqueeze(0).to(self.device)

        out_dict = self.model(image_tensor)

        v_float = out_dict["va"][0][0].item()
        a_float = out_dict["va"][0][1].item()

        return {
            "valence": v_float,
            "arousal": a_float,
        }
        
    
class MemPredictor:
    def __init__(
        self, 
        lora_reduction=8, 
        model_path="output/clip_trivialaug_lamem/MemCLIPVitLora/clip_lamem_vit_lora_mse_srcc_ccc_ep20/LaMem/seed1_fold1/model/model-best.pth.tar",
        droprate=0.,
    ):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = MemModel(device, lora_reduction, droprate)
        self.model_path = model_path

        self.device = device
        self.model = self.model.to(self.device)

        self.model = load_pretrained_model_lora_classhead(self.model, torch.load(model_path))

        self.model.eval()

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),  # Notice this is the OpenAI's CLIP mean and std
        ])
    
    @torch.inference_mode()
    def __call__(self, input_image_path: str) -> dict:
        image = Image.open(input_image_path).convert('RGB')

        image_tensor = self.test_transform(image).unsqueeze(0).to(self.device)
        
        with torch.amp.autocast(device_type="cuda"):
            image_memoriability = self.model(image_tensor)

        return image_memoriability


@torch.no_grad()
def inference(
    model: nn.Module, 
    dataset_name: str,
    data_root: str,
    data: pd.DataFrame, 
    test_transforms,
    tta_transforms=None,
    tta_times=0,
    output_results_dir: str = None,
    device=torch.device("cpu")
):
    model.eval()

    prediction_df = pd.DataFrame(columns=["image_filename", "pred_valence", "pred_arousal", "pred_anger", "pred_disgust", "pred_fear", "pred_joy", "pred_sadness", "pred_surprise", "pred_neutral"])

    for row in tqdm(data.itertuples(), total=len(data)):
        if dataset_name == "emotion6":
            img_name = os.path.join(data_root, "images", row.image_filename)
        elif dataset_name == "naps":
            img_name = os.path.join(data_root, "images", row.image_filename + ".jpg")
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}, we cannot infer the image path.")
        
        image = Image.open(img_name).convert('RGB')

        # original image
        image_tensor = test_transforms(image).unsqueeze(0).to(device)
        
        if tta_transforms and tta_times > 0:
            tta_images = []
            set_random_seed(42)
            for _ in range(tta_times):
                tta_image = tta_transforms(image)
                tta_images.append(tta_image)
            tta_images = torch.stack(tta_images).to(device)
            image_tensor = torch.cat([image_tensor, tta_images], dim=0)
        
        out_dict = model(image_tensor)

        va_outputs = out_dict["va"].mean(dim=0)
        v_output = va_outputs[0].item()
        a_output = va_outputs[1].item()

        emotions = out_dict["emotions"].softmax(dim=1).mean(dim=0)
        anger = emotions[0].item()
        disgust = emotions[1].item()
        fear = emotions[2].item()
        joy = emotions[3].item()
        sadness = emotions[4].item()
        surprise = emotions[5].item()
        neutral = emotions[6].item()
        
        prediction_df.loc[len(prediction_df)] = [
            row.image_filename, v_output, a_output, anger, disgust, fear, joy, sadness, surprise, neutral
        ]

    out_csv_path = os.path.join(output_results_dir, dataset_name, f"{dataset_name}_tta{tta_times}_predictions.csv")
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    prediction_df.to_csv(out_csv_path, index=False)

    # Evaluation: we compare the predictions with the ground truth
    if data['valence'].isnull().any() or data['arousal'].isnull().any():
        print("Ground truth values are missing. Skipping evaluation.")
        return
    
    valence = data['valence'].values
    arousal = data['arousal'].values

    valence_pred = prediction_df['pred_valence'].values
    arousal_pred = prediction_df['pred_arousal'].values

    valence_mse = np.mean((valence - valence_pred) ** 2)
    arousal_mse = np.mean((arousal - arousal_pred) ** 2)

    print(f"Valence MSE: {valence_mse:.4f}")
    print(f"Arousal MSE: {arousal_mse:.4f}")
    print(f"Total MSE: {(valence_mse + arousal_mse):.4f}")

    # save concatenated results
    concat_results = pd.merge(data, prediction_df, on="image_filename", how="inner")
    csv_path = os.path.join(output_results_dir, dataset_name, f"{dataset_name}_tta{tta_times}_merged_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    concat_results.to_csv(csv_path, index=False)

    if dataset_name == "naps":
        # we further give the results for each category
        for category in data['Category'].unique():
            category_data = data[data['Category'] == category]
            category_prediction_df = prediction_df[prediction_df['image_filename'].isin(category_data['image_filename'])]

            valence = category_data['valence'].values
            arousal = category_data['arousal'].values

            valence_pred = category_prediction_df['pred_valence'].values
            arousal_pred = category_prediction_df['pred_arousal'].values

            valence_mse = np.mean((valence - valence_pred) ** 2)
            arousal_mse = np.mean((arousal - arousal_pred) ** 2)

            print("-------------------------------------------")
            print(f"{category} Valence MSE: {valence_mse:.4f}")
            print(f"{category} Arousal MSE: {arousal_mse:.4f}")
            print(f"{category} Total MSE: {(valence_mse + arousal_mse):.4f}")

            # save concatenated results
            concat_results = pd.merge(category_data, category_prediction_df, on="image_filename", how="inner")
            csv_path = os.path.join(output_results_dir, dataset_name, f"{dataset_name}_tta{tta_times}_{category}_merged_results.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            concat_results.to_csv(csv_path, index=False)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trainable_state_dict(model: nn.Module):
    my_state_dict = model.state_dict()

    trainable_state_dict = {k: v for (name, param), (k, v) in zip(model.named_parameters(), my_state_dict.items()) if param.requires_grad}

    return trainable_state_dict


def load_pretrained_model_lora_classhead(model: nn.Module, trainable_state_dict: dict):
    model_dict = model.state_dict()

    # update the model's state_dict with the trainable parameters
    model_dict.update(trainable_state_dict)

    model.load_state_dict(model_dict)

    return model


# 5. Main pipeline (updated)
def main():
    import argparse
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument("--pretrained_model", type=str, default="rn50", help="Pretrained model to use (rn50 or swin_tiny)")
    parser.add_argument("--lora_reduction", type=int, default=8, help="LOcal Rank Attention reduction factor")
    parser.add_argument("--model_path", type=str, help="Path to the pretrained model")
    # dataset arguments
    parser.add_argument("--dataset_name", type=str, default="emotion6", help="Dataset to use (emotion6 or naps)")
    parser.add_argument("--root_dir", type=str, help="Path to the root directory containing images")
    # evaluation arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--image_size", type=int, default=224, help="Size of the input image")
    parser.add_argument("--out_results_dir", type=str, help="Path to the output directory to save the results")
    parser.add_argument("--tta_times", type=int, default=1, help="Number of times to apply test-time augmentation, 0 for no TTA")
    args = parser.parse_args()

    set_random_seed(args.seed)

    # Load and preprocess data
    root_dir = args.root_dir  # Update this with the actual path
    if args.dataset_name == "emotion6":
        data_gt_path = join(root_dir, "ground_truth.txt")
        data = pd.read_csv(data_gt_path, sep="\t", header=0, names=["image_filename", "valence", "arousal", "anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"])
    elif args.dataset_name == "naps":
        data_gt_path = join(root_dir, "ground_truth_v_a.xlsx")
        data = pd.read_excel(data_gt_path, header=0, names=["image_filename", "Category", "Nr", "V/H", "Description", "valence", "arousal"])
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    
    # Define transforms
    image_size = args.image_size

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tta_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(image_size, scale=(0.08, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # out directory
    out_dir = args.out_results_dir
    out_csv_path = os.path.join(out_dir, args.dataset_name, f"{args.dataset_name}_tta{args.tta_times}_predictions.csv")
    if os.path.exists(out_dir) and os.path.exists(out_csv_path):
        raise ValueError(f"Output directory {out_dir} and prediction csv {out_csv_path} already exists. Please specify a new directory.")
    os.makedirs(out_dir, exist_ok=True)
    
    set_random_seed(args.seed)

    model = VAModel(model_name=args.pretrained_model, lora_reduction=args.lora_reduction, droprate=0.).to(device)
    print(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    set_random_seed(args.seed)
    
    model_path = args.model_path
    model = load_pretrained_model_lora_classhead(model, torch.load(model_path))

    inference(
        model, 
        args.dataset_name,
        root_dir,
        data,
        test_transform,
        tta_transform,
        args.tta_times,
        args.out_results_dir,
        device)

if __name__ == "__main__":
    main()