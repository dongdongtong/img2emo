"""Created by Dingsd on 2024/11/02.
For image emotion valence and arousal prediction.
"""
import time
import datetime
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import monai
from functools import partial

if torch.__version__ >= "2.4":
    from torch.amp import autocast, GradScaler
else:
    from torch.cuda.amp import autocast, GradScaler
    
from copy import deepcopy
from tqdm import tqdm

from ..build import TRAINER_REGISTRY
from ..base_trainer import TrainerX
from optim import build_optimizer, build_lr_scheduler
from models import build_model as build_modeling
from evaluation import build_evaluator

from utils import (
    MetricMeter, AverageMeter, load_pretrained_weights, 
    load_checkpoint, save_checkpoint, 
)

from torchvision.models import maxvit_t
from models.vit.lora_timm_vit import vit_giant_patch14_clip_224_lora

from PIL import Image
import open_clip
from utils.scenes import template, scenes

import loralib as lora


def CCCLoss(x, y):
    # Compute means
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    # Compute variances
    x_var = torch.var(x, dim=0)
    y_var = torch.var(y, dim=0)
    # Compute covariance matrix
    cov_matrix = torch.matmul(
        (x - x_mean).permute(*torch.arange(x.dim() - 1, -1, -1)), y - y_mean
    ) / (x.size(0) - 1)
    # Compute CCC
    numerator = 2 * cov_matrix
    denominator = x_var + y_var + torch.pow((x_mean - y_mean), 2)
    ccc = torch.mean(numerator / denominator)
    return -ccc


def spearman_correlation(predictions, targets):
    """
    Calculate Spearman rank correlation using PyTorch's built-in functions.
    Handles ties properly (matching SciPy's implementation).
    
    Args:
        predictions: tensor of predicted values
        targets: tensor of target values
        
    Returns:
        Spearman rank correlation coefficient
    """
    # Convert to numpy for ranking with proper tie handling
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    # Use scipy's rankdata for proper tie handling
    from scipy.stats import rankdata
    pred_ranks = torch.tensor(rankdata(pred_np), dtype=predictions.dtype, device=predictions.device)
    target_ranks = torch.tensor(rankdata(target_np), dtype=targets.dtype, device=targets.device)
    
    # Calculate the mean of ranks
    pred_ranks_mean = torch.mean(pred_ranks)
    target_ranks_mean = torch.mean(target_ranks)
    
    # Calculate the numerator (covariance of ranks)
    numerator = torch.sum((pred_ranks - pred_ranks_mean) * (target_ranks - target_ranks_mean))
    
    # Calculate the denominator (product of standard deviations of ranks)
    pred_deviation = torch.sqrt(torch.sum((pred_ranks - pred_ranks_mean) ** 2))
    target_deviation = torch.sqrt(torch.sum((target_ranks - target_ranks_mean) ** 2))
    denominator = pred_deviation * target_deviation
    
    # Calculate Spearman correlation
    corr = numerator / denominator
    
    return corr


def differentiable_rank_correlation(predictions, targets, temperature=0.01):
    """
    Differentiable approximation of Spearman correlation.
    
    Args:
        predictions: tensor of predicted values
        targets: tensor of target values
        temperature: controls the smoothness of the approximation
        
    Returns:
        Approximate Spearman correlation coefficient
    """
    n = predictions.size(0)
    
    # Create pairwise difference matrices
    pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
    targ_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    
    # Apply sigmoid to get soft comparisons
    pred_sign = torch.sigmoid(pred_diff / temperature)
    targ_sign = torch.sigmoid(targ_diff / temperature)
    
    # Sum to get approximate ranks (adding 1 for 1-based ranking)
    pred_ranks = torch.sum(pred_sign, dim=1) + 1
    target_ranks = torch.sum(targ_sign, dim=1) + 1
    
    # Calculate the mean of ranks
    pred_ranks_mean = torch.mean(pred_ranks)
    target_ranks_mean = torch.mean(target_ranks)
    
    # Calculate the numerator (covariance of ranks)
    numerator = torch.sum((pred_ranks - pred_ranks_mean) * (target_ranks - target_ranks_mean))
    
    # Calculate the denominator (product of standard deviations of ranks)
    pred_deviation = torch.sqrt(torch.sum((pred_ranks - pred_ranks_mean) ** 2))
    target_deviation = torch.sqrt(torch.sum((target_ranks - target_ranks_mean) ** 2))
    denominator = pred_deviation * target_deviation
    
    # Calculate correlation
    corr = numerator / denominator
    
    return corr


class SpearmanLoss(torch.nn.Module):
    def __init__(self, differentiable=True, temperature=0.01):
        super(SpearmanLoss, self).__init__()
        self.differentiable = differentiable
        self.temperature = temperature
        
    def forward(self, predictions, targets):
        # Calculate Spearman correlation
        if self.differentiable:
            corr = differentiable_rank_correlation(predictions, targets, self.temperature)
        else:
            corr = spearman_correlation(predictions, targets)
        
        # Since optimization typically minimizes a loss,
        # we return negative correlation (we want to maximize correlation)
        return -corr


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


class CLIPModel(object):
    def __init__(self, cfg, device):
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
    def __init__(self, cfg, device):
        super().__init__()
        
        self.cfg = cfg
        
        if cfg.MODEL.NAME == "maxvit_t":
            model = maxvit_t(weights="DEFAULT")
            
            block_channels = model.classifier[3].in_features
            model.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(block_channels),
                nn.Linear(block_channels, block_channels),
                nn.Tanh(),
                nn.Linear(block_channels, 1, bias=False),
            )
        elif cfg.MODEL.NAME == "vit_giant_lora":
            lora_reduction = cfg.TRAINER.MEMCLIPVITLORA.LORA_REDUCTION
            
            model = vit_giant_patch14_clip_224_lora(pretrained=True, pretrained_strict=False, lora_reduction=lora_reduction)
            model.head = None
            
            # Freeze base model parameters
            lora.mark_only_lora_as_trainable(model, bias='all')
            
            num_features = model.num_features + 1024 + 401   # here 1024 means image features from clip, 401 means scene probas
            self.class_head = nn.Sequential(
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(cfg.TRAINER.MEMCLIPVITLORA.HEAD_DROPRATE),
                    nn.Linear(256, 1)
                )
        else:
            raise ValueError(f"Unknown model: {cfg.MODEL.NAME}")
        
        self.model = model
        
        self.clipmodel = CLIPModel(cfg, device)
        self.device = device
    
    def forward(self, x):
        features = self.model.forward_features(x)
        clip_features = self.clipmodel.forward(x)
        x = features[:, self.model.num_prefix_tokens:].mean(dim=1)
        x = torch.cat((x, clip_features["image_features"], clip_features["scene_probs"]), dim=1)
        x = self.class_head(x)
        return x


@TRAINER_REGISTRY.register()
class MemCLIPVitLora(TrainerX):
    """We perform image emotion valence and arousal prediction using giant vit lora.
    We use CCC and MSE losses described in the paper: CAGE: Circumplex Affect Guided Expression Inference.
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MEMCLIPVITLORA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
            
        model = MemModel(cfg, self.device)
        model.to(self.device)
        self.model = model
        
        # Double check needed updated parameters
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        trainable_params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        whole_params = [p.numel() for p in self.model.parameters()]
        print(f"Trainable parameters count: {sum(trainable_params)}, whole parameters count: {sum(whole_params)}, ratio: {(sum(trainable_params) / sum(whole_params) * 100):.4f}%")

        if cfg.MODEL.INIT_WEIGHTS:
            state_dict = load_checkpoint(cfg.MODEL.INIT_WEIGHTS)
            # print(f"Loading pretrained model from {cfg.MODEL.INIT_WEIGHTS}, epoch={state_dict['epoch']}")
            load_pretrained_model_lora_classhead(self.model, state_dict)
            
        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        
        if cfg.TRAINER.MEMCLIPVITLORA.PREC == "amp" or cfg.TRAINER.MEMCLIPVITLORA.PREC == "fp32":
            self.model.float()
        else:
            self.model.half()
            
            # for name, module in self.model.named_modules():
            #     if isinstance(module, nn.LayerNorm):
            #         module.float()
        
        self.scaler = GradScaler() if cfg.TRAINER.MEMCLIPVITLORA.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            
        self.best_result = np.inf  # we do regression, so we use smaller loss as better
        
        self.srcc_criterion = SpearmanLoss(differentiable=True, temperature=cfg.TRAINER.MEMCLIPVITLORA.SRCC_TEMP)
        
        # self.test("val")

    def model_inference(self, image):
        with torch.autocast("cuda"):  # clip's default dtype is float16, so we autocast to float16 in inference
            out = self.model(image)
        return out
    
    def mse_loss(self, images, labels):
        logits = self.model(images)
        logits = logits.view(-1)
        
        mse_loss = F.mse_loss(logits, labels)
        
        loss = mse_loss
        
        meta_loss_dict = {
            "mse_loss": mse_loss,
        }
        
        self.model_backward_and_update(loss, meta_loss_dict, grad_record=True, names=['model', ])  # update all parameters
        
        return {
            "loss": loss.detach(),
            "mse_loss": mse_loss.detach(),
        }
    
    def mse_srcc_loss(self, images, labels):
        logits = self.model(images)
        logits = logits.view(-1)
        
        mse_loss = F.mse_loss(logits, labels)
        
        # Compute Spearman correlation loss
        spearman_corr_loss = self.srcc_criterion(logits, labels)
        
        loss = mse_loss + spearman_corr_loss
        
        meta_loss_dict = {
            "mse_loss": mse_loss,
            "spearman_corr_loss": spearman_corr_loss,
        }
        
        self.model_backward_and_update(loss, meta_loss_dict, grad_record=True, names=['model', ])
        
        return {
            "loss": loss.detach(),
            "mse_loss": mse_loss.detach(),
            "spearman_corr_loss": spearman_corr_loss.detach(),
        }
    
    def mse_ccc_srcc_loss(self, images, labels):
        logits = self.model(images)
        logits = logits.view(-1)
        
        mse_loss = F.mse_loss(logits, labels)
        
        # Compute Spearman correlation loss
        spearman_corr_loss = self.srcc_criterion(logits, labels)
        
        # Compute CCC loss
        ccc_loss = CCCLoss(logits, labels)
        
        loss = mse_loss + spearman_corr_loss + ccc_loss
        
        meta_loss_dict = {
            "mse_loss": mse_loss,
            "spearman_corr_loss": spearman_corr_loss,
            "ccc_loss": ccc_loss,
        }
        
        self.model_backward_and_update(loss, meta_loss_dict, grad_record=True, names=['model', ])
        
        return {
            "loss": loss.detach(),
            "mse_loss": mse_loss.detach(),
            "spearman_corr_loss": spearman_corr_loss.detach(),
            "ccc_loss": ccc_loss.detach(),
        }
    
    def get_loss(self, batch_x):
        images, labels = self.parse_batch_train(batch_x)
        
        if self.cfg.TRAINER.MEMCLIPVITLORA.METHOD == 1:
            loss_dict = self.mse_loss(images, labels)
        elif self.cfg.TRAINER.MEMCLIPVITLORA.METHOD == 2:
            loss_dict = self.mse_srcc_loss(images, labels)
        elif self.cfg.TRAINER.MEMCLIPVITLORA.METHOD == 3:
            loss_dict = self.mse_ccc_srcc_loss(images, labels)
        else:
            raise ValueError(f"Unknown method: {self.cfg.TRAINER.MEMCLIPVITLORA.METHOD}")
        
        return loss_dict
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            # if self.cfg.OPTIM.MAX_EPOCH > 200 and self.epoch % 10 == 0:
            curr_result = self.test(split="val")
            is_best = curr_result < self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
                
                names = self.get_model_names()
                model_dir = self.cfg.OUTPUT_DIR
                for name in names:
                    torch.save(trainable_state_dict(self._models[name]), osp.join(model_dir, name, f'model-best.pth.tar'))

        # if meet_checkpoint_freq or last_epoch:
        #     self.save_model(self.epoch, self.output_dir)

    def forward_backward(self, batch_x):
        prec = self.cfg.TRAINER.MEMCLIPVITLORA.PREC

        if prec == "amp":
            with autocast(device_type="cuda"):
                out_dict = self.get_loss(batch_x)
        else:
            # with torch.autograd.set_detect_anomaly(True):
            out_dict = self.get_loss(batch_x)
        
        loss_summary = deepcopy(out_dict)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
        
    def parse_batch_train(self, batch_x):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        
        input_x = input_x.to(self.device).float()
        label_x = label_x.to(self.device).float()
        
        return input_x, label_x

    def parse_batch_test(self, batch):
        input_x = batch["img"]
        label_x = batch["label"]
        domain_label = batch["domain"]  # strings
        
        input_x = input_x.to(self.device).float()
        label_x = label_x.to(self.device).float()
        
        return input_x, label_x, domain_label

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain_label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label, domain_label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
            
        # we eval on the test set to check its performance
        self.evaluator.reset()
        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            input, label, domain_label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label, domain_label)
        results = self.evaluator.evaluate()
        
        for k, v in results.items():
            tag = f"test/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def model_backward_and_update(self, loss, meta_loss_dict, names=None, grad_record=False, grad_clip=False, grad_tag="g"):
        names = self.get_model_names(names)
        
        accumulate_steps = self.cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS

        if self.cfg.TRAINER.MEMCLIPVITLORA.PREC == "amp":
            loss = torch.Tensor([0.0]).to(self.device)
            for k, meta_subloss in meta_loss_dict.items():
                loss += self.scaler.scale(meta_subloss / accumulate_steps)
            loss.backward()
        else:
            (loss / accumulate_steps).backward()
        
        # gradient clipping
        if grad_clip:
            for name in names:
                torch.nn.utils.clip_grad_norm_(self._models[name].parameters(), max_norm=11.0)
            
        def name_grad_available(name, update_model_names):
            available = False
            for model_name in update_model_names:
                if model_name in name:
                    available = True
                    break
            return available
        
        if self.batch_idx % self.cfg.TRAIN.PRINT_FREQ == 0 and grad_record:
            for name, param in self.model.named_parameters():
                try:
                    if param.requires_grad:
                        scale_ = self.scaler.get_scale() if self.scaler else 1.0
                        self._writer.add_scalar(f"grad_{grad_tag}/{name}", (param.grad / scale_).norm(), global_step=self.epoch * self.num_batches)
                except Exception as e:
                    print(f"Error in recording gradient: {e}, name: {name}")
        
        if (self.batch_idx + 1) % accumulate_steps == 0 or self.batch_idx == self.num_batches - 1:
            if self.cfg.TRAINER.MEMCLIPVITLORA.PREC == "amp":
                for name in names:
                    if self._optims[name] is not None:
                        self.scaler.step(self._optims[name])
                        self._optims[name].zero_grad()
                self.scaler.update()
            else:
                for name in names:
                    if self._optims[name] is not None:
                        self._optims[name].step()
                        self._optims[name].zero_grad()

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            state_dict = load_checkpoint(model_path)
            epoch = 0

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            load_pretrained_model_lora_classhead(self._models[name], state_dict)
            # self._models[name].load_state_dict(state_dict, strict=False)

