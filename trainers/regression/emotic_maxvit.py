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

from torch.cuda.amp import GradScaler, autocast
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


class VAModel(nn.Module):
    def __init__(self, cfg, ):
        super().__init__()
        
        if cfg.MODEL.NAME == "maxvit_t":
            model = maxvit_t(weights="DEFAULT")
            
            block_channels = model.classifier[3].in_features
            model.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(block_channels),
                nn.Linear(block_channels, block_channels),
                nn.Tanh(),
                nn.Linear(block_channels, 2, bias=False),
            )
        else:
            raise ValueError(f"Unknown model: {cfg.MODEL.NAME}")
        
        self.model = model
    
    def forward(self, x):
        return self.model(x)


@TRAINER_REGISTRY.register()
class EmoticMaxVit(TrainerX):
    """We perform image emotion valence and arousal prediction using MaxVit.
    As described in the paper: CAGE: Circumplex Affect Guided Expression Inference.
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.EMOTICMAXVIT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        model = VAModel(cfg)
        model.to(self.device)
        self.model = model
        
        # Double check needed updated parameters
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        trainable_params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable parameters count: {sum(trainable_params)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            
        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        
        if cfg.TRAINER.EMOTICMAXVIT.PREC == "amp" or cfg.TRAINER.EMOTICMAXVIT.PREC == "fp32":
            self.model.float()
        else:
            self.model.half()
            
            # for name, module in self.model.named_modules():
            #     if isinstance(module, nn.LayerNorm):
            #         module.float()
        
        self.scaler = GradScaler() if cfg.TRAINER.EMOTICMAXVIT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            
        self.best_result = np.inf  # we do regression, so we use smaller loss as better

    def model_inference(self, image):
        out = self.model(image)
        return out
    
    def ccc_mse_loss(self, images, labels):
        logits = self.model(images)
        
        ccc_loss = CCCLoss(logits[:, 0], labels[:, 0]) + CCCLoss(logits[:, 1], labels[:, 1])
        mse_loss = F.mse_loss(logits[:, 0], labels[:, 0]) + F.mse_loss(logits[:, 1], labels[:, 1])
        
        loss = ccc_loss + 3 * mse_loss
        
        self.model_backward_and_update(loss, grad_record=True, names=['model', ])  # update all parameters
        
        return {
            "loss": loss.detach(),
            "ccc_loss": ccc_loss.detach(),
            "mse_loss": mse_loss.detach(),
        }
    
    def get_loss(self, batch_x):
        images, labels = self.parse_batch_train(batch_x)
        
        if self.cfg.TRAINER.EMOTICMAXVIT.METHOD == 1:
            loss_dict = self.ccc_mse_loss(images, labels)
        else:
            raise ValueError(f"Unknown method: {self.cfg.TRAINER.EMOTICMAXVIT.METHOD}")
        
        return loss_dict

    def forward_backward(self, batch_x):
        prec = self.cfg.TRAINER.EMOTICMAXVIT.PREC

        if prec == "amp":
            with autocast():
                out_dict = self.get_loss(batch_x)
        else:
            # with torch.autograd.set_detect_anomaly(True):
            out_dict = self.get_loss(batch_x)
        
        loss_summary = deepcopy(out_dict)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            if self.cfg.OPTIM.MAX_EPOCH > 200 and self.epoch % 10 == 0:
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

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
    
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

        return list(results.values())[0]
    
    def parse_batch_train(self, batch_x):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        label_x = [y_.unsqueeze(1) for y_ in label_x]
        label_x = torch.cat(label_x, dim=1)
        
        input_x = input_x.to(self.device).float()
        label_x = label_x.to(self.device).float()
        
        return input_x, label_x

    def parse_batch_test(self, batch):
        input_x = batch["img"]
        label_x = batch["label"]
        domain_label = batch["domain"]  # strings
        label_x = [y_.unsqueeze(1) for y_ in label_x]
        label_x = torch.cat(label_x, dim=1)
        
        input_x = input_x.to(self.device).float()
        label_x = label_x.to(self.device).float()
        
        return input_x, label_x, domain_label

    def model_backward_and_update(self, loss, names=None, grad_record=False, grad_clip=False, grad_tag="g"):
        names = self.get_model_names(names)
        
        accumulate_steps = self.cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS

        if self.cfg.TRAINER.EMOTICMAXVIT.PREC == "amp":
            self.scaler.scale(loss / accumulate_steps).backward()
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
                        self._writer.add_scalar(f"grad_{grad_tag}/{name}", param.grad.norm(), global_step=self.epoch * self.num_batches)
                except Exception as e:
                    print(f"Error in recording gradient: {e}, name: {name}")
        
        if (self.batch_idx + 1) % accumulate_steps == 0 or self.batch_idx == self.num_batches - 1:
            if self.cfg.TRAINER.EMOTICMAXVIT.PREC == "amp":
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

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

