from ..build import EVALUATOR_REGISTRY, EvaluatorBase

from collections import defaultdict, OrderedDict

import numpy as np
from medpy.metric import dc, asd, assd, hd95
from sklearn.metrics import f1_score, confusion_matrix, precision_score

import monai
from monai.data import MetaTensor
from monai.transforms.utils import allow_missing_keys_mode
from data.transforms import build_transform


def get_spacing_from_affine(affine):
    voxel_spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return voxel_spacing


@EVALUATOR_REGISTRY.register()
class HEGrowthSegmentation(EvaluatorBase):
    """Evaluator for Hematoma Growth Prediction."""

    def __init__(self, cfg, lab2cname, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._per_class_res = None
        self._samples = 0
        
        self._dices = []               # dice of total area on 24h hematoma mask
        self._growth_area_dices = []   # dice of growth area
        self._growth_volume_maes = []  # mean absolute error in volume
        self._growth_volume_mes = []   # mean error in volume
        self._asds = []
        self._assds = []
        self._hd95s = []
        self._precisions = []
        
        self._per_class_res = defaultdict(lambda: defaultdict(list))
        
        self.need_other_metrics = cfg.TEST.OTHER_METRICS
        
        self.spacing = kwargs.get("spacing", None)
        
        self.valid_tfs = build_transform(cfg, is_train=False, only_other_transforms=False)

    def reset(self):
        self._samples = 0
        
        self._dices = []
        self._growth_area_dices = []
        self._growth_volume_maes = []
        self._asds = []
        self._assds = []
        self._hd95s = []
        self._precisions = []

        self._per_class_res = defaultdict(lambda: defaultdict(list))
    
    def invert_transform(self, pred_mask, baseline_mask: MetaTensor, followup_mask: MetaTensor):
        # pred_mask: [B, H, W, ...], torch.Tensor
        # baseline_mask: [B, H, W, ...], monai.data.MetaTensor
        # followup_mask: [B, H, W, ...], monai.data.MetaTensor
        baseline_mask = baseline_mask.cpu()
        followup_mask = followup_mask.cpu()
        
        inverted_pred_masks = []
        inverted_baseline_masks = []
        inverted_followup_masks = []
        
        for pred_mask_, baseline_mask_, followup_mask_ in zip(pred_mask, baseline_mask, followup_mask):
            
            pred_mask_metatensor = MetaTensor(pred_mask_, affine=followup_mask_.affine, applied_operations=followup_mask_.applied_operations)
            temp_dict = {"seg_2": pred_mask_metatensor}
            with allow_missing_keys_mode():
                inverted_dict = self.valid_tfs.inverse(temp_dict)
            inverted_pred_masks.append(inverted_dict["seg_2"])
            
            temp_dict = {"seg_2": baseline_mask_}
            with allow_missing_keys_mode():
                inverted_dict = self.valid_tfs.inverse(temp_dict)
            inverted_baseline_masks.append(inverted_dict["seg_2"])
            
            temp_dict = {"seg_2": followup_mask_}
            with allow_missing_keys_mode():
                inverted_dict = self.valid_tfs.inverse(temp_dict)
            inverted_followup_masks.append(inverted_dict["seg_2"])
        
        return inverted_pred_masks, inverted_baseline_masks, inverted_followup_masks

    def process_inverse(self, mo, baseline_mask, followup_mask):
        # mo (torch.Tensor): model output [B, num_classes, height, width, ...]
        # baseline_mask (torch.Tensor): baseline mask [B, height, width, ...]
        # followup_mask (torch.Tensor): followup mask [B, height, width, ...]
        # spacing (tuple): spacing of the image (spacing_x, spacing_y, spacing_z)
        
        inverted_mo, inverted_baseline_mask, inverted_followup_mask = self.invert_transform(mo, baseline_mask, followup_mask)
        
        for inverted_mo_, inverted_baseline_mask_, inverted_followup_mask_ in zip(inverted_mo, inverted_baseline_mask, inverted_followup_mask):
            affine = inverted_mo_.affine
            spacing = get_spacing_from_affine(affine)
            
            # TODO: compute metrics
            baseline_hema_mask = inverted_baseline_mask_ == 1
            followup_hema_mask = inverted_followup_mask_ == 1
            overlap_mask = baseline_hema_mask & followup_hema_mask
            growth_area_mask = followup_hema_mask & (~overlap_mask)
            
            # TODO: compute metrics
            dice_val = dc(inverted_mo_.numpy(), followup_hema_mask.numpy())
            growth_area_dice_val = dc(inverted_mo_[growth_area_mask].numpy(), followup_hema_mask[growth_area_mask].numpy())
        
        pass

    def process(self, mo, baseline_mask, followup_mask, spacing=None):
        # mo (torch.Tensor): model output [B, num_classes, height, width, ...]
        # baseline_mask (torch.Tensor): baseline mask [B, height, width, ...]
        # followup_mask (torch.Tensor): followup mask [B, height, width, ...]
        # spacing (tuple): spacing of the image (spacing_x, spacing_y, spacing_z)
        if self.spacing is None:
            self.spacing = spacing
        
        pred = mo.cpu().numpy()
        baseline_mask = baseline_mask.cpu().numpy()
        followup_mask = followup_mask.cpu().numpy()
        
        # compute growth area gt
        baseline_hema_mask = baseline_mask == 1
        followup_hema_mask = followup_mask == 1
        overlap_mask = baseline_hema_mask & followup_hema_mask
        growth_area_mask = followup_hema_mask & (~overlap_mask)
        
        # compute growth area pred
        overlap_pred_baseline_mask = baseline_hema_mask & (pred == 1)
        growth_area_pred_mask = (pred == 1) & (~overlap_pred_baseline_mask)
        
        for pred_, followup_hema_mask_, growth_area_mask_, growth_area_pred_mask_ in zip(pred, followup_hema_mask, growth_area_mask, growth_area_pred_mask):
            dice_val = dc(pred_, followup_hema_mask_)
            growth_area_dice_val = dc(growth_area_pred_mask_, growth_area_mask_)
            
            self._growth_area_dices.append(growth_area_dice_val)
            self._dices.append(dice_val)
            
            self._samples += 1

    def evaluate(self):
        results = OrderedDict()
        
        avg_dice = np.mean(self._dices) * 100
        avg_growth_area_dice = np.mean(self._growth_area_dices) * 100
        
        print(
              "=> result\n"
              f"* total: {self._samples:,}\n"
              f"* dice: {avg_dice:.1f}%\n"
              f"* growth area dice: {avg_growth_area_dice:.1f}%\n")

        results["dice"] = avg_dice
        results["growth_area_dice"] = avg_growth_area_dice

        return results