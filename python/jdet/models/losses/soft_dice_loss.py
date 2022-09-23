# Standard Library
from typing import Optional
import warnings
import logging
import sys

import jittor as jt
from jittor import nn
from jdet.utils.registry import LOSSES

@LOSSES.register_module()
class SoftDiceLoss(nn.Module):
    """

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=-1,
                 num_classes=1203,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=2.0,
                 smooth=1,
                 vis_grad=False):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True
        self.gamma = gamma
        self.smooth = smooth
        self.counter = 0

        # cfg for efl
        self.vis_grad = vis_grad

    def execute(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size() # (N, C)

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = jt.zeros((self.n_i, self.n_c+1))
            target[jt.arange(self.n_i), gt_classes] = 1
            return target[:, :-1] # (N, C)

        # print("label.max():", label.max())
        target = expand_label(cls_score, label) # (N, C)

        pred = jt.sigmoid(cls_score) # (N, C)
        # print("jt.sum(pred, dim=0):", jt.sum(pred, dim=0))
        # focal_pred = ((1 - pred.detach()) ** self.gamma) * pred
        focal_pred = pred
        intersection = focal_pred * target
        union = focal_pred + target - intersection
        cls_intersection = jt.sum(intersection, dim=0) + 0.00001
        cls_union = jt.sum(union, dim=0)  + 0.00001
        # print("cls_intersection:", cls_intersection)
        # print("cls_union:", cls_union)
        # (2 * interection + 0.000001) / (focal_pred + target + 0.000001)
        cls_loss = 1 - cls_intersection / cls_union
        # cls_focal_pred = jt.sum(focal_pred, dim=0)
        # cls_target = jt.sum(target, dim=0)
        # cls_interection = jt.sum(interection, dim=0)
        # print("cls_pred:", cls_focal_pred)
        # print("cls_target:", cls_target)
        # print("cls_interection:", cls_interection)
        # cls_loss = 1 - ((2 * cls_interection + self.smooth) /
        #                 (cls_focal_pred + cls_target + self.smooth))
        # self.counter += 1
        # if self.counter % 5 == 0:
        #     self.counter = 0
        #     cls_pred = jt.sum(focal_pred, dim=0)  + 0.00001
        #     cls_target = jt.sum(target, dim=0)  + 0.00001
        #     print("cls_pred:        ", cls_pred)
        #     print("cls_target:      ", cls_target)
        #     print("cls_intersection:", cls_intersection)
        #     print("cls_union:       ", cls_union)
        #     print("cls_loss:        ", cls_loss)
        #     print("\n")
            
        return self.loss_weight * cls_loss.mean()

    def get_activation(self, cls_score):
        cls_score = jt.sigmoid(cls_score)
        return cls_score

    def get_channel_num(self, num_classes):
        num_channel = num_classes
        return num_channel


