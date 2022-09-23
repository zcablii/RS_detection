# Standard Library
from typing import Optional
import warnings
import logging
import sys

import jittor as jt
from jittor import nn
from jdet.utils.registry import LOSSES

@LOSSES.register_module()
class EqualizedFocalLossV1(nn.Module):
    """我把 obj 也当作一类，也采用 focal

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
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 scale_factor=8.0,
                 vis_grad=False,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for efl
        self.vis_grad = vis_grad
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.scale_factor = scale_factor

        # initial variables
        self.pos_grad = jt.zeros(self.num_classes+1).stop_grad()
        self.neg_grad = jt.zeros(self.num_classes+1).stop_grad()
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.pos_neg = jt.ones(self.num_classes+1).stop_grad()

        self.test_with_obj = test_with_obj

    def execute(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size() # (N, C+1)

        # print("label:", label)
        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = jt.zeros_like(pred)
            target[jt.arange(self.n_i), gt_classes] = 1
            return target # (N, C+1)

        target = expand_label(cls_score, label) # (N, C+1)

        pred = jt.sigmoid(cls_score) # (N, C+1)
        pred_t = pred * target + (1 - pred) * (1 - target) # (N, C+1)

        map_val = 1 - self.pos_neg.detach() # (C+1,)
        dy_gamma = self.focal_gamma + self.scale_factor * map_val # (C+1,)
        # focusing factor, (N, C+1)
        ff = dy_gamma.view(1, -1).expand(self.n_i, self.n_c)
        # weighting factor, (N, C+1)
        wf = ff / self.focal_gamma

        # ce_loss
        ce_loss = -jt.log(pred_t) # (N, C+1)
        cls_loss = ce_loss * jt.pow((1 - pred_t), ff.detach()) * wf.detach() # (N, C+1)

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * target + (
                1 - self.focal_alpha) * (1 - target) # (N, C+1)
            cls_loss = alpha_t * cls_loss # (N, C+1)

        print("cls_loss", cls_loss.sum(dim=0).numpy())
        cls_loss = jt.sum(cls_loss) / self.n_i
        # cls_loss = jt.mean(cls_loss)

        self.collect_grad(cls_score.detach(), target.detach())

        return self.loss_weight * cls_loss

    def collect_grad(self, cls_score, target):
        prob = jt.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = jt.abs(grad)

        # Attention: I collect grad for objectiveness branch [:-1]
        pos_grad = jt.sum(grad * target, dim=0)
        neg_grad = jt.sum(grad * (1 - target), dim=0)

        if jt.in_mpi:
            pos_grad = pos_grad.mpi_all_reduce()
            neg_grad = neg_grad.mpi_all_reduce()

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        tmp_pos_neg = self.pos_grad / (self.neg_grad + 1e-10)
        # print('tmp_pos_neg:', tmp_pos_neg)
        self.pos_neg = jt.clamp(tmp_pos_neg, min_v=0, max_v=1)

    def get_activation(self, cls_score):
        cls_score = jt.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel