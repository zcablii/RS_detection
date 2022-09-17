from functools import partial

import jittor as jt
from jittor import nn
from jdet.utils.registry import LOSSES

@LOSSES.register_module()
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=1203,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        self.test_with_obj = test_with_obj

    def execute(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = jt.zeros_like(pred)
            target[jt.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)

        cls_loss = nn.binary_cross_entropy_with_logits(cls_score, target,
                                                       size_average=True)
        cls_loss = jt.sum(cls_loss) / self.n_i

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = jt.sigmoid(cls_score)
        # print("cls_score.max():", cls_score.max())
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score


