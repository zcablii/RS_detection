import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
from functools import partial

import jittor as jt
from jittor import nn
from jdet.utils.registry import LOSSES
from .binary_cross_entropy_loss import binary_cross_entropy_with_logits

@LOSSES.register_module()
class EQLv2Impr(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=1203,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 new_gamma=0.5,
                 new_mu=0.08,
                 scale_factor=10,
                 vis_grad=False,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.pos_grad = jt.zeros(self.num_classes).stop_grad()
        self.neg_grad = jt.zeros(self.num_classes).stop_grad()
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.pos_neg = (jt.ones(self.num_classes) * 100).stop_grad()
        self.acc_loss = (jt.zeros(self.num_classes) + 1e-10).stop_grad()

        self.test_with_obj = test_with_obj
        def _func(x, gamma, mu):
            return 1 / (1 + jt.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

        def new_func(g, gamma, mu, scale_factor):
            x = g / mu
            y = 1 / (jt.pow(x, gamma) + 1/scale_factor)
            return y
        self.map_new_func = partial(new_func,
                                    gamma=new_gamma,
                                    mu=new_mu,
                                    scale_factor=scale_factor)

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

        pos_w, neg_w = self.get_weight(cls_score)

        bce_loss = binary_cross_entropy_with_logits(cls_score, target,
                                                    reduction='none')

        weight = pos_w * target + neg_w * (1 - target)
        cls_weight = self.get_cls_weight()

        eql_loss = jt.sum(bce_loss * weight, dim=0) / self.n_i
        imp_loss = eql_loss * cls_weight
        # print("eql_loss.shape:", eql_loss.shape)
        # print("cls_weight.shape:", cls_weight.shape)
        # print("imp_loss.shape:", imp_loss.shape)
        
        # print("cls_weight:", cls_weight)
        cls_loss = jt.sum(imp_loss)

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())
        self.update_cls_weight(imp_loss.detach())

        # if jt.isnan(cls_loss):
        # print("cls_weight:", cls_weight)
        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = jt.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = jt.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = jt.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = jt.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = jt.sum(grad * (1 - target) * weight, dim=0)[:-1]

        if jt.in_mpi:
            pos_grad = pos_grad.mpi_all_reduce()
            neg_grad = neg_grad.mpi_all_reduce()

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)
        # print("pos_neg:", self.pos_neg.numpy())

    def get_weight(self, cls_score):
        # neg_w = self.map_func(self.pos_neg)
        neg_w = jt.concat([self.map_func(self.pos_neg), jt.ones(1)])
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w

    def get_cls_weight(self):
        ratios = self.acc_loss / jt.sum(self.acc_loss)
        print("ratios:", ratios.numpy())
        weight = jt.concat([self.map_new_func(ratios), jt.ones(1)])
        return weight

    def update_cls_weight(self, imp_loss):
        self.acc_loss += imp_loss[:-1]
