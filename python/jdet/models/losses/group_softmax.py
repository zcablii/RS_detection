import bisect
import numpy as np

import jittor as jt
from jittor import nn

from jdet.models.losses import weighted_cross_entropy
from jdet.utils.registry import LOSSES


FAIR1M_1_5_CATEGORIES = [
    {'name': 'Airplane', 'instance_count': 10671, 'id': 1},
    {'name': 'Ship', 'instance_count': 8689, 'id': 2},
    {'name': 'Vehicle', 'instance_count': 66017, 'id': 3},
    {'name': 'Basketball_Court', 'instance_count': 394, 'id': 4},
    {'name': 'Tennis_Court', 'instance_count': 731, 'id': 5},
    {'name': 'Football_Field', 'instance_count': 236, 'id': 6},
    {'name': 'Baseball_Field', 'instance_count': 252, 'id': 7},
    {'name': 'Intersection', 'instance_count': 1549, 'id': 8},
    {'name': 'Roundabout', 'instance_count': 136, 'id': 9},
    {'name': 'Bridge', 'instance_count': 311, 'id': 10},
]

def get_instance_count():
    instance_count = [None] * len(FAIR1M_1_5_CATEGORIES)
    for c in FAIR1M_1_5_CATEGORIES:
        category_id = c["id"] - 1
        instance_count[category_id] = c["instance_count"]
    return instance_count


@LOSSES.register_module()
class GroupSoftmax(nn.Module):
    """
    This uses a different encoding from v1.
    v1: [cls1, cls2, ..., other1_for_group0, other_for_group_1, bg, bg_others]
    this: [group0_others, group0_cls0, ..., group1_others, group1_cls0, ...]
    """
    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 beta=8,
                 bin_split=(500, 5000),
                 version="v1"):
        super(GroupSoftmax, self).__init__()
        self.use_sigmoid = False
        self.group = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.beta = beta
        self.bin_split = bin_split
        self.version = version
        self.cat_instance_count = get_instance_count()
        self.num_classes = len(self.cat_instance_count)
        assert not self.use_sigmoid
        self._assign_group()
        self._prepare_for_label_remapping()

        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True

    def _assign_group(self):
        self.num_group = (len(self.bin_split) + 1) + 1  # add a group for background
        self.group_cls_ids = [[] for _ in range(self.num_group)]
        group_ids = list(map(lambda x: bisect.bisect_right(self.bin_split, x), self.cat_instance_count))
        self.group_ids = group_ids + [self.num_group - 1]
        for cls_id, group_id in enumerate(self.group_ids):
            self.group_cls_ids[group_id].append(cls_id)
        self.n_cls_group = list(map(lambda x: len(x), self.group_cls_ids))

    def _get_group_pred(self, cls_score, apply_activation_func=False):
        group_pred = []
        start = 0
        for group_id, n_cls in enumerate(self.n_cls_group):
            num_logits = n_cls + 1  # + 1 for "others"
            # pred = cls_score.narrow(1, start, num_logits)
            # 实现 jittor 版本的 narrow
            # print("cls_score.shape:", cls_score.shape)
            # print("start:", start)
            # print("start+num_logits:", start+num_logits)
            pred = cls_score[:, start:start+num_logits, ...]
            start = start + num_logits
            if apply_activation_func:
                pred = nn.softmax(pred, dim=1)
            group_pred.append(pred)
        assert start == self.num_classes + 1 + self.num_group
        return group_pred

    def _prepare_for_label_remapping(self):
        group_label_maps = []
        for group_id, n_cls in enumerate(self.n_cls_group):
            label_map = [0 for _ in range(self.num_classes + 1)]
            group_label_maps.append(label_map)
        # init value is 1 because 0 is set for "others"
        _tmp_group_num = [1 for _ in range(self.num_group)]
        for cls_id, group_id in enumerate(self.group_ids):
            g_p = _tmp_group_num[group_id]  # position in group
            group_label_maps[group_id][cls_id] = g_p
            _tmp_group_num[group_id] += 1
        self.group_label_maps = jt.Var(group_label_maps).long()

    def _remap_labels(self, labels):
        new_labels = []
        new_weights = []  # use this for sampling others
        new_avg = []
        for group_id in range(self.num_group):
            mapping = self.group_label_maps[group_id]
            new_bin_label = mapping[labels]
            new_bin_label = jt.Var(new_bin_label).long()
            if self.is_background_group(group_id):
                weight = jt.ones_like(new_bin_label)
            else:
                weight = self._sample_others(new_bin_label)
            new_labels.append(new_bin_label)
            new_weights.append(weight)

            avg_factor = max(jt.sum(weight).float().item(), 1.)
            new_avg.append(avg_factor)
        return new_labels, new_weights, new_avg

    def _sample_others(self, label):

        # only works for non bg-fg bins

        fg = jt.where(label > 0, jt.ones_like(label),
                      jt.zeros_like(label))
        # fg_idx = fg.nonzero(as_tuple=True)[0]
        fg_idx = jt.nonzero(fg).reshape(-1)
        fg_num = fg_idx.shape[0]
        if fg_num == 0:
            return jt.zeros_like(label)

        bg = 1 - fg
        # bg_idx = bg.nonzero(as_tuple=True)[0]
        bg_idx = jt.nonzero(bg).reshape(-1)
        bg_num = bg_idx.shape[0]

        bg_sample_num = int(fg_num * self.beta)

        if bg_sample_num >= bg_num:
            weight = jt.ones_like(label)
        else:
            sample_idx = np.random.choice(bg_idx.numpy(),
                                          (bg_sample_num, ), replace=False)
            sample_idx = jt.Var(sample_idx)
            fg[sample_idx] = 1
            weight = fg

        # return weight.to(label.device)
        return weight

    def execute(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        group_preds = self._get_group_pred(cls_score, apply_activation_func=False)
        new_labels, new_weights, new_avg = self._remap_labels(label)

        cls_loss = []
        for group_id in range(self.num_group):
            pred_in_group = group_preds[group_id]
            label_in_group = new_labels[group_id]
            weight_in_group = new_weights[group_id]
            avg_in_group = new_avg[group_id]
            loss_in_group = weighted_cross_entropy(pred_in_group,
                                                   label_in_group,
                                                   weight=weight_in_group,
                                                   avg_factor=avg_in_group,
                                                   reduce=True)
            cls_loss.append(loss_in_group)
        cls_loss = sum(cls_loss)
        return cls_loss * self.loss_weight

    def get_activation(self, cls_score):
        n_i, n_c = cls_score.size()
        group_activation = self._get_group_pred(cls_score, apply_activation_func=True)
        bg_score = group_activation[-1]
        # activation = cls_score.new_zeros((n_i, len(self.group_ids)))
        activation = jt.zeros((n_i, len(self.group_ids)))
        for group_id, cls_ids in enumerate(self.group_cls_ids[:-1]):
            activation[:, cls_ids] = group_activation[group_id][:, 1:]
        activation *= bg_score[:, [0]]
        activation[:, -1] = bg_score[:, 1]

        return activation

    def get_cls_channels(self, num_classes):
        num_channel = num_classes + 1 + self.num_group
        return num_channel

    def is_background_group(self, group_id):
        return group_id == self.num_group - 1
