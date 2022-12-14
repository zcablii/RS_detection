import numpy as np

import pickle
import jittor as jt
from jittor import nn
from jittor import init
from jdet.data.devkits.result_merge import py_cpu_nms_poly_fast
from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS,BOXES,LOSSES, ROI_EXTRACTORS,build_from_cfg

from jdet.ops.bbox_transforms import *
from jdet.models.utils.modules import ConvModule

from jittor.misc import _pair

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        init.gauss_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return jt.concat((arr1,arr2),dim=0)

def del_tensor_eles(arr,indexs):
    if arr is None:
        return arr
    indexs.sort(reverse=True)
    for each in indexs:
        arr = del_tensor_ele(arr,each)
    return arr

def rm_illegal_targets(scores, bbox_deltas, rois, bbox_targets):
    labels, label_weights, bbox_targets, bbox_targets_decode, bbox_weights = bbox_targets
    if bbox_targets_decode is None:
        return scores, bbox_deltas, rois, (labels, label_weights, bbox_targets, bbox_targets_decode, bbox_weights)

    else:
        idx3 = (bbox_targets_decode>1.024e4).nonzero()[:,0]
        idx4 = (bbox_targets_decode<-1.024e4).nonzero()[:,0]
        idx6 = (bbox_targets_decode[:,:4]<0).nonzero()[:,0]
    idx1 = (bbox_targets>1.024e3).nonzero()[:,0]
    idx2 = (bbox_targets<-1.024e3).nonzero()[:,0]
    idx5 = (bbox_targets[:,:4]<0).nonzero()[:,0]
    idxs = list(set([int(ind.item()) for ind in jt.concat([idx1, idx2,idx3,idx4,idx5,idx6])]))
    scores = del_tensor_eles(scores,idxs)
    bbox_deltas = del_tensor_eles(bbox_deltas,idxs)
    rois = del_tensor_eles(rois,idxs)
    labels = del_tensor_eles(labels,idxs)
    label_weights = del_tensor_eles(label_weights,idxs)
    bbox_targets = del_tensor_eles(bbox_targets,idxs)
    bbox_targets_decode = del_tensor_eles(bbox_targets_decode,idxs)
    bbox_weights = del_tensor_eles(bbox_weights,idxs)
    return scores, bbox_deltas, rois, (labels, label_weights, bbox_targets, bbox_targets_decode, bbox_weights)

@HEADS.register_module()
class OrientedBCEHead(nn.Module):

    def __init__(self,
                 num_classes=15,
                 in_channels=256,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 fc_out_channels=1024,
                 conv_out_channels=256,
                 score_thresh=0.05,
                 assigner=dict(
                     type='MaxIoUAssigner',
                     pos_iou_thr=0.5,
                     neg_iou_thr=0.5,
                     min_pos_iou=0.5,
                     ignore_iof_thr=-1,
                     match_low_quality=False,
                     assigned_labels_filled=-1,
                     iou_calculator=dict(type='BboxOverlaps2D_rotated_v1')),
                 sampler=dict(
                     type='RandomSamplerRotated',
                     num=512,
                     pos_fraction=0.25,
                     neg_pos_ub=-1,
                     add_gt_as_proposals=True),
                 bbox_coder=dict(
                     type='OrientedDeltaXYWHTCoder',
                     target_means=[0., 0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
                 bbox_roi_extractor=dict(
                     type='OrientedSingleRoIExtractor',
                     roi_layer=dict(type='ROIAlignRotated_v1', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     extend_factor=(1.4, 1.2),
                     featmap_strides=[4, 8, 16, 32]),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     ),
                 loss_bbox=dict(
                     type='SmoothL1Loss', 
                     beta=1.0, 
                     loss_weight=1.0
                     ),
                 with_bbox=True,
                 with_shared_head=False,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 start_bbox_type='obb',
                 end_bbox_type='obb',
                 reg_dim=None,
                 reg_class_agnostic=True,
                 reg_decoded_bbox=False,
                 pos_weight=-1,
     ):
        super().__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.with_bbox = with_bbox
        self.with_shared_head = with_shared_head
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.fp16_enabled = False
        self.pos_weight = pos_weight
        self.score_thresh = score_thresh
        self.use_delta_and_encode_loss = ['kfiou']
        # TODO Add to config
        roi_feat_size = 7
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]

        self.start_bbox_type = start_bbox_type
        self.end_bbox_type = end_bbox_type
        assert self.start_bbox_type in ['hbb', 'obb', 'poly']
        assert self.end_bbox_type in ['hbb', 'obb', 'poly']
        self.reg_dim = get_bbox_dim(self.end_bbox_type) \
                if reg_dim is None else reg_dim

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels

        self.bbox_coder = build_from_cfg(bbox_coder, BOXES)
        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)
        self.assigner = build_from_cfg(assigner, BOXES)
        self.sampler = build_from_cfg(sampler, BOXES)
        self.bbox_roi_extractor = build_from_cfg(bbox_roi_extractor, ROI_EXTRACTORS)

        if self.with_cls:
            if self.use_sigmoid:
                # sigmoid
                if self.group_activation:
                    cls_channels = self.loss_cls.get_channel_num(
                        self.num_classes)
                    self.fc_cls = nn.Linear(in_channels, cls_channels)
                else:
                    self.fc_cls = nn.Linear(in_channels, self.num_classes)
            else:
                # softmax
                raise ValueError("Must use_sigmoid")
        if self.with_reg:
            out_dim_reg = self.reg_dim if reg_class_agnostic else self.reg_dim * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)

        self._init_layers()
        self.init_weights()

    @property
    def use_sigmoid(self):
        use_sigmoid = getattr(self.loss_cls, 'use_sigmoid', False)
        use_bce = getattr(self.loss_cls, 'use_bce', False)
        return (use_sigmoid or use_bce)

    @property
    def group_activation(self):
        return getattr(self.loss_cls, 'group', False)

    def _add_conv_fc_branch(self, num_branch_convs,  num_branch_fcs, in_channels, is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels

        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=None,
                        norm_cfg=None))
            last_layer_dim = self.conv_out_channels

        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append( nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels

        return branch_convs, branch_fcs, last_layer_dim

    def _init_layers(self):

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(self.num_shared_convs, self.num_shared_fcs, self.in_channels, True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.use_sigmoid:
                # sigmoid
                if self.group_activation:
                    cls_channels = self.loss_cls.get_channel_num(
                        self.num_classes)
                else:
                    cls_channels = self.num_classes
                self.fc_cls = nn.Linear(self.cls_last_dim, cls_channels)
            else:
                # softmax
                raise ValueError("Must use_sigmoid")

        if self.with_reg:
            out_dim_reg = self.reg_dim if self.reg_class_agnostic else \
                    self.reg_dim * self.num_classes
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def init_weights(self):

        if self.with_cls:
            nn.init.gauss_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.gauss_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        if self.use_sigmoid:
            bias_cls = bias_init_with_prob(0.001)
            normal_init(self.fc_cls, std=0.01, bias=bias_cls)

    def arb2roi(self, bbox_list, bbox_type='hbb'):

        assert bbox_type in ['hbb', 'obb', 'poly']
        bbox_dim = get_bbox_dim(bbox_type)

        rois_list = []
        for img_id, bboxes in enumerate(bbox_list):
            if bboxes.size(0) > 0:
                img_inds = jt.full((bboxes.size(0), 1), img_id, dtype=bboxes.dtype)
                rois = jt.concat([img_inds, bboxes[:, :bbox_dim]], dim=-1)
            else:
                rois = jt.zeros((0, bbox_dim + 1), dtype=bboxes.dtype)
            rois_list.append(rois)
        rois = jt.concat(rois_list, 0)
        return rois

    def get_results(self, multi_bboxes, multi_scores, score_factors=None, bbox_type='hbb'):

        bbox_dim = get_bbox_dim(bbox_type)
        num_classes = multi_scores.size(1) - 1

        # exclude background category
        if multi_bboxes.shape[1] > bbox_dim:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, bbox_dim)
        else:
            bboxes = multi_bboxes[:, None].expand(-1, num_classes, bbox_dim)
        scores = multi_scores[:, :-1]

        # filter out boxes with low scores
        valid_mask = scores > self.score_thresh
        bboxes = bboxes[valid_mask]
        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = scores[valid_mask]
        labels = valid_mask.nonzero()[:, 1]

        if bboxes.numel() == 0:
            bboxes = jt.zeros((0, 9), dtype=multi_bboxes.dtype)
            labels = jt.zeros((0, ), dtype="int64")
            return bboxes, labels

        dets = jt.concat([obb2poly(bboxes), scores.unsqueeze(1)], dim=1)
        return dets, labels

    def forward_single(self, x, sampling_results, test=False):

        if test:
            rois = self.arb2roi(sampling_results, bbox_type=self.start_bbox_type)
        else:
            rois = self.arb2roi([res.bboxes for res in sampling_results], bbox_type=self.start_bbox_type)

        x = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)

        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = nn.relu(fc(x))

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.ndim > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = nn.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.ndim > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = nn.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, rois

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_targets_decode, bbox_weights, reduction_override=None):

        losses = dict()
        if cls_score is not None:
            avg_factor = max(jt.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
        # print('bbox_pred',bbox_pred)
        if bbox_pred is not None:

            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)

            use_delta_and_decode = False
            bbox_pred_decode = None
            pos_bbox_pred_decode = None
            if hasattr(self.loss_bbox,'loss_type'):
                if self.loss_bbox.loss_type in self.use_delta_and_encode_loss:
                    use_delta_and_decode=True


            # do not perform bounding box regression for BG anymore.
            if pos_inds.any_():
                if use_delta_and_decode:
                    bbox_pred_decode = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                else:
                    if self.reg_decoded_bbox:
                        # print('rois.shape',rois.shape,bbox_pred.shape) # rois 4096,6  bbox_pred 4096,5
                        bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), self.reg_dim)[pos_inds.astype(jt.bool)]
                    if not (bbox_pred_decode is None):    
                        pos_bbox_pred_decode = bbox_pred_decode.view(bbox_pred_decode.size(0), self.reg_dim)[pos_inds.astype(jt.bool)]
                    # print('pos_bbox_pred',pos_bbox_pred.shape,pos_bbox_pred) # shape num_pos ,5
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, self.reg_dim)[pos_inds.astype(jt.bool), labels[pos_inds.astype(jt.bool)]]
                    if not (bbox_pred_decode is None):    
                        pos_bbox_pred_decode = pos_bbox_pred_decode.view(pos_bbox_pred_decode.size(0), -1, self.reg_dim)[pos_inds.astype(jt.bool), labels[pos_inds.astype(jt.bool)]]
                # print('len(pos_bbox_pred)',len(pos_bbox_pred))
                # print('pos_bbox_pred',pos_bbox_pred)
                # print('[pos_inds.astype(jt.bool)]',len(bbox_targets[pos_inds.astype(jt.bool)]),bbox_targets[pos_inds.astype(jt.bool)])
                # print('pos_bbox_pred',pos_bbox_pred)
                # print('bbox_targets[pos_inds.astype(jt.bool)]',bbox_targets[pos_inds.astype(jt.bool)])
                # print('bbox_weights[pos_inds.astype(jt.bool)]',bbox_weights[pos_inds.astype(jt.bool)])
                # print('shapes',pos_bbox_pred.shape, bbox_targets[pos_inds.astype(jt.bool)].shape, bbox_weights[pos_inds.astype(jt.bool)].shape)
                
                if not (bbox_targets_decode is None):  
                    bbox_targets_decode = bbox_targets_decode[pos_inds.astype(jt.bool)]
                
                # print('pos_bbox_pred',pos_bbox_pred)
                # print('bbox_targets[pos_inds.astype(jt.bool)]',bbox_targets[pos_inds.astype(jt.bool)])
                # print('pos_bbox_pred_decode',pos_bbox_pred_decode)
                # print('targets_decode',bbox_targets_decode[pos_inds.astype(jt.bool)])
                
                # if (bbox_targets[pos_inds.astype(jt.bool)]>1e5).nonzero().any():
                #     idx = (bbox_targets[pos_inds.astype(jt.bool)]>1e5).nonzero()[0]
                #     print('idx',idx)
                #     print('!!!!exp bbox_targets in loss', bbox_targets[pos_inds.astype(jt.bool)][idx[0]])
                # if (bbox_targets[pos_inds.astype(jt.bool)]<-1e5).nonzero().any():
                #     idx = (bbox_targets[pos_inds.astype(jt.bool)]<-1e5).nonzero()[0]
                #     print('idx',idx)
                #     print('!!!!exp bbox_targets in -loss', bbox_targets[pos_inds.astype(jt.bool)][idx[0]])
      
                # if (bbox_targets_decode>1e5).nonzero().any():
                #     idx = (bbox_targets_decode>1e5).nonzero()[0]
                #     print('idx',idx)
                #     print('!!!!exp bbox_targets_decode in loss', bbox_targets_decode[idx[0]])
                # if (bbox_targets_decode<-1e5).nonzero().any():
                #     idx = (bbox_targets_decode<-1e5).nonzero()[0]
                #     print('idx',idx)
                #     print('!!!!exp bbox_targets_decode in -loss', bbox_targets_decode[idx[0]])
                if hasattr(self.loss_bbox,'loss_type'):
                    orcnn_bbox_loss = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.astype(jt.bool)],
                        bbox_weights[pos_inds.astype(jt.bool)],
                        avg_factor=bbox_targets.size(0),
                        pred_decode = pos_bbox_pred_decode,
                        targets_decode  = bbox_targets_decode,
                        reduction_override=reduction_override)
                else:
                    orcnn_bbox_loss = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.astype(jt.bool)],
                        bbox_weights[pos_inds.astype(jt.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)

                # print('orcnn_bbox_loss',orcnn_bbox_loss)
                # print(len(pos_bbox_pred) == len(bbox_targets[pos_inds.astype(jt.bool)]) and len(pos_bbox_pred_decode)==len(bbox_targets_decode[pos_inds.astype(jt.bool)]))
                # print('pos_bbox_pred',pos_bbox_pred[:3])
                # print('bbox_targets[pos_inds.astype(jt.bool)]',bbox_targets[pos_inds.astype(jt.bool)][:3])
                # print('pos_bbox_pred_decode',pos_bbox_pred_decode[:3])
                # print('targets_decode',bbox_targets_decode[pos_inds.astype(jt.bool)][:3])
                
                losses['orcnn_bbox_loss'] = orcnn_bbox_loss
            else:
                losses['orcnn_bbox_loss'] = bbox_pred.sum() * 0

        return losses

    def get_bboxes_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, use_delta_and_decode=False):

        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        # print('num_pos,num_neg',num_pos,num_neg)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes - 1]
        labels = jt.full((num_samples,), self.num_classes).long()
        label_weights = jt.zeros((num_samples,), dtype=pos_bboxes.dtype)
        bbox_targets = jt.zeros((num_samples, self.reg_dim), dtype=pos_bboxes.dtype)
        bbox_targets_decode = jt.zeros((num_samples, self.reg_dim), dtype=pos_bboxes.dtype)
        bbox_weights = jt.zeros((num_samples, self.reg_dim), dtype=pos_bboxes.dtype)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if self.pos_weight <= 0 else self.pos_weight
            label_weights[:num_pos] = pos_weight
            if use_delta_and_decode:
                pos_bbox_targets_decode = pos_gt_bboxes
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets_decode = None
                if not self.reg_decoded_bbox:
                    pos_bbox_targets = self.bbox_coder.encode(
                        pos_bboxes, pos_gt_bboxes)
                else:
                    pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            if not pos_bbox_targets_decode is None:
                bbox_targets_decode[:num_pos, :] = pos_bbox_targets_decode
            else:
                bbox_targets_decode = None
            bbox_weights[:num_pos, :] = 1

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        # print('bbox_targets',bbox_targets)
        # print('bbox_targets_decode',bbox_targets_decode)
      
        # if (bbox_targets>1.024e3).nonzero().any():
        #     idx = (bbox_targets>1.024e3).nonzero()[:,0]
        #     bbox_weights[idx,:] = 0
        #     print('idx',idx)
        #     print('!!!!exp bbox_targets get_bboxes_target_single', bbox_targets[idx[0]])
        # if (bbox_targets<-1.024e3).nonzero().any():
        #     idx = (bbox_targets<-1.024e3).nonzero()[:,0]
        #     bbox_weights[idx,:] = 0
        #     print('idx',idx)
        #     print('!!!!!exp -bbox_targets -get_bboxes_target_single', bbox_targets[idx[0]])

        # if (bbox_targets_decode>1.024e3).nonzero().any():
        #     idx = (bbox_targets_decode>1.024e3).nonzero()[:,0]
        #  #   bbox_weights[idx,:] = 0
        #     print('idx',idx)
        #     print('!!!!exp bbox_targets_decode get_bboxes_target_single', bbox_targets_decode[idx[0]])
        # if (bbox_targets_decode<-1.024e3).nonzero().any():
        #     idx = (bbox_targets_decode<-1.024e3).nonzero()[:,0]
        # #    bbox_weights[idx,:] = 0
        #     print('idx',idx)
        #     print('!!!!!exp bbox_targets_decode -get_bboxes_target_single', bbox_targets_decode[idx[0]])

        return (labels, label_weights, bbox_targets,bbox_targets_decode, bbox_weights)

    def get_bboxes_targets(self, sampling_results, concat=True, use_delta_and_decode=False):

        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]

        l = len(pos_gt_bboxes_list[0])
        if l > 3: l=3
        # print('pos_bboxes_list', pos_bboxes_list[0][:l])
        # print('pos_gt_bboxes_list',pos_gt_bboxes_list[0][:l])
        # print('pos_gt_labels_list',pos_gt_labels_list)
        outputs = multi_apply(
            self.get_bboxes_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            use_delta_and_decode = use_delta_and_decode)

        (labels, label_weights, bbox_targets, bbox_targets_decode, bbox_weights) = outputs

        if concat:
            labels = jt.concat(labels, 0)
            label_weights = jt.concat(label_weights, 0)
            bbox_targets = jt.concat(bbox_targets, 0)
            bbox_weights = jt.concat(bbox_weights, 0)
            if use_delta_and_decode:
                bbox_targets_decode = jt.concat(bbox_targets_decode, 0)
            else:
                bbox_targets_decode = None
        # print('labels',len(labels),labels)
        # print('label_weights',len(label_weights),label_weights)
        # print('bbox_targets',len(bbox_targets),bbox_targets)
        # print('bbox_weights',len(bbox_weights),bbox_weights)
        
      
        # if (bbox_targets>1e5).nonzero().any():
        #     idx = (bbox_targets>1e5).nonzero()[0]
        #     print('idx',idx)
        #     print('!!!!exp bbox_targets get_bboxes_targets', bbox_targets[idx[0]])
        # if (bbox_targets<-1e5).nonzero().any():
        #     idx = (bbox_targets<-1e5).nonzero()[0]
        #     print('idx',idx)
        #     print('!!!!!exp bbox_targets -get_bboxes_targets', bbox_targets[idx[0]])
      
        # if (bbox_targets_decode>1e5).nonzero().any():
        #     idx = (bbox_targets_decode>1e5).nonzero()[0]
        #     print('idx',idx)
        #     print('!!!!exp bbox_targets_decode get_bboxes_targets', bbox_targets_decode[idx[0]])
        # if (bbox_targets_decode<-1e5).nonzero().any():
        #     idx = (bbox_targets_decode<-1e5).nonzero()[0]
        #     print('idx',idx)
        #     print('!!!!!exp bbox_targets_decode -get_bboxes_targets', bbox_targets_decode[idx[0]])

        return (labels, label_weights, bbox_targets, bbox_targets_decode, bbox_weights)

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False):

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        # some loss (Seesaw loss..) may have custom activation
        assert cls_score.ndim == 2, "Check cls_score.ndim"
        if cls_score is not None:
            if self.use_sigmoid:
                if self.group_activation:
                    scores = self.loss_cls.get_activation(cls_score)
                else:
                    # raise ValueError("Must group_activation")
                    scores = jt.sigmoid(cls_score)
            else:
                # softmax
                raise ValueError("Must use_sigmoid")
        else:
            scores = None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            assert self.start_bbox_type == self.end_bbox_type
            bboxes = rois[:, 1:].clone()

        if rescale:
            if isinstance(scale_factor, float):
                scale_factor = [scale_factor for _ in range(4)]
            scale_factor = jt.array(scale_factor, dtype=bboxes.dtype)

            bboxes = bboxes.view(bboxes.size(0), -1, get_bbox_dim(self.end_bbox_type))
            if self.end_bbox_type == 'hbb':
                bboxes /= scale_factor
            elif self.end_bbox_type == 'obb':
                bboxes[..., :4] = bboxes[..., :4] / scale_factor
            elif self.end_bbox_type == 'poly':
                bboxes /= scale_factor.repeat(2)
            bboxes = bboxes.view(bboxes.size(0), -1)

        det_bboxes, det_labels = self.get_results(bboxes, scores, bbox_type=self.end_bbox_type)

        # det_labels = det_labels + 1 # output label range should be adjusted back to [1, self.class_NUm]

        return det_bboxes, det_labels

    def execute(self, x, proposal_list, targets):

        if self.is_training():

            gt_obboxes = []
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_obboxes_ignore = []

            for target in targets:
                if target["rboxes"] is None:
                    obb = None
                else:
                    obb = target["rboxes"].clone()
                    obb[:, -1] *= -1

                if target["rboxes_ignore"] is None or target["rboxes_ignore"].numel() == 0:
                    obb_ignore = None
                else:
                    obb_ignore = target["rboxes_ignore"].clone()
                    obb_ignore[:, -1] *= -1

                gt_obboxes.append(obb)
                gt_obboxes_ignore.append(obb_ignore)
                gt_bboxes.append(target["hboxes"])
                gt_bboxes_ignore.append(target["hboxes_ignore"])
                gt_labels.append(target["labels"] - 1)

            # assign gts and sample proposals
            if self.with_bbox:
                start_bbox_type = self.start_bbox_type
                end_bbox_type = self.end_bbox_type
                target_bboxes = gt_bboxes if start_bbox_type == 'hbb' else gt_obboxes
                target_bboxes_ignore = gt_bboxes_ignore if start_bbox_type == 'hbb' else gt_obboxes_ignore

                num_imgs = len(targets)
                if target_bboxes_ignore is None:
                    target_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []

                for i in range(num_imgs):

                    assign_result = self.assigner.assign(proposal_list[i], target_bboxes[i], target_bboxes_ignore[i], gt_labels[i])

                    sampling_result = self.sampler.sample(
                        assign_result,
                        proposal_list[i],
                        target_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])

                    if start_bbox_type != end_bbox_type:
                        if gt_obboxes[i].numel() == 0:
                            sampling_result.pos_gt_bboxes = jt.zeros((0, gt_obboxes[0].size(-1)), dtype=gt_obboxes[i].dtype)
                        else:
                            sampling_result.pos_gt_bboxes = gt_obboxes[i][sampling_result.pos_assigned_gt_inds, :]

                    sampling_results.append(sampling_result)

            scores, bbox_deltas, rois = self.forward_single(x, sampling_results, test=False)
            # print('scores', scores[0]) # len 4096, num_classes + 1
            # print('bbox_deltas', bbox_deltas[0])# len 4096,5, 5param deltas
            # print('rois', rois[0])# len 4096,6
            use_delta_and_decode = False
            if hasattr(self.loss_bbox,'loss_type'):
                if self.loss_bbox.loss_type in self.use_delta_and_encode_loss:
                    use_delta_and_decode=True
            bbox_targets = self.get_bboxes_targets(sampling_results,use_delta_and_decode = use_delta_and_decode)
            scores, bbox_deltas, rois,bbox_targets = rm_illegal_targets(scores, bbox_deltas, rois,bbox_targets)

            # print('shapes', scores.shape, bbox_deltas.shape, rois.shape, labels.shape, label_weights.shape, bbox_targets.shape, bbox_targets_decode.shape, bbox_weights.shape)
            # print('bbox_targets'), bbox_targets)# labels, label_weights, bbox_targets, bbox_weights each len 4096
            loss = self.loss(scores, bbox_deltas, rois, *bbox_targets)

            return loss

        else:
            result = []
            for i in range(len(targets)):

                scores, bbox_deltas, rois = self.forward_single(x, [proposal_list[i]], test=True)
                img_shape = targets[i]['img_size']
                scale_factor = targets[i]['scale_factor']

                det_bboxes, det_labels = self.get_bboxes(rois, scores, bbox_deltas, img_shape, scale_factor, rescale=True)

                poly = det_bboxes[:, :8]
                scores = det_bboxes[:, 8]
                labels = det_labels

                result.append((poly, scores, labels))

            return result