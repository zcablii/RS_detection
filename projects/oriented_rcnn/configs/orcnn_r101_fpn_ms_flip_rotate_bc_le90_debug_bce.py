"""Script
# 单卡训练
python tools/run_net.py --config-file projects/oriented_rcnn/configs/orcnn_r101_fpn_ms_flip_rotate_bc_le90_debug_bce.py

# 指定多卡训练
CUDA_VISIBLE_DEVICES="0,1,2,3" mpirun --allow-run-as-root -np 4 python tools/run_net.py --config-file projects/oriented_rcnn/configs/orcnn_r101_fpn_ms_flip_rotate_bc_le90_debug_bce.py

# 单卡测试生成 csv
python tools/run_net.py --config-file projects/oriented_rcnn/configs/orcnn_r101_fpn_ms_flip_rotate_bc_le90_debug_bce.py --task test

# 指定多卡测试生成 csv
CUDA_VISIBLE_DEVICES="0,1,2,3" mpirun --allow-run-as-root -np 4 python tools/run_net.py --config-file projects/oriented_rcnn/configs/orcnn_r101_fpn_ms_flip_rotate_bc_le90_debug_bce.py --task test

# 根据 csv 离线算分数
python tools/val.py --csvfile_path submit_zips/orcnn_r101_fpn_ms_flip_rotate_bc_le90_debug_bce.csv
"""

dataset_root = '/yimian'
# model settings
num_classes = 10
model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='Resnet101',
        frozen_stages=1,
        return_stages=["layer1","layer2","layer3","layer4"],
        pretrained= True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn = dict(
        type = "OrientedRPNHead",
        in_channels=256,
        num_classes=1,
        min_bbox_size=0,
        nms_thresh=0.8,
        nms_pre=2000,
        nms_post=2000,
        # nms_pre=4000,
        # nms_post=4000,
        feat_channels=256,
        bbox_type='obb',
        reg_dim=6,
        background_label=0,
        reg_decoded_bbox=False,
        pos_weight=-1,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            target_means=[.0, .0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(type='CrossEntropyLossForRcnn', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1,
            match_low_quality=True,
            assigned_labels_filled=-1,
            ),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False)
    ),
    bbox_head=dict(
        type='OrientedEQLv2Head',
        num_classes=num_classes,
        in_channels=256,
        fc_out_channels=1024,
        score_thresh=0.05,
        # score_thresh=0.001,
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
            type='BinaryCrossEntropyLoss',
            num_classes=num_classes),
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
        )
    )

angle_version = 'le90'
dataset = dict(
    train=dict(
        type="FAIR1M_1_5_Dataset",
        dataset_dir=f'{dataset_root}/preprocessed_ms_le90/train_1024_200_0.5-1.0-1.5',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
                angle_version=angle_version
            ),
            dict(type='RotatedRandomFlip', prob=0.5),
            dict(
                type="RandomRotateAug",
                random_rotate_on=True,
                angle_version=angle_version
            ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean= [123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False,)
        ],
        batch_size=8,
        num_workers=16,
        shuffle=True,
        filter_empty_gt=False
    ),
    val=dict(
        type="FAIR1M_1_5_Dataset",
        dataset_dir=f'{dataset_root}/preprocessed_ms_le90/train_1024_200_0.5-1.0-1.5',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
                angle_version=angle_version
            ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean= [123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        batch_size=8,
        num_workers=16,
        shuffle=False
    ),
    test=dict(
        type="ImageDataset",
        images_dir=f'{dataset_root}/preprocessed_ms_le90/test_1024_200_0.5-1.0-1.5/images',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
                angle_version=angle_version
            ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean= [123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False,),
        ],
        dataset_type="FAIR1M_1_5",
        num_workers=16,
        batch_size=1,
    )
)


# optimizer = dict(type='SGD',  lr=0.005, momentum=0.9, weight_decay=0.0001, grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type='SGD',  lr=0.02, momentum=0.9, weight_decay=0.0001, grad_clip=dict(max_norm=35, norm_type=2))

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    milestones=[7, 10])

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
max_epoch = 12 # 12
eval_interval = 100
checkpoint_interval = 1
log_interval = 50