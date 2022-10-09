dataset_root = '/opt/data/private/LYX/data'
# model settings
num_classes = 10
model = dict(
    type='OrientedRCNN',
    backbone=dict( 
        type='van_b3',
        img_size=1024,
        num_stages=4,
        out_indices = (0, 1, 2, 3),
        pretrained= True),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
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
        feat_channels=256,
        bbox_type='obb',
        reg_dim=6,
        background_label=0,
        reg_decoded_bbox=False,
        pos_weight=-1,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
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
        type='OrientedHead',
        num_classes=num_classes,
        in_channels=256,
        fc_out_channels=1024,
        score_thresh=0.01,
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
        )
    )
    
angle_version = 'le90'
dataset = dict(
    train=dict(
        type="FAIR1M_1_5_Dataset",
        dataset_dir=f'{dataset_root}/FAIR1M2.0_v2_extra_gray_color_preprocessed_ms/train_1024_200_0.5-1.0-1.5',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
                angle_version = angle_version
            ),
            dict(type='RotatedRandomFlip', prob=0.5),
            dict(
                type="RandomRotateAug",
                random_rotate_on=True,
                angle_version = angle_version
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,)
            
        ],
        batch_size=8,
        num_workers=8,
        shuffle=True,
        filter_empty_gt=False
    ),
    val=dict(
        type="FAIR1M_1_5_Dataset",
        dataset_dir=f'{dataset_root}/preprocessed_ms/train_1024_200_0.5-1.0-1.5',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
                angle_version = angle_version
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        batch_size=8,
        num_workers=8,
        shuffle=False
    ),
    test=dict(
        type="ImageDataset",
        images_dir=f'{dataset_root}/test_2_preprocessed_ms/test_1024_200_0.5-1.0-1.5/images', # test_2_preprocessed_ms, testa_3_ms
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
                angle_version = angle_version
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,),
        ],
        dataset_type="FAIR1M_1_5",
        num_workers=4,
        batch_size=1,
    )
)


optimizer = dict(type='AdamW',  lr=0.0001, weight_decay=0.05)

# learning policy
scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[7, 10])


optimizer_swa = dict(type='AdamW',  lr=0.0001, weight_decay=0.05)

scheduler_swa = dict(
    type='CosineAnnealingLR',
    min_lr = 0.000001
    )

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
swa_start_epoch = 12

max_epoch = 9
eval_interval = 20
checkpoint_interval = 1
log_interval = 200

# resume_path = '/opt/data/private/LYX/RS_detection/work_dirs/orcnn_r152_fpn_1-2-4-8_anchor/checkpoints/swa_17-24.pkl'
# model_only = True

merge_nms_threshold_type = 1 