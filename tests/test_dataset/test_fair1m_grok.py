"""Script
python tests/test_dataset/test_fair1m_grok.py
"""

from jdet.utils.registry import build_from_cfg, DATASETS
import numpy as np

dataset = build_from_cfg(dict(
        type="FAIR1M_1_5_Dataset",
        dataset_dir='/yimian/preprocessed_ms_le90/train_1024_200_0.5-1.0-1.5',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
                angle_version='le90'
            ),
            dict(type='RotatedRandomFlip', prob=0.5),
            dict(
                type="RandomRotateAug",
                random_rotate_on=True,
                angle_version='le90'
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
        batch_size=12,
        num_workers=16,
        shuffle=True,
        filter_empty_gt=False
    ), DATASETS)

unique_labels = []
max_val = -1
min_val = np.inf
for img_info in dataset.img_infos:
    labels = np.unique(img_info["ann"]['labels'])
    if len(labels) > 0:
        if max_val < labels.min():
            max_val = labels.min()
        if min_val > labels.max():
            min_val = labels.min()
    unique_labels.append(labels)
print(unique_labels)
print(max_val)
print(min_val)