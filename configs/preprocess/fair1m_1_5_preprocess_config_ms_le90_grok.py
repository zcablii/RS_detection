"""Script
python tools/preprocess.py --config-file configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_grok.py
"""

type='FAIR1M_1_5'
source_fair_dataset_path='/yimian/jdet'
source_dataset_path='/yimian/dota_ms'
target_dataset_path='/yimian/preprocessed_ms_le90'
convert_tasks=['train','test']
angle_version='le90'
# available labels: train, val, test, trainval
tasks=[
    dict(
        label='train',
        config=dict(
            subimage_size=1024,
            overlap_size=200,
            multi_scale=[0.5,1., 1.5], # use ms 0.5,1,1.5
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    ),
    dict(
        label='test',
        config=dict(
            subimage_size=1024,
            overlap_size=200,
            multi_scale=[0.5,1., 1.5],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    )
]