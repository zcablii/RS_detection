type='FAIR1M_1_5'
source_fair_dataset_path='/opt/data/private/LYX/data/testa-3sdfs'
source_dataset_path='/opt/data/private/LYX/data/FAIR1M2.0_dota'
target_dataset_path='/opt/data/private/LYX/data/FAIR1M2.0_ms'
convert_tasks=['train']
angle_version='le90'
# available labels: train, val, test, trainval
tasks=[
    dict(
        label='train',
        config=dict(
            subimage_size=1024,
            overlap_size=200,
            multi_scale=[0.25,0.4,0.5,0.7,0.8,0.9,1,1.2,1.4,1.5,1.6,1.8,2.0], # use ms 0.5,1,1.5
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    )
]

