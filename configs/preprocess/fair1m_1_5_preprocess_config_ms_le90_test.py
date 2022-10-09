type='FAIR1M_1_5'
source_fair_dataset_path='/opt/data/private/LYX/data/testa-3sdfs'
source_dataset_path='/opt/data/private/LYX/data/FAIR1M2.0_dota'
target_dataset_path='/opt/data/private/LYX/data/FAIR1M2.0_ms'
convert_tasks=['test']
angle_version='le90'
# available labels: train, val, test, trainval
tasks=[
    dict(
        label='test',
        config=dict(
            subimage_size=1024,
            overlap_size=200,
            multi_scale=[0.5,1., 1.5], # use ms 0.5,1,1.5
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    )
]

