type='FAIR1M_1_5'
source_fair_dataset_path='/media/data3/lyx/Detection/data_train_val'
source_dataset_path='/media/data3/lyx/Detection/train_val_dota_ms'
target_dataset_path='/media/data3/lyx/Detection/train_val_preprocessed_ms'
convert_tasks=['train','val']

# available labels: train, val, test, trainval
# tasks=[
#     dict(
#         label='train',
#         config=dict(
#             subimage_size=1024,
#             overlap_size=200,
#             multi_scale=[0.5,1., 1.5], # use ms 0.5,1,1.5
#             horizontal_flip=False,
#             vertical_flip=False,
#             rotation_angles=[0.] 
#         )
#     ),
#     dict(
#         label='test',
#         config=dict(
#             subimage_size=1024,
#             overlap_size=200,
#             multi_scale=[0.5,1., 1.5],
#             horizontal_flip=False,
#             vertical_flip=False,
#             rotation_angles=[0.] 
#         )
#     )
# ]

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
        label='val',
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