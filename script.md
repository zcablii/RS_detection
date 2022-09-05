0. Install env:

pip install -r requirements.txt


1. Data preprocess:

进入 `configs/preprocess/fair1m_1_5_preprocess_config_ms_le90.py`文件，修改这个文件中的三个路径参数为

```python
source_fair_dataset_path='{DATASET_PATH}/data'
source_dataset_path='{DATASET_PATH}/dota'
target_dataset_path='{DATASET_PATH}/preprocessed'
```

其中 `{DATASET_PATH}`与前一步相同。

在 `JDet`目录下执行 `python tools/preprocess.py --config-file configs/preprocess/fair1m_1_5_preprocess_config_ms_le90.py`，即可自动进行数据预处理。


2. Run Oriented R-CNN:

修改config

oriented_rcnn_r101_fpn_1x_dota_ms_with_flip_rotate_balance_cate.py

把dataset_root 改成自己的数据路径。

python tools/run_net.py --config-file projects/oriented_rcnn/configs/oriented_rcnn_r101_fpn_1x_dota_ms_with_flip_rotate_balance_cate.py

3. Run Tensorboard on Server-side, monitor on local:

server: tensorboard --logdir=work_dirs

local: ssh -L 16006:127.0.0.1:6006 lyx@localhost -p 9009

local browser: http://127.0.0.1:16006/
