## 0 Install env:

```shell
# 安装 OpenMPI, 为多卡训练做准备
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
# 安装环境
conda create --name jittor python=3.8
cd RS_detection
pip install -r requirements.txt
# 以 develop 方式安装比赛提供的 JDet
python setup.py develop
```

- Python 必须 3.8 及以上

## 1 Data preprocess:

在比赛页面下载FAIR1M 1.5数据集，解压后修改成以下形式：

```
{DATASET_PATH}
    |
    └──data
        ├──train
        |     ├──images
        |     |    ├──1.tif
        |     |    └──...
        |     └──labelXml
        |          ├──1.xml
        |          └──...
        └──test
                └──images
                    ├──1.tif
                    └──...
```

- testa 也要改成 test

进入 `configs/preprocess/fair1m_1_5_preprocess_config_ms_le90.py`文件，修改这个文件中的三个路径参数为

```python
source_fair_dataset_path='{DATASET_PATH}/data'
source_dataset_path='{DATASET_PATH}/dota'
target_dataset_path='{DATASET_PATH}/preprocessed'
```

其中 `{DATASET_PATH}`与前一步相同。

在 `JDet`目录下执行 `python tools/preprocess.py --config-file configs/preprocess/fair1m_1_5_preprocess_config_ms_le90.py`，即可自动进行数据预处理。

## 2 Run Oriented R-CNN:

修改 `projects/oriented_rcnn/configs` 下的 config 文件 `oriented_rcnn_r101_fpn_1x_dota_ms_with_flip_rotate_balance_cate.py`

把 `dataset_root` 改成自己的数据路径。

```shell
python tools/run_net.py --config-file projects/oriented_rcnn/configs/oriented_rcnn_r101_fpn_1x_dota_ms_with_flip_rotate_balance_cate.py
```

3. Run Tensorboard on Server-side, monitor on local:

server: tensorboard --logdir=work_dirs

local: ssh -L 16006:127.0.0.1:6006 lyx@localhost -p 9009

local browser: http://127.0.0.1:16006/

## 3 FAIR1M 的离线验证集

data/fair1m/splits 文件夹下存有 val1k.txt 等多个 txt 文件，生成 val 的命令为

```shell
python tools/preprocess.py --config-file configs/preprocess/fair1m_2_preprocess_config_ms_le90_grok.py
```

fair1m 的数据按照如下存放：

```
{DATASET_PATH}
    |
    └──data
        ├──train
        |     ├──images
        |     |    ├──1.tif
        |     |    └──...
        |     └──labelXml
        |          ├──1.xml
        |          └──...
        ├──val
        |     ├──images
        |     |    ├──1.tif
        |     |    └──...
        |     └──labelXml
        |          ├──1.xml
        |          └──...
```

在我的电脑上，{DATASET_PATH} 为 '/yimian'，需要替换成相应的路径