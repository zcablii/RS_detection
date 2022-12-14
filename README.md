## 0 环境安装：


```shell
https://github.com/zcablii/RS_detection.git # 克隆代码
cd RS_detection
python -m pip install -r requirements.txt # 安装环境
python setup.py develop
```


## 1 数据集处理:

我们的训练模型是在 FAIR1M2.0 遥感监测数据集的基础上训练的，数据集可在https://www.gaofen-challenge.com/benchmark 下载获得。

FAIR1M2.0 数据集包含train， validation 和 test集，我们将有标签的train和validation合并，并命名为train_color，并将数据集中分辨率大于2,500 * 2500 的进行灰度化处理，并将灰度化的副本单独存放为train_gray用于后续处理。


train_color和train_gray修改成以下形式：

    {DATASET_PATH}
        |
        └──data
            |
            ├train_color
            |   |
            |   └──train
            |     ├──images
            |     |    ├──1.tif
            |     |    └──...
            |     └──labelXml
            |          ├──1.xml
            |          └──...
            | 
            └train_gray
                |
                └──train
                    ├──images
                    |    ├──1.tif
                    |    └──...
                    └──labelXml
                        ├──1.xml
                        └──...


其中`{DATASET_PATH}`为数据集路径，用户可以自行选择。

**注意：直接解压数据集得到的文件树可能与说明不同（如labelXml、test的名称），请将其修改为说明中的格式。**

进入`configs/preprocess/fair1m_1_5_preprocess_config.py`文件，修改这个文件中的三个路径参数为

```python
source_fair_dataset_path='{DATASET_PATH}/data'
source_dataset_path='{DATASET_PATH}/dota'
target_dataset_path='{DATASET_PATH}/preprocessed'
```

在`configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_train_color.py` 中修改以下三个数据路径参数, 其中`{DATASET_PATH}`与前一步相同。
```python
    source_fair_dataset_path='{DATASET_PATH}/data/train_color'
    source_dataset_path='{DATASET_PATH}/dota_train_color'
    target_dataset_path='{DATASET_PATH}/preprocessed_train_color'
```
并运行`python tools/preprocess.py --config-file configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_train_color.py`，即可自动进行FAIR1M2.0数据预处理。

相似的，修改和运行`configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_train_gray.py`，即可自动进行灰度图的数据预处理。
最后将处理后的train_color 与 train_gray 合并，作为本次比赛的train训练集（本报告后续提及的train数据默认指此数据集）。

下载比赛官方提供的测试集并解压，通过修改和运行`configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_test.py`即可自动进行测试的数据预处理。

## 2 模型训练:

修改 `./configs/orcnn_van3_7_anchor_swa_1.py` 和`./configs/orcnn_van3_7_anchor_swa_2.py` config 文件，把 `dataset_root` 改成数据存放路径，并根据具体情况修改训练集和测试集的数据目录。

这两个配置文件所训练的模型完全相同，我们训练两个模型的目的是为了后期进行模型融合。两个模型都训练9个epoch，lr初始为0.0001，其于第8个epoch下降10倍。

单个模型训练在8卡上进行：
```shell
mpirun --allow-run-as-root -np 8 python tools/run_net.py --config-file configs/orcnn_van3_7_anchor_swa_1.py
mpirun --allow-run-as-root -np 8 python tools/run_net.py --config-file configs/orcnn_van3_7_anchor_swa_2.py
```

通过对第8和第9个epoch的模型快照进行权重融合生成新的单模型:
```shell
python tools/get_SWA_model.py --model_dir work_dirs/orcnn_van3_7_anchor_swa_1/checkpoints/ --starting_model_id 8 --ending_model_id 9 --save_dir work_dirs/orcnn_van3_7_anchor_swa_1/checkpoints/
python tools/get_SWA_model.py --model_dir work_dirs/orcnn_van3_7_anchor_swa_2/checkpoints/ --starting_model_id 8 --ending_model_id 9 --save_dir work_dirs/orcnn_van3_7_anchor_swa_2/checkpoints/
```

### 预训练模型
我们使用了在Imagenet上预训练的Visual Attention Network（VAN）作为骨干网络，VAN权重文件可在 `https://huggingface.co/Visual-Attention-Network/VAN-Large-original/resolve/main/van_large_839.pth.tar` 下载。

## 3 模型测试
**测试前请确保RS_detection文件夹下没有submit_zips和data文件夹。当换测试集时，请确保删除submit_zips和data文件夹。**

进入RS_detection文件夹后，进入Jittor虚拟环境：
```shell
    cd ~/RS_detection
    source activate jittor
```

### 3.1 快速测试：
处理测试数据和运行测试操作而得到最终结果可直接通过运行 `python test.py --test_path PATH_TO_TEST` 来实现。其中 `PATH_TO_TEST` 为解压后测试集目录，且需要遵循**测试数据处理**中的文件结构。`test.py`脚本包含**测试数据处理**和**运行测试**两个阶段，具体如下所述。


### 3.2 测试数据处理：
解压测试数据后修改成以下形式：
```
    {DATASET_PATH}
        |
        └──data 
            └──test
                  └──images
                       ├──1.tif
                       └──...
```

其中`{DATASET_PATH}`为数据集路径，用户可以自行选择。
在
`configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_test.py` 中修改以下三个数据路径参数
```python
    source_fair_dataset_path='{DATASET_PATH}/data'
    source_dataset_path='{DATASET_PATH}/dota_test'
    target_dataset_path='{DATASET_PATH}/preprocessed_test'
```
其中source_fair_dataset_path设为测试数据集的路径。并运行
`python tools/preprocess.py --config-file configs/preprocess/fair1m_1_5_preprocess_config_ms_le90_test.py`，即可自动进行测试数据预处理。

### 3.3 运行测试:
修改 `./configs/orcnn_van3_for_test_1.py` 和`./configs/orcnn_van3_for_test_2.py` config 文件，把 `dataset_root` 改成数据存放路径，即`fair1m_1_5_preprocess_config_ms_le90_test.py`中的`target_dataset_path`，并根据具体情况修改训练集和测试集的数据目录。然后运行：
```shell
python tools/run_net.py --config-file configs/orcnn_van3_for_test_1.py --task test
python tools/run_net.py --config-file configs/orcnn_van3_for_test_2.py --task test
```
此时，请检查`./submit_zips`文件夹下有且仅有两个文件，分别为`orcnn_van3_for_test_1_epoch0.csv` 和 `orcnn_van3_for_test_2_epoch0.csv`。然后运行
```shell
python merge.py
```
完成对两个模型输出结果的融合。
最终结果将被存放于`./csv_merge/merged_result.csv`，提交线上可以得到mAP为0.8111的结果。
