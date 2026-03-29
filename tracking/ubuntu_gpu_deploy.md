# LCCTV Ubuntu GPU 部署说明

这份文档的目标是让你在一台有 NVIDIA GPU 的 Ubuntu Linux 机器上，从 0 开始把 LCCTV 新推理流程部署并跑通。

注意：

- 新的推理入口是 `tracking/run_inference.py`
- 不要使用 `tracking/test.py`
- 原因是 `tracking/test.py` 开头会强制关闭 CUDA，只适合当前这个 CPU 安全模式

## 1. 推荐目录结构

建议把项目和数据放成同级目录：

```text
/home/yourname/dz/
  LCCTV/
  DATA/
```

也就是：

- 项目目录：`/home/yourname/dz/LCCTV`
- 数据目录：`/home/yourname/dz/DATA`

`run_inference.py` 默认就会去找项目同级的 `DATA`。

## 2. 先检查 GPU 和驱动

先在 Ubuntu 终端执行：

```bash
nvidia-smi
```

如果能看到：

- 显卡型号
- 驱动版本
- 显存信息

说明 GPU 驱动基本正常。

如果这一步都不通，先不要继续配 Python 环境，先把驱动解决掉。

## 3. 安装 Miniconda 或 Anaconda

如果服务器上还没有 conda，建议装 Miniconda。

安装完成后，重新打开一个终端，确认：

```bash
conda --version
```

## 4. 创建推理环境

建议新建一个独立环境：

```bash
conda create -n lcctv-gpu python=3.10 -y
conda activate lcctv-gpu
```

## 5. 安装 PyTorch GPU 版本

优先推荐 CUDA 12.1 这组。

### 方案 A：CUDA 12.1

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 方案 B：CUDA 11.8

如果你的服务器环境更适合 11.8，就执行：

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

说明：

- 这里用的是 conda 官方推荐安装方式
- 一般不需要你自己单独安装完整 CUDA Toolkit
- 只要驱动足够新，PyTorch 这套 runtime 通常就可以直接跑

官方参考：

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)
- [PyTorch Previous Versions](https://docs.pytorch.org/get-started/previous-versions/)

## 6. 安装项目依赖

在 `lcctv-gpu` 环境里执行：

```bash
pip install opencv-python-headless scipy pandas easydict pyyaml timm yacs tensorboard tensorboardX einops
```

## 7. 验证 PyTorch 和 GPU

执行：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期：

- `torch.cuda.is_available()` 输出 `True`
- 能打印出 GPU 名称

如果这里输出 `False`，先不要继续跑推理，先排查：

- 驱动是否正常
- 安装的是不是 CPU 版 PyTorch
- `pytorch-cuda` 版本是否匹配

## 8. 准备项目代码

把代码拉到服务器，例如：

```bash
cd /home/yourname/dz
git clone <你的仓库地址> LCCTV
```

或者你已经有现成代码，就直接放到：

```text
/home/yourname/dz/LCCTV
```

进入项目目录：

```bash
cd /home/yourname/dz/LCCTV
```

## 9. 准备数据

把数据放到：

```text
/home/yourname/dz/DATA
```

每个序列目录应当长这样：

```text
DATA/
  1/
    groundtruth.txt
    imgs/
      00000001.jpg
      ...
  2_all/
    groundtruth.txt
    img/
      00000001.jpg
      ...
  3/
    groundtruth.txt
    img/
      00000001.jpg
      ...
```

要求：

- 每个序列目录必须有 `groundtruth.txt`
- 图片目录名支持 `img`、`imgs`、`images`

## 10. 准备权重文件

默认会去找这个位置：

```text
LCCTV/output/checkpoints/train/lcctv/B9_cae_center_all_ep300/lcctv_ep0300.pth.tar
```

例如：

```text
/home/yourname/dz/LCCTV/output/checkpoints/train/lcctv/B9_cae_center_all_ep300/lcctv_ep0300.pth.tar
```

如果你权重不在默认位置，也没关系，运行时用 `--checkpoint` 指定即可。

## 11. 开始推理

### 11.1 跑完整数据

```bash
cd /home/yourname/dz/LCCTV

python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/gpu_full
```

### 11.2 只跑一个序列

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --sequence 1 \
  --sequence-size 1=0.4
```

### 11.3 如果 `DATA` 不在默认位置

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --data-dir /data/project/DATA \
  --sequence-size 1=0.4 2_all=0.1 3=0.1
```

### 11.4 如果权重不在默认位置

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --checkpoint /home/yourname/models/lcctv_ep0300.pth.tar \
  --sequence-size 1=0.4 2_all=0.1 3=0.1
```

### 11.5 如果你只想先验证流程，不算烈度

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --skip-metrics
```

### 11.6 如果你希望框宽高允许变化

默认现在保持和旧逻辑一致，宽高锁定为首帧。

如果你想让模型输出的宽高也生效：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --allow-size-change \
  --sequence-size 1=0.4 2_all=0.1 3=0.1
```

## 12. 运行时你会看到什么输出

控制台现在会打印类似下面这种格式：

```text
Tracker: lcctv B9_cae_center_all_ep300 300 ,  Sequence: 1
地震烈度计算结果 - 序列: 1
烈度指数(Ii): 6.2, PGA: 0.395261..., PGV: 0.130363...
最大X位移: 0.005227...m, 最大Y位移: 0.009492...m
结果已保存到: /home/yourname/dz/LCCTV/output/inference/gpu_full/1
```

## 13. 输出文件怎么存

输出目录类似：

```text
output/inference/gpu_full/
  1/
    bboxes.txt
    time.txt
    summary.json
    report.txt
  2_all/
    bboxes.txt
    time.txt
    summary.json
    report.txt
  3/
    bboxes.txt
    time.txt
    summary.json
    report.txt
  summary.json
  run_report.txt
```

各文件作用：

- `bboxes.txt`
  每一帧的跟踪框，格式是 `x y w h`
- `time.txt`
  每一帧推理耗时
- `summary.json`
  结构化结果，适合程序读取
- `report.txt`
  每个序列的中文文本报告，内容和控制台打印风格一致
- 根目录 `summary.json`
  整次运行的结构化汇总
- 根目录 `run_report.txt`
  整次运行的文本汇总，适合直接查看或归档

## 14. 如果你想把控制台输出也保存到日志文件

可以直接这样运行：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/gpu_full | tee output/inference/gpu_full/console.log
```

这样会额外得到：

```text
output/inference/gpu_full/console.log
```

## 15. 常见问题

### 15.1 `torch.cuda.is_available()` 是 `False`

先检查：

```bash
nvidia-smi
```

再检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

通常原因是：

- 驱动没装好
- PyTorch 装成了 CPU 版
- 服务器驱动和你选的 CUDA runtime 不兼容

### 15.2 报 `Checkpoint was not found`

说明权重路径不对。

检查默认路径是否存在，或者直接手动指定：

```bash
--checkpoint /your/path/to/lcctv_ep0300.pth.tar
```

### 15.3 报 `No valid sequences found`

说明 `DATA` 目录结构不符合要求。

检查每个序列是否有：

- `groundtruth.txt`
- `img/` 或 `imgs/`

### 15.4 想看所有参数

```bash
python tracking/run_inference.py --help
```

## 16. 我推荐你在 Ubuntu 上最终用的标准命令

如果你的目录结构就是推荐结构，直接用这条：

```bash
conda activate lcctv-gpu
cd /home/yourname/dz/LCCTV
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/gpu_full
```

如果你跑通了，后面主要就看：

- `output/inference/gpu_full/run_report.txt`
- `output/inference/gpu_full/summary.json`

这两个文件。
