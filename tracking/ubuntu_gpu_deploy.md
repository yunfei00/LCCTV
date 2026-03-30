# LCCTV Ubuntu GPU 部署说明

这份文档的目标，是让你在一台有 NVIDIA GPU 的 Ubuntu Linux 机器上，从 0 开始把 LCCTV 的新推理流程部署并跑通。

注意：

- 新的推理入口是 `tracking/run_inference.py`
- 不要使用 `tracking/test.py`
- 原因是 `tracking/test.py` 里有旧的 CUDA 兼容处理逻辑，会干扰现在这套独立推理流程

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

`run_inference.py` 默认会去找项目同级的 `DATA`。

## 2. 先检查 GPU 和驱动

在 Ubuntu 终端执行：

```bash
nvidia-smi
```

如果能看到：

- 显卡型号
- 驱动版本
- 显存信息

说明驱动基本正常。

## 3. 安装 Miniconda

如果机器上还没有 `conda`，建议安装 Miniconda：

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda init bash
exec bash
```

确认：

```bash
conda --version
```

## 4. 创建推理环境

```bash
conda create -n lcctv-gpu python=3.10 -y
conda activate lcctv-gpu
```

## 5. 安装 PyTorch GPU 版本

优先推荐 CUDA 12.1：

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

如果你的服务器更适合 CUDA 11.8：

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 6. 固定 NumPy 版本

这一条非常重要。

你这次 Ubuntu 报错里已经明确出现了：

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

所以在 GPU 环境里，建议安装完 PyTorch 后立刻执行：

```bash
pip install --force-reinstall "numpy<2"
```

更稳一点的话，直接固定到：

```bash
pip install --force-reinstall numpy==1.26.4
```

## 7. 安装项目依赖

```bash
pip install opencv-python-headless scipy pandas easydict pyyaml timm yacs tensorboard tensorboardX einops
```

## 8. 验证 PyTorch 和 GPU

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期：

- `torch.cuda.is_available()` 输出 `True`
- 能打印出 GPU 名称

## 9. 准备项目代码

```bash
cd /home/yourname/dz
git clone <你的仓库地址> LCCTV
cd /home/yourname/dz/LCCTV
```

如果你已经有代码，就直接进入项目目录即可。

## 10. 更新到最新代码

新的独立推理入口已经不再依赖 `lib.train.data.processing_utils` 这条训练侧导入链。

所以在 Ubuntu 上请先同步最新代码：

```bash
git pull
```

## 11. 准备数据

把数据放到：

```text
/home/yourname/dz/DATA
```

目录结构类似：

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

## 12. 准备权重

默认权重路径是：

```text
/home/yourname/dz/LCCTV/output/checkpoints/train/lcctv/B9_cae_center_all_ep300/lcctv_ep0300.pth.tar
```

如果权重不在这里，运行时用：

```bash
--checkpoint /your/path/to/lcctv_ep0300.pth.tar
```

## 13. 先做一次冒烟测试

建议先跑一个短序列验证环境：

```bash
cd /home/yourname/dz/LCCTV

python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --sequence 1 \
  --max-frames 20 \
  --sequence-size 1=0.4 \
  --output-dir output/inference/gpu_smoke
```

## 14. 跑完整 GPU 推理

```bash
cd /home/yourname/dz/LCCTV

python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/gpu_full
```

## 15. 常用变体命令

只跑一个序列：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --sequence 1 \
  --sequence-size 1=0.4
```

数据目录不在默认位置：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --data-dir /data/project/DATA \
  --sequence-size 1=0.4 2_all=0.1 3=0.1
```

权重不在默认位置：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --checkpoint /home/yourname/models/lcctv_ep0300.pth.tar \
  --sequence-size 1=0.4 2_all=0.1 3=0.1
```

跳过烈度计算：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --skip-metrics
```

允许框宽高变化：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --allow-size-change \
  --sequence-size 1=0.4 2_all=0.1 3=0.1
```

## 16. 运行时输出

控制台会打印类似：

```text
Tracker: lcctv B9_cae_center_all_ep300 300 ,  Sequence: 1
地震烈度计算结果 - 序列: 1
烈度指数(Ii): 6.2, PGA: 0.395261..., PGV: 0.130363...
最大X位移: 0.005227...m, 最大Y位移: 0.009492...m
结果已保存到: /home/yourname/dz/LCCTV/output/inference/gpu_full/1
```

## 17. 输出文件

目录结构类似：

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

含义：

- `bboxes.txt`：每帧预测框
- `time.txt`：每帧耗时
- `summary.json`：结构化结果
- `report.txt`：单序列中文报告
- 根目录 `summary.json`：整次运行的结构化汇总
- 根目录 `run_report.txt`：整次运行的中文汇总

## 18. 把控制台输出保存成日志

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/gpu_full | tee output/inference/gpu_full/console.log
```

## 19. 常见问题

### 19.1 `torch.cuda.is_available()` 是 `False`

先检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

常见原因：

- 驱动没装好
- PyTorch 装成了 CPU 版
- CUDA runtime 和驱动不匹配

### 19.2 报 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

执行：

```bash
pip install --force-reinstall numpy==1.26.4
```

### 19.3 报 `ModuleNotFoundError: No module named 'lib.train.data.processing_utils'`

先更新代码：

```bash
git pull
```

再重新运行 `tracking/run_inference.py`。

### 19.4 报 `Checkpoint was not found`

检查默认权重路径，或者显式传：

```bash
--checkpoint /your/path/to/lcctv_ep0300.pth.tar
```

### 19.5 报 `No valid sequences found`

检查每个序列目录是否都有：

- `groundtruth.txt`
- `img/`、`imgs/` 或 `images/`

### 19.6 想看所有参数

```bash
python tracking/run_inference.py --help
```

## 20. 最推荐的最终命令

```bash
conda activate lcctv-gpu
cd /home/yourname/dz/LCCTV

python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cuda:0 \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/gpu_full
```

跑完后优先看：

- `output/inference/gpu_full/run_report.txt`
- `output/inference/gpu_full/summary.json`
