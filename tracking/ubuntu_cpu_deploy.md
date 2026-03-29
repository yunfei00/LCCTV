# LCCTV Ubuntu CPU 部署说明

这份文档的目标，是让你在一台没有 GPU 的 Ubuntu Linux 机器上，从 0 开始把 LCCTV 的新推理流程部署并跑通。

适用场景：

- Ubuntu 服务器或工作站
- 没有 NVIDIA GPU，或者你就是只想用 CPU 跑
- 想直接使用新的推理入口 `tracking/run_inference.py`

注意：

- CPU 环境请直接使用 `tracking/run_inference.py`
- 不要使用 `tracking/test.py`
- 原因是 `tracking/test.py` 里带有旧的 CUDA 兼容处理逻辑，不适合作为现在这套独立推理入口

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

新的推理脚本默认会去找项目同级的 `DATA` 目录。

## 2. 先准备 Ubuntu 基础环境

先更新系统包索引：

```bash
sudo apt update
```

安装常用基础工具：

```bash
sudo apt install -y git wget curl build-essential
```

如果你后面要查看图片、处理 OpenCV 相关依赖，也建议顺手装上：

```bash
sudo apt install -y libglib2.0-0 libsm6 libxext6 libxrender1
```

## 3. 安装 Miniconda

如果你的 Ubuntu 上还没有 `conda`，建议安装 Miniconda。

下载官方安装脚本：

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

执行安装：

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中：

- 连续按回车查看协议
- 输入 `yes`
- 安装路径可以直接用默认值

安装完成后，让 shell 加载 conda：

```bash
source ~/miniconda3/bin/activate
conda init bash
exec bash
```

确认安装成功：

```bash
conda --version
```

如果你已经装过 conda，可以直接跳到下一步。

## 4. 创建独立的 CPU 推理环境

建议单独创建一个环境，避免污染你机器上原有的 Python：

```bash
conda create -n lcctv-cpu python=3.10 -y
conda activate lcctv-cpu
```

## 5. 安装 PyTorch CPU 版本

CPU 环境请安装 PyTorch 官方 CPU wheel。

推荐命令：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

如果你想安装一个固定版本，也可以使用 PyTorch 官方历史版本页里的 CPU 命令，例如：

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
```

## 6. 安装项目依赖

在 `lcctv-cpu` 环境里继续执行：

```bash
pip install opencv-python-headless scipy pandas easydict pyyaml timm yacs tensorboard tensorboardX einops
```

## 7. 验证 CPU 环境是否正常

执行下面的检查命令：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

预期结果：

- 能正常打印 PyTorch 版本号
- `torch.cuda.is_available()` 输出 `False`

这里输出 `False` 是正常的，因为这是 CPU 环境。

你也可以再试一条更直接的 CPU 张量命令：

```bash
python -c "import torch; x=torch.rand(2,3); print(x.device); print(x)"
```

如果输出设备是 `cpu`，说明 PyTorch CPU 已经可用。

## 8. 准备项目代码

把代码放到推荐目录，例如：

```bash
cd /home/yourname/dz
git clone <你的仓库地址> LCCTV
```

如果你已经有代码，也可以直接保证项目路径是：

```text
/home/yourname/dz/LCCTV
```

进入项目目录：

```bash
cd /home/yourname/dz/LCCTV
```

## 9. 准备数据目录

把数据放到：

```text
/home/yourname/dz/DATA
```

每个序列目录需要长这样：

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
- `groundtruth.txt` 的第一行会被当成首帧初始化框，格式是 `x,y,w,h`

## 10. 准备模型权重

默认权重位置是：

```text
/home/yourname/dz/LCCTV/output/checkpoints/train/lcctv/B9_cae_center_all_ep300/lcctv_ep0300.pth.tar
```

如果你的权重就是放在这个位置，那运行时不用额外指定。

如果权重在别的地方，后面运行时加：

```bash
--checkpoint /your/path/to/lcctv_ep0300.pth.tar
```

## 11. 先做一次最小化检查

建议先跑一个短序列，确认环境、路径、权重都没问题：

```bash
cd /home/yourname/dz/LCCTV

python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --sequence 1 \
  --max-frames 20 \
  --sequence-size 1=0.4 \
  --output-dir output/inference/ubuntu_cpu_smoke
```

这一步的目的不是拿最终结果，而是先确认：

- 脚本能启动
- 数据能扫描到
- 权重能加载
- 输出目录能正常写入

## 12. 跑完整 CPU 推理

如果上一步没问题，就可以跑完整数据。

推荐命令：

```bash
cd /home/yourname/dz/LCCTV

python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_full
```

这里：

- `--device cpu` 表示明确使用 CPU
- `--sequence-size` 是每个序列的物理尺寸参数
- `--output-dir` 是这次推理的输出目录

## 13. 如果你只想跑一个序列

例如只跑序列 `1`：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --sequence 1 \
  --sequence-size 1=0.4 \
  --output-dir output/inference/ubuntu_cpu_seq1
```

## 14. 如果 `DATA` 不在默认位置

假设你的数据放在 `/data/project/DATA`：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --data-dir /data/project/DATA \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_full
```

## 15. 如果权重不在默认位置

例如：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --checkpoint /home/yourname/models/lcctv_ep0300.pth.tar \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_full
```

## 16. 如果你只想先验证流程，不计算烈度指标

可以先跳过地震指标计算：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --skip-metrics \
  --output-dir output/inference/ubuntu_cpu_nometrics
```

## 17. 如果你希望框宽高允许变化

当前默认行为和旧逻辑保持一致，也就是宽高锁定为首帧大小。

如果你想让模型预测出来的宽高也生效，可以加：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --allow-size-change \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_size_change
```

## 18. 运行时你会看到什么输出

控制台现在会打印类似下面这种内容：

```text
Tracker: lcctv B9_cae_center_all_ep300 300 ,  Sequence: 1
地震烈度计算结果 - 序列: 1
烈度指数(Ii): 6.2, PGA: 0.395261..., PGV: 0.130363...
最大X位移: 0.005227...m, 最大Y位移: 0.009492...m
结果已保存到: /home/yourname/dz/LCCTV/output/inference/ubuntu_cpu_full/1
```

## 19. 输出文件会保存到哪里

如果你使用：

```bash
--output-dir output/inference/ubuntu_cpu_full
```

那么输出目录结构类似：

```text
output/inference/ubuntu_cpu_full/
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

这些文件的作用：

- `bboxes.txt`
  每一帧的预测框，格式是 `x y w h`
- `time.txt`
  每一帧的推理耗时
- `summary.json`
  当前序列的结构化结果
- `report.txt`
  当前序列的中文文本报告，适合直接打开查看
- 根目录 `summary.json`
  整次运行的结构化汇总
- 根目录 `run_report.txt`
  整次运行的中文汇总文本

## 20. 如果你还想把控制台输出保存成日志

可以在命令后面接 `tee`：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_full | tee output/inference/ubuntu_cpu_full/console.log
```

这样会多出一个文件：

```text
output/inference/ubuntu_cpu_full/console.log
```

## 21. 常见问题排查

### 21.1 报 `ModuleNotFoundError: No module named ...`

说明依赖没装全。先确认你已经进入了正确环境：

```bash
conda activate lcctv-cpu
```

然后重新安装依赖：

```bash
pip install opencv-python-headless scipy pandas easydict pyyaml timm yacs tensorboard tensorboardX einops
```

### 21.2 报 `Checkpoint was not found`

说明权重路径不对。

先检查默认路径是否存在：

```bash
ls /home/yourname/dz/LCCTV/output/checkpoints/train/lcctv/B9_cae_center_all_ep300/
```

如果不在默认位置，就显式指定：

```bash
--checkpoint /your/path/to/lcctv_ep0300.pth.tar
```

### 21.3 报 `No valid sequences found`

说明 `DATA` 目录结构不符合要求。

请检查每个序列目录是否都有：

- `groundtruth.txt`
- `img/`、`imgs/` 或 `images/`

### 21.4 CPU 跑得太慢

这是正常现象，因为没有 GPU。

你可以先这样做：

- 先用 `--max-frames 20` 或 `--max-frames 100` 做小范围验证
- 只跑一个序列，确认配置无误后再跑全量
- 给推理进程单独留出 CPU 资源，避免和别的高负载程序抢占

### 21.5 想看所有参数说明

直接执行：

```bash
python tracking/run_inference.py --help
```

## 22. 最推荐你在 Ubuntu CPU 上使用的完整命令

如果你的目录结构和本文一致，最标准的运行方式就是：

```bash
conda activate lcctv-cpu
cd /home/yourname/dz/LCCTV

python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_full
```

跑完以后，你最应该先看这两个文件：

- `output/inference/ubuntu_cpu_full/run_report.txt`
- `output/inference/ubuntu_cpu_full/summary.json`

## 23. 参考

PyTorch CPU 安装命令参考官方文档：

- [Start Locally](https://docs.pytorch.org/get-started/locally/)
- [Previous Versions](https://docs.pytorch.org/get-started/previous-versions/)
