# LCCTV Ubuntu CPU 部署说明

这份文档的目标，是让你在一台没有 GPU 的 Ubuntu Linux 机器上，从 0 开始把 LCCTV 的新推理流程部署并跑通。

注意：

- CPU 环境请直接使用 `tracking/run_inference.py`
- 不要使用 `tracking/test.py`
- 原因是 `tracking/test.py` 里的旧逻辑并不是为现在这套独立推理入口准备的

## 1. 推荐目录结构

```text
/home/yourname/dz/
  LCCTV/
  DATA/
```

也就是：

- 项目目录：`/home/yourname/dz/LCCTV`
- 数据目录：`/home/yourname/dz/DATA`

## 2. 准备 Ubuntu 基础环境

```bash
sudo apt update
sudo apt install -y git wget curl build-essential
sudo apt install -y libglib2.0-0 libsm6 libxext6 libxrender1
```

## 3. 安装 Miniconda

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

## 4. 创建 CPU 推理环境

```bash
conda create -n lcctv-cpu python=3.10 -y
conda activate lcctv-cpu
```

## 5. 固定 NumPy 版本

Ubuntu 上建议一开始就固定到 `NumPy 1.26.4`，避免出现 `NumPy 2.x` 兼容问题：

```bash
pip install --force-reinstall numpy==1.26.4
```

## 6. 安装 PyTorch CPU 版本

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

如果你想装固定版本，也可以：

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
```

## 7. 安装项目依赖

```bash
pip install opencv-python-headless scipy pandas easydict pyyaml timm yacs tensorboard tensorboardX einops
```

## 8. 验证 CPU 环境

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import torch; x=torch.rand(2,3); print(x.device); print(x)"
```

预期：

- `torch.cuda.is_available()` 输出 `False`
- 张量设备显示为 `cpu`

## 9. 准备项目代码

```bash
cd /home/yourname/dz
git clone <你的仓库地址> LCCTV
cd /home/yourname/dz/LCCTV
git pull
```

## 10. 准备数据

把数据放到：

```text
/home/yourname/dz/DATA
```

结构类似：

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

## 11. 准备权重

默认路径：

```text
/home/yourname/dz/LCCTV/output/checkpoints/train/lcctv/B9_cae_center_all_ep300/lcctv_ep0300.pth.tar
```

如果不在默认位置，运行时加：

```bash
--checkpoint /your/path/to/lcctv_ep0300.pth.tar
```

## 12. 先跑一个冒烟测试

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

## 13. 跑完整 CPU 推理

```bash
cd /home/yourname/dz/LCCTV

python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_full
```

## 14. 常用变体命令

只跑一个序列：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --sequence 1 \
  --sequence-size 1=0.4 \
  --output-dir output/inference/ubuntu_cpu_seq1
```

数据目录不在默认位置：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --data-dir /data/project/DATA \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_full
```

跳过烈度计算：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --skip-metrics \
  --output-dir output/inference/ubuntu_cpu_nometrics
```

允许框宽高变化：

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --allow-size-change \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_size_change
```

## 15. 运行时输出

控制台会打印类似：

```text
Tracker: lcctv B9_cae_center_all_ep300 300 ,  Sequence: 1
地震烈度计算结果 - 序列: 1
烈度指数(Ii): 6.2, PGA: 0.395261..., PGV: 0.130363...
最大X位移: 0.005227...m, 最大Y位移: 0.009492...m
结果已保存到: /home/yourname/dz/LCCTV/output/inference/ubuntu_cpu_full/1
```

## 16. 输出文件

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

## 17. 保存控制台日志

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_full | tee output/inference/ubuntu_cpu_full/console.log
```

## 18. 常见问题

### 18.1 报 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

执行：

```bash
pip install --force-reinstall numpy==1.26.4
```

### 18.2 报 `ModuleNotFoundError: No module named 'lib.train.data.processing_utils'`

先更新代码：

```bash
git pull
```

然后重新运行 `tracking/run_inference.py`。

### 18.3 报 `Checkpoint was not found`

显式指定权重路径：

```bash
--checkpoint /your/path/to/lcctv_ep0300.pth.tar
```

### 18.4 报 `No valid sequences found`

检查每个序列目录是否都有：

- `groundtruth.txt`
- `img/`、`imgs/` 或 `images/`

### 18.5 CPU 跑得太慢

这是正常现象。建议先用：

- `--max-frames 20`
- `--max-frames 100`
- `--sequence 1`

先做小范围验证。

### 18.6 想看所有参数

```bash
python tracking/run_inference.py --help
```

## 19. 最推荐的最终命令

```bash
conda activate lcctv-cpu
cd /home/yourname/dz/LCCTV

python tracking/run_inference.py B9_cae_center_all_ep300 \
  --epoch 300 \
  --device cpu \
  --sequence-size 1=0.4 2_all=0.1 3=0.1 \
  --output-dir output/inference/ubuntu_cpu_full
```

跑完后优先看：

- `output/inference/ubuntu_cpu_full/run_report.txt`
- `output/inference/ubuntu_cpu_full/summary.json`
