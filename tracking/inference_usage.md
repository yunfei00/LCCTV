# LCCTV Inference Usage

This repository now includes a standalone inference entrypoint.

Detailed deployment guides:

- Ubuntu CPU: `tracking/ubuntu_cpu_deploy.md`
- Ubuntu GPU: `tracking/ubuntu_gpu_deploy.md`

Recommended CPU environment:

```bash
conda create -n lcctv-cpu python=3.10 -y
conda activate lcctv-cpu
pip install --force-reinstall numpy==1.26.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless scipy pandas easydict pyyaml timm yacs tensorboard tensorboardX einops
```

If Ubuntu reports `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`,
repair the environment with:

```bash
pip install --force-reinstall numpy==1.26.4
```

Standalone inference entrypoint:

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300
```

Useful options:

```bash
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --sequence 1
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --max-frames 100
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --device cpu
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --sequence-size 1=0.4 2_all=0.1 3=0.1
python tracking/run_inference.py B9_cae_center_all_ep300 --epoch 300 --device cpu --sequence-size 1=0.4 2_all=0.1 3=0.1 --output-dir output/inference/cpu_full
```

Default paths:

- project: repository root
- data: sibling `DATA` directory
- output: `output/inference/lcctv/<tracker_param>_epXXXX`

Per-sequence outputs:

- `bboxes.txt`
- `time.txt`
- `summary.json`
- `report.txt`

Run summary:

- `summary.json` in the run output directory
- `run_report.txt` in the run output directory
