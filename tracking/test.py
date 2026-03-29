import os

# ========= CPU-ONLY: hard disable any CUDA usage =========
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import torch

    # 1) 禁止任何 CUDA 初始化/探测
    torch.cuda._lazy_init = lambda *a, **k: None
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

    # 2) 如果代码里有人直接调用 .cuda()，让它在 CPU 上变成 no-op
    if hasattr(torch, "Tensor"):
        torch.Tensor.cuda = lambda self, *a, **k: self
    if hasattr(torch.nn, "Module"):
        torch.nn.Module.cuda = lambda self, *a, **k: self

except Exception:
    pass
# =========================================================
# =======================================================

import sys
import argparse


# ================= CPU-ONLY 规范 =================
# num_gpus <= 0 视为 CPU 模式，强制禁用 CUDA
# ================================================

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset


def run_tracker(
    tracker_name,
    tracker_param,
    run_id=None,
    dataset_name='otb',
    sequence=None,
    debug=0,
    threads=0,
    num_gpus=0
):
    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [
        (tracker_name, tracker_param, dataset_name, ep_id)
        for ep_id in run_id
    ]

    run_dataset(
        dataset,
        trackers,
        debug,
        threads,
        num_gpus=max(0, num_gpus)   # 明确 CPU=0
    )


def main():
    #lcctv B9_cae_center_all_ep300 --dataset my --threads 2 --num_gpus 1 --ep 300
    parser = argparse.ArgumentParser(description='Run tracker (CPU-only safe).')
    parser.add_argument('tracker_name', type=str)
    parser.add_argument('tracker_param', type=str)
    parser.add_argument('--dataset_name', type=str, default='otb')
    parser.add_argument('--sequence', type=str, default=None)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=0)  # CPU 默认
    parser.add_argument('--ep', nargs='+', type=int, default=[100])

    args = parser.parse_args()

    # ===== CPU 模式：彻底禁用 CUDA =====
    if args.num_gpus <= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(
        args.tracker_name,
        args.tracker_param,
        args.ep,
        args.dataset_name,
        seq_name,
        args.debug,
        args.threads,
        num_gpus=args.num_gpus
    )


if __name__ == '__main__':
    main()

