import os
import argparse
import random
import subprocess

# ================= CPU-ONLY 规范 =================
# single 模式 = CPU
# 不允许在代码中写死 CUDA_VISIBLE_DEVICES
# ================================================


def parse_args():
    parser = argparse.ArgumentParser(description='CPU-safe training launcher')

    parser.add_argument('--script', type=str, required=True)
    parser.add_argument('--config', type=str, default='baseline')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"],
                        default="single")   # CPU 默认
    parser.add_argument('--nproc_per_node', type=int, default=1)
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)
    parser.add_argument('--script_prv', type=str, default="")
    parser.add_argument('--config_prv', type=str, default="")
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)
    parser.add_argument('--script_teacher', type=str, default="")
    parser.add_argument('--config_teacher', type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()

    # ===== CPU-only：禁用 CUDA =====
    if args.mode == "single":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.mode == "single":
        train_cmd = (
            f"python lib/train/run_training.py "
            f"--script {args.script} "
            f"--config {args.config} "
            f"--save_dir {args.save_dir} "
            f"--use_lmdb {args.use_lmdb} "
            f"--script_prv {args.script_prv} "
            f"--config_prv {args.config_prv} "
            f"--distill {args.distill} "
            f"--script_teacher {args.script_teacher} "
            f"--config_teacher {args.config_teacher} "
            f"--use_wandb {args.use_wandb}"
        )

    elif args.mode == "multiple":
        train_cmd = (
            f"torchrun --standalone "
            f"--nnodes=1 "
            f"--nproc_per_node {args.nproc_per_node} "
            f"--master_port {random.randint(10000, 50000)} "
            f"lib/train/run_training.py "
            f"--script {args.script} "
            f"--config {args.config} "
            f"--save_dir {args.save_dir}"
        )

    else:
        raise ValueError("CPU-only currently supports mode=single")

    subprocess.run(train_cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
