import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List
import warnings

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm(\..*)?$")

from lib.inference import LcctvInferencer, discover_sequences


TRACKER_NAME = "lcctv"


def parse_args():
    default_data_dir = PROJECT_DIR.parent / "DATA"

    parser = argparse.ArgumentParser(description="Run LCCTV inference on a data directory.")
    parser.add_argument("tracker_param", type=str, help="Experiment yaml name, for example B9_cae_center_all_ep300")
    parser.add_argument("--epoch", type=int, required=True, help="Checkpoint epoch number, for example 300")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir, help="Directory that contains sequence folders")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR, help="LCCTV project root")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override")
    parser.add_argument("--device", type=str, default=None, help="torch device, for example cuda, cuda:0, or cpu")
    parser.add_argument("--sequence", nargs="*", default=None, help="Only run specific sequence names")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per sequence")
    parser.add_argument("--fps", type=float, default=24.0, help="Default video fps for earthquake metrics")
    parser.add_argument("--default-size", type=float, default=0.4, help="Default scale factor passed into the metrics module")
    parser.add_argument(
        "--sequence-size",
        nargs="*",
        default=None,
        help="Per-sequence metric scale override, format: name=value",
    )
    parser.add_argument("--skip-metrics", action="store_true", help="Skip earthquake metric calculation")
    parser.add_argument(
        "--allow-size-change",
        action="store_true",
        help="Disable the legacy behavior that keeps bbox width and height fixed to the init box",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sequence_size_overrides = parse_sequence_size_overrides(args.sequence_size or [])
    sequences = discover_sequences(args.data_dir, include_names=args.sequence, max_frames=args.max_frames)

    output_dir = args.output_dir or (
        args.project_dir
        / "output"
        / "inference"
        / "lcctv"
        / f"{args.tracker_param}_ep{args.epoch:04d}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inferencer = LcctvInferencer(
        project_dir=args.project_dir,
        tracker_param=args.tracker_param,
        epoch=args.epoch,
        checkpoint_path=args.checkpoint,
        device=args.device,
        lock_bbox_size=not args.allow_size_change,
    )

    results = []
    report_blocks: List[str] = []
    for sequence in sequences:
        print(f"Tracker: {TRACKER_NAME} {args.tracker_param} {args.epoch} ,  Sequence: {sequence.name}")
        result = inferencer.run_sequence(
            sequence,
            metric_provider=None if args.skip_metrics else lambda seq: {
                "fps": args.fps,
                "size": sequence_size_overrides.get(seq.name, args.default_size),
            },
        )

        sequence_dir = output_dir / sequence.name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        report_text = build_sequence_report_text(TRACKER_NAME, args.tracker_param, args.epoch, result)
        save_sequence_outputs(sequence_dir, result, report_text)
        results.append(result)
        report_blocks.append(report_text)
        print(report_text)
        print(f"结果已保存到: {sequence_dir}")

    save_run_summary(
        output_dir=output_dir,
        tracker_param=args.tracker_param,
        epoch=args.epoch,
        device=str(inferencer.device),
        data_dir=args.data_dir.resolve(),
        results=results,
        default_size=args.default_size,
        fps=args.fps,
        sequence_size_overrides=sequence_size_overrides,
        lock_bbox_size=not args.allow_size_change,
        run_report_text="\n\n".join(report_blocks),
    )

    print(f"全部序列处理完成，共 {len(results)} 个序列")
    print(f"汇总输出目录: {output_dir}")


def parse_sequence_size_overrides(items: List[str]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --sequence-size entry: {item}. Expected name=value")
        name, value = item.split("=", 1)
        overrides[name] = float(value)
    return overrides


def save_sequence_outputs(output_dir: Path, result, report_text: str):
    np.savetxt(output_dir / "bboxes.txt", np.asarray(result.bboxes, dtype=np.float64), delimiter="\t", fmt="%.6f")
    np.savetxt(output_dir / "time.txt", np.asarray(result.timings, dtype=np.float64), delimiter="\t", fmt="%.6f")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)
    with (output_dir / "report.txt").open("w", encoding="utf-8") as file:
        file.write(report_text + "\n")


def build_sequence_report_text(tracker_name: str, tracker_param: str, epoch: int, result) -> str:
    lines = [
        f"Tracker: {tracker_name} {tracker_param} {epoch} ,  Sequence: {result.sequence_name}",
    ]
    if result.metrics is not None:
        lines.extend(
            [
                f"地震烈度计算结果 - 序列: {result.sequence_name}",
                f"烈度指数(Ii): {result.metrics.intensity}, PGA: {result.metrics.pga}, PGV: {result.metrics.pgv}",
                f"最大X位移: {result.metrics.max_x}m, 最大Y位移: {result.metrics.max_y}m",
            ]
        )
    elif result.metrics_error:
        lines.extend(
            [
                f"地震烈度计算结果 - 序列: {result.sequence_name}",
                f"计算失败: {result.metrics_error}",
            ]
        )
    else:
        lines.extend(
            [
                f"地震烈度计算结果 - 序列: {result.sequence_name}",
                "本次运行跳过了地震烈度计算",
            ]
        )
    return "\n".join(lines)


def save_run_summary(
    output_dir: Path,
    tracker_param: str,
    epoch: int,
    device: str,
    data_dir: Path,
    results,
    default_size: float,
    fps: float,
    sequence_size_overrides: Dict[str, float],
    lock_bbox_size: bool,
    run_report_text: str,
):
    payload = {
        "tracker_param": tracker_param,
        "epoch": epoch,
        "device": device,
        "data_dir": str(data_dir),
        "default_size": default_size,
        "fps": fps,
        "sequence_size_overrides": sequence_size_overrides,
        "lock_bbox_size": lock_bbox_size,
        "sequences": [result.to_dict() for result in results],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    with (output_dir / "run_report.txt").open("w", encoding="utf-8") as file:
        file.write(run_report_text.rstrip() + "\n")


if __name__ == "__main__":
    main()
