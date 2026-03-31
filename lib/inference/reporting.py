from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


def save_sequence_outputs(output_dir: Path, result, report_text: str):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "bboxes.txt", np.asarray(result.bboxes, dtype=np.float64), delimiter="\t", fmt="%.6f")
    np.savetxt(output_dir / "time.txt", np.asarray(result.timings, dtype=np.float64), delimiter="\t", fmt="%.6f")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)
    with (output_dir / "report.txt").open("w", encoding="utf-8") as file:
        file.write(report_text.rstrip() + "\n")


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
    results: Sequence,
    default_size: float,
    fps: float,
    sequence_size_overrides: Dict[str, float],
    lock_bbox_size: bool,
    run_report_text: str,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tracker_param": tracker_param,
        "epoch": epoch,
        "device": device,
        "data_dir": str(Path(data_dir).resolve()),
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
