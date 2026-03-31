import argparse
import json
from pathlib import Path
import sys
import warnings

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm(\..*)?$")

from lib.inference import run_video_inference_pipeline


DEFAULT_TRACKER_PARAM = "B9_cae_center_all_ep300"
DEFAULT_EPOCH = 300


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the complete LCCTV pipeline from a single video file: extract frames, auto-select target, and infer."
    )
    parser.add_argument("video_path", type=Path, help="Path to the source video file.")
    parser.add_argument(
        "--tracker-param",
        type=str,
        default=DEFAULT_TRACKER_PARAM,
        help=f"Experiment yaml name. Default: {DEFAULT_TRACKER_PARAM}",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=DEFAULT_EPOCH,
        help=f"Checkpoint epoch. Default: {DEFAULT_EPOCH}",
    )
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR, help="LCCTV project root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root directory. Defaults to output/video_inference/<video_name>/<tracker_param>_epXXXX.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override.")
    parser.add_argument("--device", type=str, default=None, help="torch device, for example cpu or cuda:0.")
    parser.add_argument("--sequence-name", type=str, default=None, help="Optional sequence name inside the workspace.")
    parser.add_argument("--reference-data-dir", type=Path, default=None, help="Optional historical DATA root used to guide auto target selection.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for extraction and inference.")
    parser.add_argument("--every-n", type=int, default=1, help="Keep one frame every N source frames.")
    parser.add_argument("--fps", type=float, default=24.0, help="FPS used by the earthquake metrics module.")
    parser.add_argument("--default-size", type=float, default=0.4, help="Metric scale factor passed into the earthquake module.")
    parser.add_argument("--image-format", type=str, default="jpg", choices=["jpg", "jpeg", "png"], help="Image format used for extracted frames.")
    parser.add_argument(
        "--reuse-workspace",
        action="store_true",
        help="Reuse existing extracted frames instead of cleaning the workspace first.",
    )
    parser.add_argument("--skip-metrics", action="store_true", help="Skip earthquake metric calculation.")
    parser.add_argument(
        "--allow-size-change",
        action="store_true",
        help="Disable the legacy behavior that keeps bbox width and height fixed to the init box.",
    )
    return parser.parse_args()


def build_default_output_root(project_dir: Path, video_path: Path, tracker_param: str, epoch: int) -> Path:
    video_name = sanitize_name(video_path.stem)
    return project_dir / "output" / "video_inference" / video_name / f"{tracker_param}_ep{epoch:04d}"


def sanitize_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value.strip())
    safe = safe.strip("._")
    return safe or "video"


def main():
    args = parse_args()
    output_root = args.output_root or build_default_output_root(
        args.project_dir.resolve(),
        args.video_path.resolve(),
        args.tracker_param,
        args.epoch,
    )

    result = run_video_inference_pipeline(
        video_path=args.video_path,
        project_dir=args.project_dir,
        tracker_param=args.tracker_param,
        epoch=args.epoch,
        output_root=output_root,
        device=args.device,
        checkpoint_path=args.checkpoint,
        max_frames=args.max_frames,
        every_n=args.every_n,
        fps=args.fps,
        default_size=args.default_size,
        image_format=args.image_format,
        sequence_name=args.sequence_name,
        reference_data_dir=args.reference_data_dir,
        clean_workspace=not args.reuse_workspace,
        skip_metrics=args.skip_metrics,
        lock_bbox_size=not args.allow_size_change,
    )

    print(result.report_text)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
