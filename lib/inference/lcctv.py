import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import torch

from lib.config.lcctv.config import cfg as default_cfg
from lib.config.lcctv.config import update_config_from_file
from lib.inference.data import SequenceInput
from lib.inference.earthquake import EarthquakeMetrics, compute_metrics_from_bboxes
from lib.inference.processing import sample_target, transform_image_to_crop
from lib.models import build_Lcctv
from lib.test.tracker.data_utils import Preprocessor
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import box_xywh_to_xyxy, clip_box


@dataclass
class InferenceResult:
    sequence_name: str
    frame_count: int
    init_bbox: List[float]
    bboxes: List[List[float]]
    timings: List[float]
    metrics: Optional[EarthquakeMetrics]
    metrics_error: Optional[str] = None

    def to_dict(self):
        payload = asdict(self)
        if self.metrics is not None:
            payload["metrics"] = self.metrics.to_dict()
        return payload


class LcctvInferencer:
    def __init__(
        self,
        project_dir: Path,
        tracker_param: str,
        epoch: int,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
        lock_bbox_size: bool = True,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.tracker_param = tracker_param
        self.epoch = int(epoch)
        self.device = self._resolve_device(device)
        self.lock_bbox_size = lock_bbox_size

        self.cfg = self._load_config()
        self.template_factor = self.cfg.TEST.TEMPLATE_FACTOR
        self.template_size = self.cfg.TEST.TEMPLATE_SIZE
        self.search_factor = self.cfg.TEST.SEARCH_FACTOR
        self.search_size = self.cfg.TEST.SEARCH_SIZE
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.network = self._build_network()
        self.preprocessor = Preprocessor(device=self.device)
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz], device=self.device).long(),
            centered=True,
        ).to(self.device)

        self.state: Optional[List[float]] = None
        self.z_feat = None

    def run_sequence(
        self,
        sequence: SequenceInput,
        metric_provider: Optional[Callable[[SequenceInput], Optional[Dict[str, float]]]] = None,
    ) -> InferenceResult:
        first_image = self._read_image(sequence.frame_paths[0])
        start_time = time.time()
        self.initialize(first_image, sequence.init_bbox)
        timings = [time.time() - start_time]
        bboxes = [list(sequence.init_bbox)]

        for frame_path in sequence.frame_paths[1:]:
            image = self._read_image(frame_path)
            start_time = time.time()
            bbox = self.track(image)
            timings.append(time.time() - start_time)
            bboxes.append([float(value) for value in bbox])

        metrics = None
        metrics_error = None
        if metric_provider is not None:
            metric_config = metric_provider(sequence)
            if metric_config is not None:
                try:
                    metrics = compute_metrics_from_bboxes(
                        bboxes,
                        size=metric_config["size"],
                        fps=metric_config["fps"],
                    )
                except Exception as exc:
                    metrics_error = str(exc)

        return InferenceResult(
            sequence_name=sequence.name,
            frame_count=len(sequence.frame_paths),
            init_bbox=list(sequence.init_bbox),
            bboxes=bboxes,
            timings=timings,
            metrics=metrics,
            metrics_error=metrics_error,
        )

    def initialize(self, image, init_bbox: List[float]):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image,
            init_bbox,
            self.template_factor,
            output_sz=self.template_size,
        )
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        template_bbox = self._transform_bbox_to_crop(init_bbox, resize_factor, crop_type="template").squeeze(1)
        template_bbox = box_xywh_to_xyxy(template_bbox).float()

        with torch.no_grad():
            self.z_feat = self.network.forward_z(template.tensors, template_bb=template_bbox)

        self.state = [float(value) for value in init_bbox]

    def track(self, image) -> List[float]:
        if self.state is None:
            raise RuntimeError("Tracker must be initialized before calling track().")

        height, width, _ = image.shape
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            self.state,
            self.search_factor,
            output_sz=self.search_size,
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            out_dict = self.network(template_feats=self.z_feat, search=search.tensors)

        pred_score_map = out_dict["score_map"]
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(
            response,
            int(self.feat_sz),
            out_dict["size_map"],
            out_dict["offset_map"],
        )
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()
        next_state = clip_box(self._map_box_back(pred_box, resize_factor), height, width, margin=10)

        if self.lock_bbox_size:
            next_state[2] = self.state[2]
            next_state[3] = self.state[3]

        self.state = [float(value) for value in next_state]
        return self.state

    def _build_network(self):
        network = build_Lcctv(self.cfg, training=False)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        network.load_state_dict(checkpoint["net"], strict=False)
        network = network.to(self.device)
        network.eval()
        return network

    def _load_config(self):
        cfg_copy = copy.deepcopy(default_cfg)
        yaml_path = self.project_dir / "experiments" / "lcctv" / f"{self.tracker_param}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Experiment yaml was not found: {yaml_path}")
        update_config_from_file(str(yaml_path), base_cfg=cfg_copy)
        return cfg_copy

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[Path]) -> Path:
        if checkpoint_path is not None:
            resolved_path = Path(checkpoint_path).resolve()
        else:
            resolved_path = (
                self.project_dir
                / "output"
                / "checkpoints"
                / "train"
                / "lcctv"
                / self.tracker_param
                / f"lcctv_ep{self.epoch:04d}.pth.tar"
            ).resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"Checkpoint was not found: {resolved_path}")
        return resolved_path

    def _transform_bbox_to_crop(self, box_in, resize_factor, crop_type="template", box_extract=None):
        if crop_type == "template":
            crop_size = self.template_size
        elif crop_type == "search":
            crop_size = self.search_size
        else:
            raise ValueError(f"Unsupported crop type: {crop_type}")

        crop_sz = torch.tensor([crop_size, crop_size], dtype=torch.float32)
        box_tensor = torch.tensor(box_in, dtype=torch.float32)
        box_extract_tensor = box_tensor if box_extract is None else torch.tensor(box_extract, dtype=torch.float32)
        crop_bbox = transform_image_to_crop(box_tensor, box_extract_tensor, resize_factor, crop_sz, normalize=True)
        return crop_bbox.view(1, 1, 4).to(self.device)

    def _map_box_back(self, pred_box: List[float], resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, width, height = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * width, cy_real - 0.5 * height, width, height]

    @staticmethod
    def _read_image(path: Path):
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _resolve_device(device: Optional[str]):
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
