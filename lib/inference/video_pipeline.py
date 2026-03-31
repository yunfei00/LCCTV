from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .data import discover_sequences
from .lcctv import LcctvInferencer
from .reporting import build_sequence_report_text, save_run_summary, save_sequence_outputs


TRACKER_NAME = "lcctv"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class FrameExtractionReport:
    video_path: str
    output_dir: str
    image_format: str
    saved_frames: int
    source_frame_count: int
    fps: float
    width: int
    height: int
    every_n: int
    max_frames: Optional[int]
    start_index: int
    zero_pad: int
    manifest_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelectionCandidate:
    bbox: List[int]
    score: float
    metrics: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AutoSelectionResult:
    image_dir: str
    reference_frame: str
    init_frame: str
    reference_bbox: List[int]
    init_bbox: List[int]
    centroid: List[int]
    score: float
    confidence: float
    candidate_count: int
    debug_image: str
    patch_image: str
    report_path: str
    annotation_path: Optional[str]
    top_candidates: List[dict]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoInferencePipelineResult:
    video_path: str
    sequence_name: str
    workspace_root: str
    sequence_dir: str
    image_dir: str
    results_root: str
    sequence_result_dir: str
    tracker_param: str
    epoch: int
    device: str
    extraction: dict
    selection: dict
    sequence_result: dict
    report_text: str
    inference_result_path: str
    run_summary_path: str
    pipeline_summary_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    image_format: str = "jpg",
    every_n: int = 1,
    max_frames: Optional[int] = None,
    overwrite: bool = False,
    clean_output: bool = False,
    start_index: int = 1,
    zero_pad: int = 8,
    write_manifest: bool = True,
) -> FrameExtractionReport:
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    image_format = image_format.lower().lstrip(".")
    if image_format not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported image format: {image_format}")
    if every_n < 1:
        raise ValueError("every_n must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if zero_pad < 1:
        raise ValueError("zero_pad must be >= 1")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_images = list_image_files(output_dir)
    if existing_images:
        if clean_output:
            for path in existing_images:
                path.unlink()
        elif not overwrite:
            raise FileExistsError(
                f"Output directory already contains {len(existing_images)} image files: {output_dir}. "
                "Use overwrite or clean_output."
            )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        source_index = 0
        saved_frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if source_index % every_n == 0:
                frame_id = start_index + saved_frames
                frame_path = output_dir / f"{frame_id:0{zero_pad}d}.{image_format}"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write frame: {frame_path}")
                saved_frames += 1
                if saved_frames % 100 == 0:
                    print(f"[extract] saved {saved_frames} frames to {output_dir}")
                if max_frames is not None and saved_frames >= max_frames:
                    break

            source_index += 1
    finally:
        cap.release()

    manifest_path = output_dir / "frame_manifest.json"
    report = FrameExtractionReport(
        video_path=str(video_path),
        output_dir=str(output_dir),
        image_format=image_format,
        saved_frames=saved_frames,
        source_frame_count=source_frame_count,
        fps=fps,
        width=width,
        height=height,
        every_n=every_n,
        max_frames=max_frames,
        start_index=start_index,
        zero_pad=zero_pad,
        manifest_path=str(manifest_path),
    )
    if write_manifest:
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(report.to_dict(), file, ensure_ascii=False, indent=2)

    print(
        f"[extract] completed video={video_path.name} saved={saved_frames} fps={fps:.3f} "
        f"size={width}x{height} output={output_dir}"
    )
    return report


class AutoTargetSelector:
    def __init__(
        self,
        *,
        analysis_frames: int = 6,
        min_area_ratio: float = 0.002,
        max_area_ratio: float = 0.12,
        preferred_area_ratio: float = 0.02,
        min_side: int = 18,
        max_aspect_ratio: float = 3.5,
        temporal_window_ratio: float = 1.8,
        candidate_limit: int = 48,
    ):
        self.analysis_frames = analysis_frames
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.preferred_area_ratio = preferred_area_ratio
        self.min_side = min_side
        self.max_aspect_ratio = max_aspect_ratio
        self.temporal_window_ratio = temporal_window_ratio
        self.candidate_limit = candidate_limit

        self.last_result: Optional[AutoSelectionResult] = None
        self.reference_prototypes: List[np.ndarray] = []

    def select_from_sequence_dir(
        self,
        sequence_dir: str | Path,
        *,
        annotation_path: str | Path | None = None,
        debug_dir: str | Path | None = None,
        write_groundtruth: bool = True,
        prototype_root: str | Path | None = None,
        exclude_sequence_names: Optional[Sequence[str]] = None,
    ) -> AutoSelectionResult:
        sequence_dir = Path(sequence_dir).expanduser().resolve()
        image_dir = find_image_dir(sequence_dir)
        annotation_path = Path(annotation_path).expanduser().resolve() if annotation_path else (sequence_dir / "groundtruth.txt")
        debug_dir = Path(debug_dir).expanduser().resolve() if debug_dir else sequence_dir
        debug_dir.mkdir(parents=True, exist_ok=True)

        if prototype_root is not None:
            self.reference_prototypes = load_reference_prototypes(
                prototype_root,
                exclude_sequence_names=exclude_sequence_names or [],
            )
        else:
            self.reference_prototypes = []

        frame_paths = list_image_files(image_dir)
        if not frame_paths:
            raise FileNotFoundError(f"No image files were found under: {image_dir}")

        analysis_paths = frame_paths[: max(1, min(self.analysis_frames, len(frame_paths)))]
        color_frames = [load_color_frame(path) for path in analysis_paths]
        gray_frames = [prepare_gray(frame) for frame in color_frames]

        reference_index = self._choose_reference_frame(gray_frames)
        reference_gray = gray_frames[reference_index]
        first_frame = color_frames[0]
        first_gray = gray_frames[0]

        candidates = self._generate_candidates(reference_gray)
        if not candidates:
            raise RuntimeError(f"Failed to generate target candidates from: {analysis_paths[reference_index]}")

        scored_candidates = [
            self._score_candidate(candidate_bbox, reference_gray, gray_frames, reference_index)
            for candidate_bbox in candidates
        ]
        scored_candidates = [candidate for candidate in scored_candidates if candidate is not None]
        scored_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        scored_candidates = deduplicate_candidates(scored_candidates)
        if not scored_candidates:
            raise RuntimeError("No valid candidates remained after scoring.")

        best = scored_candidates[0]
        init_bbox = list(best.bbox)
        if reference_index != 0:
            mapped_bbox, match_score = self._localize_template(
                first_gray,
                reference_gray[best.bbox[1] : best.bbox[1] + best.bbox[3], best.bbox[0] : best.bbox[0] + best.bbox[2]],
                best.bbox,
                search_scale=2.2,
            )
            init_bbox = mapped_bbox
            best.metrics["first_frame_match"] = float(match_score)
        else:
            best.metrics["first_frame_match"] = 1.0

        centroid = [int(round(init_bbox[0] + init_bbox[2] / 2.0)), int(round(init_bbox[1] + init_bbox[3] / 2.0))]
        confidence = float(np.clip(best.score, 0.0, 1.0))
        debug_image_path = debug_dir / "auto_target_debug.jpg"
        patch_image_path = debug_dir / "auto_target_patch.jpg"
        report_path = debug_dir / "auto_target_result.json"

        save_patch(first_frame, init_bbox, patch_image_path)
        render_debug_image(
            image=first_frame,
            selected_bbox=init_bbox,
            candidates=scored_candidates[:5],
            output_path=debug_image_path,
            heading=f"auto target | ref={analysis_paths[reference_index].name} | conf={confidence:.3f}",
        )

        if write_groundtruth:
            write_groundtruth_file(annotation_path, init_bbox)

        result = AutoSelectionResult(
            image_dir=str(image_dir),
            reference_frame=str(analysis_paths[reference_index]),
            init_frame=str(frame_paths[0]),
            reference_bbox=list(best.bbox),
            init_bbox=init_bbox,
            centroid=centroid,
            score=float(best.score),
            confidence=confidence,
            candidate_count=len(scored_candidates),
            debug_image=str(debug_image_path),
            patch_image=str(patch_image_path),
            report_path=str(report_path),
            annotation_path=str(annotation_path),
            top_candidates=[candidate.to_dict() for candidate in scored_candidates[:5]],
        )
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)

        self.last_result = result
        print(
            f"[select] reference={Path(result.reference_frame).name} init_bbox={tuple(result.init_bbox)} "
            f"confidence={result.confidence:.3f} candidates={result.candidate_count}"
        )
        print(f"[select] wrote groundtruth to {annotation_path}")
        return result

    def _choose_reference_frame(self, gray_frames: Sequence[np.ndarray]) -> int:
        sharpness_scores = [float(cv2.Laplacian(frame, cv2.CV_32F).var()) for frame in gray_frames]
        return int(np.argmax(sharpness_scores))

    def _generate_candidates(self, gray_frame: np.ndarray) -> List[List[int]]:
        height, width = gray_frame.shape[:2]
        image_area = float(height * width)
        min_area = image_area * self.min_area_ratio
        max_area = image_area * self.max_area_ratio

        equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_frame)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)
        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=2,
        )

        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        proposal_boxes: List[List[int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_side or h < self.min_side:
                continue
            area = float(w * h)
            if area < min_area or area > max_area:
                continue
            aspect_ratio = w / max(h, 1)
            if aspect_ratio > self.max_aspect_ratio or (1.0 / max(aspect_ratio, 1e-6)) > self.max_aspect_ratio:
                continue
            if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
                continue
            proposal_boxes.append([int(x), int(y), int(w), int(h)])

        proposal_boxes.extend(self._generate_corner_window_candidates(equalized))
        proposal_boxes.extend(self._generate_grid_window_candidates(equalized))
        if not proposal_boxes:
            return []

        ranked = sorted(
            unique_bboxes(proposal_boxes),
            key=lambda bbox: self._pre_score_bbox(bbox, gray_frame, edges),
            reverse=True,
        )
        return ranked[: self.candidate_limit]

    def _generate_corner_window_candidates(self, gray_frame: np.ndarray) -> List[List[int]]:
        height, width = gray_frame.shape[:2]
        image_area = float(height * width)
        corners = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=28,
            qualityLevel=0.01,
            minDistance=16,
            blockSize=5,
        )
        if corners is None:
            return []

        area_ratios = sorted(
            {
                round(max(self.min_area_ratio * 2.5, 0.006), 4),
                round(max(self.preferred_area_ratio * 0.65, 0.008), 4),
                round(self.preferred_area_ratio, 4),
                round(min(self.max_area_ratio * 0.4, self.preferred_area_ratio * 1.5), 4),
            }
        )
        aspect_ratios = (0.6, 0.9, 1.3, 1.8)
        offsets = ((0.0, 0.0), (-0.18, 0.0), (0.18, 0.0), (0.0, -0.18), (0.0, 0.18))

        proposals: List[List[int]] = []
        for corner in corners.reshape(-1, 2):
            cx, cy = float(corner[0]), float(corner[1])
            for area_ratio in area_ratios:
                window_area = image_area * area_ratio
                for aspect_ratio in aspect_ratios:
                    window_w = int(round(math.sqrt(window_area * aspect_ratio)))
                    window_h = int(round(window_area / max(window_w, 1)))
                    if window_w < self.min_side or window_h < self.min_side:
                        continue
                    if window_w >= width or window_h >= height:
                        continue
                    if window_w * window_h < image_area * self.min_area_ratio:
                        continue
                    if window_w * window_h > image_area * self.max_area_ratio:
                        continue
                    for offset_x, offset_y in offsets:
                        x = int(round(cx - window_w / 2.0 + offset_x * window_w))
                        y = int(round(cy - window_h / 2.0 + offset_y * window_h))
                        x = int(np.clip(x, 0, width - window_w))
                        y = int(np.clip(y, 0, height - window_h))
                        proposals.append([x, y, window_w, window_h])
        return proposals

    def _generate_grid_window_candidates(self, gray_frame: np.ndarray) -> List[List[int]]:
        height, width = gray_frame.shape[:2]
        image_area = float(height * width)
        area_ratios = (0.0075, 0.012, self.preferred_area_ratio)
        aspect_ratios = (0.65, 1.0, 1.6)

        proposals: List[List[int]] = []
        for area_ratio in area_ratios:
            window_area = image_area * area_ratio
            for aspect_ratio in aspect_ratios:
                window_w = int(round(math.sqrt(window_area * aspect_ratio)))
                window_h = int(round(window_area / max(window_w, 1)))
                if window_w < self.min_side or window_h < self.min_side:
                    continue
                if window_w >= width or window_h >= height:
                    continue

                step_x = max(12, window_w // 2)
                step_y = max(12, window_h // 2)
                max_x = width - window_w
                max_y = height - window_h
                for y in range(0, max_y + 1, step_y):
                    for x in range(0, max_x + 1, step_x):
                        proposals.append([x, y, window_w, window_h])
                if max_x > 0:
                    for y in range(0, max_y + 1, step_y):
                        proposals.append([max_x, y, window_w, window_h])
                if max_y > 0:
                    for x in range(0, max_x + 1, step_x):
                        proposals.append([x, max_y, window_w, window_h])
        return proposals

    def _pre_score_bbox(self, bbox: Sequence[int], gray_frame: np.ndarray, edges: np.ndarray) -> float:
        x, y, w, h = bbox
        roi = gray_frame[y : y + h, x : x + w]
        if roi.size == 0:
            return 0.0
        band = max(2, int(round(min(w, h) * 0.08)))
        edge_strength = border_edge_density(edges[y : y + h, x : x + w], band=band)
        texture = normalize_value(float(cv2.Laplacian(roi, cv2.CV_32F).var()), scale=500.0)
        area_ratio = (w * h) / float(gray_frame.shape[0] * gray_frame.shape[1])
        area_preference = area_preference_score(area_ratio, self.preferred_area_ratio)
        return 0.42 * edge_strength + 0.36 * texture + 0.22 * area_preference

    def _score_candidate(
        self,
        bbox: Sequence[int],
        reference_gray: np.ndarray,
        gray_frames: Sequence[np.ndarray],
        reference_index: int,
    ) -> Optional[SelectionCandidate]:
        x, y, w, h = [int(value) for value in bbox]
        roi = reference_gray[y : y + h, x : x + w]
        if roi.size == 0 or roi.shape[0] < 6 or roi.shape[1] < 6:
            return None

        edges = cv2.Canny(reference_gray, 60, 180)
        roi_edges = edges[y : y + h, x : x + w]
        band = max(2, int(round(min(w, h) * 0.08)))

        border_density = border_edge_density(roi_edges, band=band)
        texture = normalize_value(float(cv2.Laplacian(roi, cv2.CV_32F).var()), scale=450.0)
        corner_score = corner_density_score(roi)
        contrast = local_contrast_score(reference_gray, bbox)
        area_ratio = (w * h) / float(reference_gray.shape[0] * reference_gray.shape[1])
        area_preference = area_preference_score(area_ratio, self.preferred_area_ratio)
        center_bias = center_bias_score(reference_gray.shape[1], reference_gray.shape[0], bbox)
        temporal_score, temporal_corr, temporal_disp = self._temporal_consistency(gray_frames, reference_index, bbox)
        prototype_score = patch_similarity_score(roi, self.reference_prototypes)

        final_score = (
            0.30 * temporal_score
            + 0.17 * border_density
            + 0.16 * texture
            + 0.12 * corner_score
            + 0.09 * area_preference
            + 0.08 * contrast
            + 0.04 * center_bias
            + 0.04 * prototype_score
        )
        metrics = {
            "border_density": float(border_density),
            "texture": float(texture),
            "corner_score": float(corner_score),
            "contrast": float(contrast),
            "area_preference": float(area_preference),
            "center_bias": float(center_bias),
            "prototype_score": float(prototype_score),
            "temporal_score": float(temporal_score),
            "temporal_corr": float(temporal_corr),
            "temporal_disp": float(temporal_disp),
            "area_ratio": float(area_ratio),
        }
        return SelectionCandidate(bbox=[x, y, w, h], score=float(final_score), metrics=metrics)

    def _temporal_consistency(
        self,
        gray_frames: Sequence[np.ndarray],
        reference_index: int,
        bbox: Sequence[int],
    ) -> Tuple[float, float, float]:
        x, y, w, h = [int(value) for value in bbox]
        template = gray_frames[reference_index][y : y + h, x : x + w]
        if template.size == 0:
            return 0.0, 0.0, 0.0

        correlations: List[float] = []
        displacements: List[float] = []
        bbox_diag = math.sqrt(float(w * w + h * h))
        for frame_index, frame in enumerate(gray_frames):
            if frame_index == reference_index:
                continue
            _, match_score, center_distance = self._localize_template(frame, template, bbox, return_distance=True)
            correlations.append(match_score)
            displacements.append(center_distance / max(bbox_diag, 1.0))

        if not correlations:
            return 1.0, 1.0, 1.0

        corr_score = float(np.clip(np.mean(correlations), 0.0, 1.0))
        disp_score = float(np.clip(1.0 - np.mean(displacements), 0.0, 1.0))
        temporal_score = 0.7 * corr_score + 0.3 * disp_score
        return float(temporal_score), corr_score, disp_score

    def _localize_template(
        self,
        gray_frame: np.ndarray,
        template: np.ndarray,
        approximate_bbox: Sequence[int],
        *,
        search_scale: Optional[float] = None,
        return_distance: bool = False,
    ):
        x, y, w, h = [int(value) for value in approximate_bbox]
        search_scale = search_scale if search_scale is not None else self.temporal_window_ratio
        margin_x = max(12, int(round(w * search_scale)))
        margin_y = max(12, int(round(h * search_scale)))

        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(gray_frame.shape[1], x + w + margin_x)
        y1 = min(gray_frame.shape[0], y + h + margin_y)

        search = gray_frame[y0:y1, x0:x1]
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            localized_bbox = [x, y, w, h]
            if return_distance:
                return localized_bbox, 0.0, 1.0
            return localized_bbox, 0.0

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        localized_bbox = [int(x0 + max_loc[0]), int(y0 + max_loc[1]), int(w), int(h)]

        prev_center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
        new_center = np.array([localized_bbox[0] + w / 2.0, localized_bbox[1] + h / 2.0], dtype=np.float32)
        center_distance = float(np.linalg.norm(new_center - prev_center) / max(max(gray_frame.shape[:2]), 1))
        match_score = float(np.clip((max_val + 1.0) / 2.0, 0.0, 1.0))

        if return_distance:
            return localized_bbox, match_score, center_distance
        return localized_bbox, match_score


def run_video_inference_pipeline(
    *,
    video_path: str | Path,
    project_dir: str | Path,
    tracker_param: str,
    epoch: int,
    output_root: str | Path,
    device: Optional[str] = None,
    checkpoint_path: str | Path | None = None,
    max_frames: Optional[int] = None,
    every_n: int = 1,
    fps: float = 24.0,
    default_size: float = 0.4,
    image_format: str = "jpg",
    sequence_name: Optional[str] = None,
    reference_data_dir: str | Path | None = None,
    clean_workspace: bool = True,
    skip_metrics: bool = False,
    lock_bbox_size: bool = True,
) -> VideoInferencePipelineResult:
    video_path = Path(video_path).expanduser().resolve()
    project_dir = Path(project_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    sequence_name = sanitize_sequence_name(sequence_name or video_path.stem)
    workspace_root = output_root / "workspace"
    sequence_dir = workspace_root / sequence_name
    image_dir = sequence_dir / "img"
    results_root = output_root / "results"
    sequence_result_dir = results_root / sequence_name

    sequence_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    extraction = extract_video_frames(
        video_path=video_path,
        output_dir=image_dir,
        image_format=image_format,
        every_n=every_n,
        max_frames=max_frames,
        overwrite=not clean_workspace,
        clean_output=clean_workspace,
        start_index=1,
        zero_pad=8,
    )

    selector = AutoTargetSelector()
    prototype_root, exclude_sequence_names = resolve_reference_context(
        video_path=video_path,
        project_dir=project_dir,
        reference_data_dir=reference_data_dir,
        workspace_sequence_name=sequence_name,
    )
    selection = selector.select_from_sequence_dir(
        sequence_dir=sequence_dir,
        annotation_path=sequence_dir / "groundtruth.txt",
        debug_dir=sequence_dir,
        write_groundtruth=True,
        prototype_root=prototype_root,
        exclude_sequence_names=exclude_sequence_names,
    )

    sequence = discover_sequences(workspace_root, include_names=[sequence_name])[0]

    inferencer = LcctvInferencer(
        project_dir=project_dir,
        tracker_param=tracker_param,
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        device=device,
        lock_bbox_size=lock_bbox_size,
    )
    result = inferencer.run_sequence(
        sequence,
        metric_provider=None if skip_metrics else (lambda _: {"fps": fps, "size": default_size}),
    )
    report_text = build_sequence_report_text(TRACKER_NAME, tracker_param, epoch, result)
    save_sequence_outputs(sequence_result_dir, result, report_text)
    save_run_summary(
        output_dir=results_root,
        tracker_param=tracker_param,
        epoch=epoch,
        device=str(inferencer.device),
        data_dir=workspace_root,
        results=[result],
        default_size=default_size,
        fps=fps,
        sequence_size_overrides={sequence_name: default_size},
        lock_bbox_size=lock_bbox_size,
        run_report_text=report_text,
    )

    pipeline_summary_path = output_root / "video_pipeline_summary.json"
    pipeline_result = VideoInferencePipelineResult(
        video_path=str(video_path),
        sequence_name=sequence_name,
        workspace_root=str(workspace_root),
        sequence_dir=str(sequence_dir),
        image_dir=str(image_dir),
        results_root=str(results_root),
        sequence_result_dir=str(sequence_result_dir),
        tracker_param=tracker_param,
        epoch=epoch,
        device=str(inferencer.device),
        extraction=extraction.to_dict(),
        selection=selection.to_dict(),
        sequence_result=result.to_dict(),
        report_text=report_text,
        inference_result_path=str(sequence_result_dir / "summary.json"),
        run_summary_path=str(results_root / "summary.json"),
        pipeline_summary_path=str(pipeline_summary_path),
    )
    with pipeline_summary_path.open("w", encoding="utf-8") as file:
        json.dump(pipeline_result.to_dict(), file, ensure_ascii=False, indent=2)

    return pipeline_result


def resolve_reference_context(
    *,
    video_path: Path,
    project_dir: Path,
    reference_data_dir: str | Path | None,
    workspace_sequence_name: str,
) -> Tuple[Optional[Path], List[str]]:
    exclude_sequence_names = [workspace_sequence_name]

    if reference_data_dir is None:
        candidate = project_dir.parent / "DATA"
        reference_root = candidate if candidate.is_dir() else None
    else:
        candidate = Path(reference_data_dir).expanduser().resolve()
        reference_root = candidate if candidate.is_dir() else None

    if reference_root is None:
        return None, exclude_sequence_names

    if video_path.parent.parent == reference_root:
        exclude_sequence_names.append(video_path.parent.name)

    return reference_root, sorted(set(exclude_sequence_names))


def list_image_files(image_dir: str | Path) -> List[Path]:
    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        return []
    return sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: natural_key(path.name),
    )


def find_image_dir(sequence_dir: str | Path) -> Path:
    sequence_dir = Path(sequence_dir).expanduser().resolve()
    for folder_name in ("img", "imgs", "images"):
        candidate = sequence_dir / folder_name
        if candidate.is_dir():
            return candidate

    matching_dirs = [
        candidate
        for candidate in sequence_dir.iterdir()
        if candidate.is_dir() and any(path.suffix.lower() in IMAGE_SUFFIXES for path in candidate.iterdir() if path.is_file())
    ]
    if len(matching_dirs) == 1:
        return matching_dirs[0]
    raise FileNotFoundError(f"Unable to determine image directory under: {sequence_dir}")


def load_color_frame(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return frame


def prepare_gray(color_frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def normalize_value(value: float, scale: float) -> float:
    return float(np.clip(value / max(scale, 1e-6), 0.0, 1.0))


def area_preference_score(area_ratio: float, preferred_ratio: float) -> float:
    anchors = (
        max(preferred_ratio * 0.4, 1e-6),
        max(preferred_ratio * 0.7, 1e-6),
        max(preferred_ratio, 1e-6),
        max(preferred_ratio * 1.35, 1e-6),
    )
    return float(max(np.exp(-abs(math.log(max(area_ratio, 1e-6) / anchor))) for anchor in anchors))


def border_edge_density(edge_roi: np.ndarray, *, band: int) -> float:
    if edge_roi.size == 0:
        return 0.0

    h, w = edge_roi.shape[:2]
    band = max(1, min(band, h // 2 if h > 1 else 1, w // 2 if w > 1 else 1))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:band, :] = 1
    mask[-band:, :] = 1
    mask[:, :band] = 1
    mask[:, -band:] = 1
    border_pixels = float(mask.sum())
    if border_pixels <= 0:
        return 0.0
    return float((edge_roi > 0)[mask.astype(bool)].mean())


def corner_density_score(gray_roi: np.ndarray) -> float:
    if gray_roi.size == 0:
        return 0.0
    corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=32, qualityLevel=0.01, minDistance=4)
    if corners is None:
        return 0.0
    return float(np.clip(len(corners) / 20.0, 0.0, 1.0))


def local_contrast_score(gray_frame: np.ndarray, bbox: Sequence[int]) -> float:
    x, y, w, h = [int(value) for value in bbox]
    roi = gray_frame[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0

    margin = max(4, int(round(min(w, h) * 0.2)))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(gray_frame.shape[1], x + w + margin)
    y1 = min(gray_frame.shape[0], y + h + margin)
    expanded = gray_frame[y0:y1, x0:x1].astype(np.float32)
    outer_mask = np.ones(expanded.shape, dtype=bool)
    outer_mask[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = False
    if not np.any(outer_mask):
        return 0.0

    inner_mean = float(np.mean(roi))
    outer_mean = float(np.mean(expanded[outer_mask]))
    return float(np.clip(abs(inner_mean - outer_mean) / 64.0, 0.0, 1.0))


def center_bias_score(frame_width: int, frame_height: int, bbox: Sequence[int]) -> float:
    x, y, w, h = [float(value) for value in bbox]
    center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)
    bbox_center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
    diag = math.sqrt(float(frame_width * frame_width + frame_height * frame_height))
    distance = float(np.linalg.norm(bbox_center - center))
    return float(np.clip(1.0 - distance / max(diag * 0.75, 1.0), 0.0, 1.0))


def deduplicate_candidates(candidates: Sequence[SelectionCandidate], *, iou_threshold: float = 0.65) -> List[SelectionCandidate]:
    filtered: List[SelectionCandidate] = []
    for candidate in candidates:
        if any(calculate_iou(candidate.bbox, existing.bbox) >= iou_threshold for existing in filtered):
            continue
        filtered.append(candidate)
    return filtered


def unique_bboxes(bboxes: Sequence[Sequence[int]]) -> List[List[int]]:
    unique = []
    seen = set()
    for bbox in bboxes:
        key = tuple(int(value) for value in bbox)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(key))
    return unique


def calculate_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax0, ay0, aw, ah = box_a
    bx0, by0, bw, bh = box_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0

    intersection = float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))
    union = float(aw * ah + bw * bh - intersection)
    return 0.0 if union <= 0 else intersection / union


def render_debug_image(
    image: np.ndarray,
    selected_bbox: Sequence[int],
    candidates: Sequence[SelectionCandidate],
    output_path: str | Path,
    *,
    heading: str,
):
    canvas = image.copy()
    for rank, candidate in enumerate(candidates, start=1):
        x, y, w, h = [int(value) for value in candidate.bbox]
        color = (0, 180, 255) if rank > 1 else (0, 255, 0)
        thickness = 2 if rank > 1 else 3
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            canvas,
            f"#{rank} {candidate.score:.3f}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    sx, sy, sw, sh = [int(value) for value in selected_bbox]
    cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 4)
    cv2.putText(
        canvas,
        heading,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (40, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), canvas)


def save_patch(image: np.ndarray, bbox: Sequence[int], output_path: str | Path):
    x, y, w, h = [int(value) for value in bbox]
    patch = image[y : y + h, x : x + w]
    cv2.imwrite(str(output_path), patch)


def write_groundtruth_file(annotation_path: str | Path, bbox: Sequence[int]):
    annotation_path = Path(annotation_path).expanduser().resolve()
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    values = [int(round(value)) for value in bbox]
    with annotation_path.open("w", encoding="utf-8") as file:
        file.write(",".join(str(value) for value in values) + "\n")


def load_reference_prototypes(data_root: str | Path, exclude_sequence_names: Sequence[str]) -> List[np.ndarray]:
    data_root = Path(data_root).expanduser().resolve()
    exclude_names = set(exclude_sequence_names)
    prototypes: List[np.ndarray] = []
    if not data_root.is_dir():
        return prototypes

    for sequence_dir in sorted((path for path in data_root.iterdir() if path.is_dir()), key=lambda path: natural_key(path.name)):
        if sequence_dir.name in exclude_names:
            continue
        annotation_path = sequence_dir / "groundtruth.txt"
        if not annotation_path.is_file():
            continue
        try:
            image_dir = find_image_dir(sequence_dir)
            frame_paths = list_image_files(image_dir)
            if not frame_paths:
                continue
            bbox = read_groundtruth(annotation_path)
            frame = prepare_gray(load_color_frame(frame_paths[0]))
            x, y, w, h = bbox
            patch = frame[y : y + h, x : x + w]
            if patch.size == 0:
                continue
            prototypes.append(normalize_patch(patch))
        except Exception:
            continue
    return prototypes


def read_groundtruth(annotation_path: str | Path) -> List[int]:
    annotation_path = Path(annotation_path).expanduser().resolve()
    content = annotation_path.read_text(encoding="utf-8").strip()
    tokens = [token for token in re.split(r"[\s,]+", content) if token]
    if len(tokens) < 4:
        raise ValueError(f"Invalid groundtruth format: {annotation_path}")
    return [int(round(float(token))) for token in tokens[:4]]


def normalize_patch(gray_patch: np.ndarray, *, size: int = 32) -> np.ndarray:
    resized = cv2.resize(gray_patch, (size, size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32)
    normalized -= normalized.mean()
    std = float(normalized.std())
    if std > 1e-6:
        normalized /= std
    return normalized


def patch_similarity_score(gray_patch: np.ndarray, prototypes: Sequence[np.ndarray]) -> float:
    if gray_patch.size == 0 or not prototypes:
        return 0.5
    candidate = normalize_patch(gray_patch)
    scores = []
    for prototype in prototypes:
        raw = float(np.mean(candidate * prototype))
        scores.append(float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0)))
    return max(scores) if scores else 0.5


def natural_key(text: str):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def sanitize_sequence_name(value: str) -> str:
    sanitized = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE).strip("._")
    return sanitized or "video"
