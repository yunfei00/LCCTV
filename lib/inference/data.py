import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class SequenceInput:
    name: str
    sequence_dir: Path
    image_dir: Path
    frame_paths: List[Path]
    init_bbox: List[float]
    annotation_path: Path


def discover_sequences(
    data_dir: Path,
    include_names: Optional[Sequence[str]] = None,
    max_frames: Optional[int] = None,
) -> List[SequenceInput]:
    data_dir = Path(data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    include_set = set(include_names) if include_names else None
    sequences: List[SequenceInput] = []
    for sequence_dir in sorted((path for path in data_dir.iterdir() if path.is_dir()), key=lambda p: _natural_key(p.name)):
        if include_set is not None and sequence_dir.name not in include_set:
            continue

        annotation_path = sequence_dir / "groundtruth.txt"
        if not annotation_path.exists():
            continue

        image_dir = _find_image_dir(sequence_dir)
        frame_paths = _list_frames(image_dir)
        if not frame_paths:
            continue

        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        sequences.append(
            SequenceInput(
                name=sequence_dir.name,
                sequence_dir=sequence_dir,
                image_dir=image_dir,
                frame_paths=frame_paths,
                init_bbox=_load_init_bbox(annotation_path),
                annotation_path=annotation_path,
            )
        )

    if include_set:
        found_names = {sequence.name for sequence in sequences}
        missing_names = sorted(include_set - found_names)
        if missing_names:
            raise ValueError(f"Requested sequences were not found: {', '.join(missing_names)}")

    if not sequences:
        raise ValueError(f"No valid sequences found under {data_dir}")

    return sequences


def _find_image_dir(sequence_dir: Path) -> Path:
    for folder_name in ("imgs", "img", "images"):
        candidate = sequence_dir / folder_name
        if candidate.is_dir():
            return candidate

    candidates = []
    for candidate in sequence_dir.iterdir():
        if candidate.is_dir() and any(path.suffix.lower() in IMAGE_SUFFIXES for path in candidate.iterdir() if path.is_file()):
            candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(f"Unable to determine image directory for sequence: {sequence_dir}")


def _list_frames(image_dir: Path) -> List[Path]:
    frames = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(frames, key=lambda path: _natural_key(path.name))


def _load_init_bbox(annotation_path: Path) -> List[float]:
    with annotation_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            tokens = [token for token in re.split(r"[\s,]+", line) if token]
            if len(tokens) < 4:
                raise ValueError(f"Invalid init bbox in {annotation_path}: {line}")
            return [float(token) for token in tokens[:4]]
    raise ValueError(f"Annotation file is empty: {annotation_path}")


def _natural_key(text: str):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]
