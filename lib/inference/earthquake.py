from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass(frozen=True)
class EarthquakeMetrics:
    intensity: float
    pga: float
    pgv: float
    max_x: float
    max_y: float
    scale: float
    fps: float

    def to_dict(self):
        return asdict(self)


def compute_metrics_from_file(file_path: Path, size: float, fps: float) -> EarthquakeMetrics:
    bbox_array = np.loadtxt(str(file_path), delimiter="\t", ndmin=2)
    return compute_metrics_from_bboxes(bbox_array, size=size, fps=fps)


def compute_metrics_from_bboxes(bboxes: Sequence[Sequence[float]], size: float, fps: float) -> EarthquakeMetrics:
    bbox_array = np.asarray(bboxes, dtype=np.float64)
    if bbox_array.ndim != 2 or bbox_array.shape[1] < 4:
        raise ValueError("Bounding boxes must be shaped like [N, 4].")
    if bbox_array.shape[0] < 6:
        raise ValueError("At least 6 frames are required to compute earthquake metrics reliably.")
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    if size <= 0:
        raise ValueError("Scale must be positive.")

    dt = 1.0 / float(fps)
    lowcut = 0.05
    highcut = fps / 2.0 - fps / 10.0

    x = bbox_array[:, 0] / 1000.0
    y = bbox_array[:, 1] / 1000.0

    relative_x = x - x[0]
    relative_y = y - y[0]
    max_x = float(np.max(np.abs(relative_x)))
    max_y = float(np.max(np.abs(relative_y)))

    velocity_x = np.empty_like(x)
    velocity_y = np.empty_like(y)
    velocity_x[:] = np.nan
    velocity_y[:] = np.nan
    velocity_x[1:] = (x[1:] - x[:-1]) * size / dt
    velocity_y[1:] = (y[1:] - y[:-1]) * size / dt

    acceleration_x = np.empty_like(x)
    acceleration_y = np.empty_like(y)
    acceleration_x[:] = np.nan
    acceleration_y[:] = np.nan
    acceleration_x[2:] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) * size / (dt ** 2)
    acceleration_y[2:] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) * size / (dt ** 2)

    filtered_velocity_x = _butter_bandpass_filter(velocity_x[2:], lowcut, highcut, fps)
    filtered_velocity_y = _butter_bandpass_filter(velocity_y[2:], lowcut, highcut, fps)
    filtered_acceleration_x = _butter_bandpass_filter(acceleration_x[2:], lowcut, highcut, fps)
    filtered_acceleration_y = _butter_bandpass_filter(acceleration_y[2:], lowcut, highcut, fps)

    max_velocity_x = float(np.max(np.abs(filtered_velocity_x)))
    max_velocity_y = float(np.max(np.abs(filtered_velocity_y)))
    max_acceleration_x = float(np.max(np.abs(filtered_acceleration_x)) / 10.0)
    max_acceleration_y = float(np.max(np.abs(filtered_acceleration_y)) / 10.0)

    pgv = float(np.sqrt(max_velocity_x ** 2 + max_velocity_y ** 2))
    pga = float(np.sqrt(max_acceleration_x ** 2 + max_acceleration_y ** 2))

    ia = 3.17 * np.log10(pga) + 6.59
    iv = 3.00 * np.log10(pgv) + 9.77
    intensity = iv if ia >= 6.0 and iv >= 6.0 else (ia + iv) / 2.0
    intensity = float(np.clip(intensity, 1.0, 12.0))

    return EarthquakeMetrics(
        intensity=round(intensity, 1),
        pga=pga,
        pgv=pgv,
        max_x=max_x,
        max_y=max_y,
        scale=float(size),
        fps=float(fps),
    )


def _butter_bandpass_filter(data, lowcut: float, highcut: float, fs: float, order: int = 2):
    data_array = np.asarray(data, dtype=np.float64)
    if data_array.size < max(6, order * 3):
        raise ValueError("Not enough valid samples to apply the bandpass filter.")
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band", analog=False)
    return filtfilt(b, a, data_array)
