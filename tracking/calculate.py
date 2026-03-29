from lib.inference.earthquake import compute_metrics_from_file


def calculate(file_path, size, fps):
    metrics = compute_metrics_from_file(file_path, size=size, fps=fps)
    return metrics.intensity, metrics.pga, metrics.pgv, metrics.max_x, metrics.max_y
