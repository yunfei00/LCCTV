import cv2 as cv
import numpy as np
from PIL import Image

try:
    import jpeg4py
except Exception:
    jpeg4py = None

try:
    from turbojpeg import TurboJPEG
except Exception:
    TurboJPEG = None


davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [191, 0, 0],
    [64, 128, 0],
    [191, 128, 0],
    [64, 0, 128],
    [191, 0, 128],
    [64, 128, 128],
    [191, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 191, 0],
    [128, 191, 0],
    [0, 64, 128],
    [128, 64, 128],
]

jpeg = TurboJPEG() if TurboJPEG is not None else None


def default_image_loader(path):
    """Read an image from the given path."""
    if default_image_loader.use_jpeg4py is None:
        image = jpeg4py_loader(path)
        if image is None:
            default_image_loader.use_jpeg4py = False
            print("Using opencv_loader instead.")
        else:
            default_image_loader.use_jpeg4py = True
            return image
    if default_image_loader.use_jpeg4py:
        return jpeg4py_loader(path)
    return opencv_loader(path)


default_image_loader.use_jpeg4py = None


def jpeg4py_loader(path):
    """Image reading using jpeg4py."""
    if jpeg4py is None:
        return None
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as exc:
        print('ERROR: Could not read image "{}"'.format(path))
        print(exc)
        return None


def turbojpeg_loader(img_path):
    """Image reading using TurboJPEG."""
    if jpeg is None:
        return opencv_loader(img_path)
    try:
        with open(img_path, "rb") as file:
            return jpeg.decode(file.read())
    except Exception as exc:
        print('ERROR: Could not read image "{}"'.format(img_path))
        print(exc)
        return opencv_loader(img_path)


def opencv_loader(path):
    """Read image using opencv and return rgb format."""
    try:
        image = cv.imread(path, cv.IMREAD_COLOR)
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)
    except Exception as exc:
        print('ERROR: Could not read image "{}"'.format(path))
        print(exc)
        return None


def jpeg4py_loader_w_failsafe(path):
    """Image reading using jpeg4py with opencv fallback."""
    image = jpeg4py_loader(path)
    if image is not None:
        return image
    return opencv_loader(path)


def opencv_seg_loader(path):
    """Read segmentation annotation using opencv."""
    try:
        return cv.imread(path)
    except Exception as exc:
        print('ERROR: Could not read image "{}"'.format(path))
        print(exc)
        return None


def imread_indexed(filename):
    """Load indexed image with given filename."""
    image = Image.open(filename)
    annotation = np.atleast_3d(image)[..., 0]
    return annotation


def imwrite_indexed(filename, array, color_palette=None):
    """Save indexed image as png."""
    if color_palette is None:
        color_palette = davis_palette

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    image = Image.fromarray(array)
    image.putpalette(color_palette.ravel())
    image.save(filename, format="PNG")
