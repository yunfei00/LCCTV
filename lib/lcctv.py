"""Compatibility shim for older import paths.

Some environments still import `lib.lcctv` through an older `lib.__init__`
layout. Forward those imports to the standalone inference implementation.
"""

from lib.inference.lcctv import InferenceResult, LcctvInferencer

__all__ = ["InferenceResult", "LcctvInferencer"]
