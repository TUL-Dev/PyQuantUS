import pickle
from pathlib import Path

import numpy as np

from .base import SegData

class BmodeSeg(SegData):
    """
    Class for ultrasound RF image data.
    """

    def __init__(self):
        super().__init__()
        self.spline_x: np.ndarray
        self.spline_y: np.ndarray
