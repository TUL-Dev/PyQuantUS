import pickle
from pathlib import Path

import numpy as np

from .base import SegData

class BmodeSeg3d(SegData):
    """
    Class for ultrasound RF image data.
    """

    def __init__(self):
        super().__init__()
        self.seg_mask: np.ndarray # Segmentation mask
