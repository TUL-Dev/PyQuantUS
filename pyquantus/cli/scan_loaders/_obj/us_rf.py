import pickle
from pathlib import Path

import numpy as np

from .base import ImageData

class UltrasoundRfImage(ImageData):
    """
    Class for ultrasound RF image data.
    """

    def __init__(self):
        super().__init__()
        self.rf_data: np.ndarray
        self.phantom_rf_data: np.ndarray
        self.bmode: np.ndarray
        self.axial_res: float # mm/pix
        self.lateral_res: float # mm/pix
            
class ScUltrasoundRfImage(UltrasoundRfImage):
    """
    Class for ultrasound RF image data with scan conversion.
    """

    def __init__(self):
        super().__init__()
        self.sc_bmode: np.ndarray
        self.xmap: np.ndarray # sc (y,x) --> preSC x
        self.ymap: np.ndarray # sc (y,x) --> preSC y
        self.width: float # deg
        self.tilt: float
        self.start_depth: float # mm
        self.end_depth: float # mm
        self.sc_axial_res: float # mm
        self.sc_lateral_res: float
