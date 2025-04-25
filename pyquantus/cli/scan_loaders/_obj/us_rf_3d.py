from pathlib import Path

import numpy as np

from .us_rf import UltrasoundRfImage

class UltrasoundRfImage3d(UltrasoundRfImage):
    """
    Class for ultrasound RF image data.
    """

    def __init__(self, scan_path: str, phantom_path: str):
        super().__init__(scan_path, phantom_path)
        self.coronal_res: float # mm/pix
        self.depth: float # depth in mm
        
        
class ScUltrasoundRfImage3d(UltrasoundRfImage3d):
    """
    Class for ultrasound RF image data with scan conversion.
    """

    def __init__(self, scan_path: str, phantom_path: str):
        super().__init__(scan_path, phantom_path)
        self.sc_bmode: np.ndarray
        self.coord_map_3d: np.ndarray # maps (z,y,x) in SC coords to (x,y) preSC coord
        self.sc_axial_res: float # mm/pix
        self.sc_lateral_res: float # mm/pix
        self.sc_coronal_res: float # mm/pix
        self.apex_dist: float # distance of virtual apex to probe surface in mm
        self.azim_steer_angle_start: float # azimuth steering angle (start) in degree
        self.azim_steer_angle_end: float # azimuth steering angle (end) in degree
        self.elev_steer_angle_start: float # elevation steering angle (start) in degree
        self.elev_steer_angle_end: float # elevation steering angle (end) in degree
        self.vol_depth: float # Coronal
        self.vol_width: float # Lateral
        self.vol_height: float # Axial
        self.pix_per_mm: float # pixels/mm
