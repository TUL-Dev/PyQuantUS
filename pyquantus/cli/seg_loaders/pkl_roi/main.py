import pickle
from pathlib import Path

import numpy as np

from .._obj.us_rf import BmodeSeg

class EntryClass(BmodeSeg):
    """
    Class for loading ROI data from a pickle file saved from the QuantUS UI.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the ROI file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the ROI file name.
    """

    def __init__(self, seg_path: str, **kwargs):
        super().__init__()

        with open(seg_path, "rb") as f:
            roi_info: dict = pickle.load(f)
            
        if kwargs.get("assert_scan"):
            assert roi_info["Image Name"] == Path(kwargs["scan_path"]).name, 'Scan file name mismatch'
        if kwargs.get("assert_phantom"):
            assert roi_info["Phantom Name"] == Path(kwargs["phantom_path"]).name, 'Phantom file name mismatch'
        
        self.spline_x = np.array(roi_info["Spline X"])
        self.spline_y = np.array(roi_info["Spline Y"])
        self.frame = roi_info["Frame"]
