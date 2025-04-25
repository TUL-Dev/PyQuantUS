import numpy as np
import nibabel as nib

from .._obj.us_rf_3d import BmodeSeg3d

class EntryClass(BmodeSeg3d):
    """
    Class for loading ROI data from a pickle file saved from the QuantUS UI.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the ROI file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the ROI file name.
    """

    def __init__(self, seg_path: str, **kwargs):
        super().__init__()

        seg = nib.load(seg_path)
        self.seg_mask = np.asarray(seg.dataobj, dtype=np.uint8)
        self.frame = 0
