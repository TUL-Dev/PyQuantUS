import numpy as np
from scipy.ndimage import binary_closing

from ...config_loaders._obj.us_rf_3d import RfAnalysisConfig3d
from ..spectral_paramap_3d.main import EntryClass as ParamapEntryClass3d, ResultsClass
from ...scan_loaders._obj.us_rf_3d import ScUltrasoundRfImage3d
from ...seg_loaders._obj.us_rf_3d import BmodeSeg3d
from .transforms import map_1d_to_3d

class EntryClass(ParamapEntryClass3d):
    """Class to complete spectral analysis (i.e. midband fit, spectral slope, spectral intercept)
    and generate a corresponding parametric map on scan converted data by undoing
    scan conversion on the ROI spline.
    """
    
    def __init__(self, image_data: ScUltrasoundRfImage3d, config: RfAnalysisConfig3d, seg_data: BmodeSeg3d, 
                 results_class: type = ResultsClass, **kwargs):
        super().__init__(image_data, config, seg_data, results_class, **kwargs)
        assert isinstance(image_data, ScUltrasoundRfImage3d), 'image_data must be an ScUltrasoundRfImage child class'
        assert isinstance(config, RfAnalysisConfig3d), 'config must be an RfAnalysisConfig'
        assert isinstance(seg_data, BmodeSeg3d), 'seg_data must be a BmodeSeg'
        
        self.image_data: ScUltrasoundRfImage3d = image_data
        self.config: RfAnalysisConfig3d = config
        self.seg_data: BmodeSeg3d = seg_data
        
        # Define the ROI spline on the same coordinates as the RF data
        pre_sc_bmode = self.image_data.bmode
        coord_map = self.image_data.coord_map_3d
        voi_mask = self.seg_data.seg_mask
        masked_coords_3d = np.transpose(np.where(voi_mask))
        original_coords_3d = []

        for coord in masked_coords_3d:
            original_coords_3d.append(
                map_1d_to_3d(
                    coord_map[coord[0], coord[1], coord[2]],
                    pre_sc_bmode.shape[0],
                    pre_sc_bmode.shape[1],
                    pre_sc_bmode.shape[2]
                )
            )

        # Create a blank 3D mask with the same shape as the pre-scan converted B-mode image
        self.seg_mask = np.zeros_like(pre_sc_bmode, dtype=np.uint8)
        self.seg_mask[tuple(np.transpose(original_coords_3d))] = 1
        self.seg_mask = binary_closing(self.seg_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
