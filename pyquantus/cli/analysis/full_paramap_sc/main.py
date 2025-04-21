import numpy as np

from ...config_loaders._obj.us_rf import RfAnalysisConfig
from ..full_paramap.main import EntryClass as FullParamapEntryClass, FullResults
from ...scan_loaders._obj.us_rf import ScUltrasoundRfImage
from ...seg_loaders._obj.us_rf import BmodeSeg

class EntryClass(FullParamapEntryClass):
    """Class to complete spectral analysis (i.e. midband fit, spectral slope, spectral intercept)
    and generate a corresponding parametric map on scan converted data by undoing
    scan conversion on the ROI spline.
    """
    
    def __init__(self, image_data: ScUltrasoundRfImage, config: RfAnalysisConfig, seg_data: BmodeSeg, 
                 results_class: type = FullResults, **kwargs):
        super().__init__(image_data, config, seg_data, results_class, **kwargs)
        assert isinstance(image_data, ScUltrasoundRfImage), 'image_data must be an ScUltrasoundRfImage child class'
        assert isinstance(config, RfAnalysisConfig), 'config must be an RfAnalysisConfig'
        assert isinstance(seg_data, BmodeSeg), 'seg_data must be a BmodeSeg'
        
        self.image_data: ScUltrasoundRfImage = image_data
        self.config: RfAnalysisConfig = config
        self.seg_data: BmodeSeg = seg_data
        
        # Define the ROI spline on the same coordinates as the RF data
        self.spline_x = np.array([self.image_data.xmap[int(y), int(x)] for x, y in zip(self.seg_data.spline_x, self.seg_data.spline_y)])
        self.spline_y = np.array([self.image_data.ymap[int(y), int(x)] for x, y in zip(self.seg_data.spline_x, self.seg_data.spline_y)])
