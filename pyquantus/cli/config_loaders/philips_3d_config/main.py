import pickle
from pathlib import Path

from .._obj.us_rf_3d import RfAnalysisConfig3d

class EntryClass(RfAnalysisConfig3d):
    """Class to load RF analysis configuration data from a pickle file saved from the QuantUS UI.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    
    def __init__(self, analysis_path: str, **kwargs):
        super().__init__()

        self.transducer_freq_band = [0, 7000000] # [min, max] (Hz)
        self.analysis_freq_band = [2500000, 5000000] # [lower, upper] (Hz)
        self.ax_win_size = 10 # axial length per window (mm)
        self.lat_win_size = 10 # lateral length per window (mm)
        self.cor_win_size = 20 # coronal length per window (mm)
        self.window_thresh = 0.95 # % of window area required to be considered in ROI
        self.axial_overlap = 0.5 # % of window overlap in axial direction
        self.lateral_overlap = 0.5 # % of window overlap in lateral direction
        self.coronal_overlap = 0.5 # % of window overlap in coronal direction
        self.center_frequency = 4500000 # Hz
        self.sampling_frequency = 4*self.center_frequency # Hz
