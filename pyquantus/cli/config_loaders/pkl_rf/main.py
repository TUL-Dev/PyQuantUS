import pickle
from pathlib import Path

from .._obj.us_rf import RfAnalysisConfig

class EntryClass(RfAnalysisConfig):
    """Class to load RF analysis configuration data from a pickle file saved from the QuantUS UI.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    
    def __init__(self, analysis_path: str, **kwargs):
        super().__init__()

        with open(analysis_path, "rb") as f:
            config_pkl: dict = pickle.load(f)
            
        if kwargs.get("assert_scan"):
            assert config_pkl["Image Name"] == Path(kwargs["scan_path"]).name, 'Scan file name mismatch'
        if kwargs.get("assert_phantom"):
            assert config_pkl["Phantom Name"] == Path(kwargs["phantom_path"]).name, 'Phantom file name mismatch'
        
        config = config_pkl["Config"]
        self.transducer_freq_band = config.transducerFreqBand
        self.analysis_freq_band = config.analysisFreqBand
        self.sampling_frequency = config.samplingFrequency
        self.ax_win_size = config.axWinSize
        self.lat_win_size = config.latWinSize
        self.window_thresh = config.windowThresh
        self.axial_overlap = config.axialOverlap
        self.lateral_overlap = config.lateralOverlap
        self.center_frequency = config.centerFrequency
