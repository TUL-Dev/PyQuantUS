import numpy as np

from ...config_loaders._obj.us_rf import RfAnalysisConfig
from .transforms import compute_hanning_power_spec, compute_spectral_params
from .._obj.paramap_base import ParamapAnalysis, Window
from ...scan_loaders._obj.us_rf import UltrasoundRfImage
from ...seg_loaders._obj.us_rf import BmodeSeg

class ResultsClass:
    """Class to store results of spectral analysis in each parametric map window.
    """
    def __init__(self):
        self.mbf: float = 0.0
        self.ss: float = 0.0
        self.si: float = 0.0
        self.f: np.ndarray = np.array([])
        self.nps: np.ndarray = np.array([])
        self.ps: np.ndarray = np.array([])
        self.r_ps: np.ndarray = np.array([])

class EntryClass(ParamapAnalysis):
    """Class to complete spectral analysis (i.e. midband fit, spectral slope, spectral intercept)
    and generate a correspondingparametric map.
    """
    
    def __init__(self, image_data: UltrasoundRfImage, config: RfAnalysisConfig, seg_data: BmodeSeg, results_class: type = ResultsClass, **kwargs):
        super().__init__(results_class)
        assert isinstance(image_data, UltrasoundRfImage), 'image_data must be an UltrasoundRfImage child class'
        assert isinstance(config, RfAnalysisConfig), 'config must be an RfAnalysisConfig'
        assert isinstance(seg_data, BmodeSeg), 'seg_data must be a BmodeSeg'
        
        self.image_data: UltrasoundRfImage = image_data
        self.config: RfAnalysisConfig = config
        self.seg_data: BmodeSeg = seg_data
        
        # Define the ROI spline on the same coordinates as the RF data
        self.spline_x = self.seg_data.spline_x
        self.spline_y = self.seg_data.spline_y

    def compute_window_vals(self, scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, window: Window):
        """Compute spectral analysis values for a single window.
        
        Args:
            scan_rf_window (np.ndarray): RF data of the window in the scan image.
            phantom_rf_window (np.ndarray): RF data of the window in the phantom image.
            window (Window): Window object to store results.
        """
        f, ps = compute_hanning_power_spec(
            scan_rf_window, self.config.transducer_freq_band[0], 
            self.config.transducer_freq_band[1], self.config.sampling_frequency
        ) 
        ps = 20 * np.log10(ps)
        f, rPs = compute_hanning_power_spec(
            phantom_rf_window, self.config.transducer_freq_band[0], 
            self.config.transducer_freq_band[1], self.config.sampling_frequency
        )
        rPs = 20 * np.log10(rPs)
        nps = np.asarray(ps) - np.asarray(rPs)
        
        window.results.nps = nps
        window.results.ps = np.asarray(ps)
        window.results.r_ps = np.asarray(rPs)
        window.results.f = np.asarray(f)

        mbf, _, _, p = compute_spectral_params(nps, f, self.config.analysis_freq_band[0], self.config.analysis_freq_band[1])
        window.results.mbf = mbf # dB
        window.results.ss = p[0]*1e6 # dB/MHz
        window.results.si = p[1] # dB