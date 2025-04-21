import numpy as np
from typing import Tuple
from scipy.signal import hilbert
from pyquantus.cli.analysis._obj.paramap_base import Window

from ...config_loaders._obj.us_rf import RfAnalysisConfig
from ..spectral_paramap.main import EntryClass as ParamapEntryClass, ResultsClass
from ..spectral_paramap.transforms import compute_hanning_power_spec, compute_spectral_params
from ...seg_loaders._obj.us_rf import BmodeSeg
from ...scan_loaders._obj.us_rf import UltrasoundRfImage

class FullResults(ResultsClass):
    """Class to store results of analysis in each parametric map window.
    """
    def __init__(self):
        super().__init__()
        self.att_coef: float  # attenuation coefficient (dB/cm/MHz)
        self.bsc: float  # backscatter coefficient (1/cm-sr)
        self.u_nakagami: float  # Nakagami shape parameter
    

class EntryClass(ParamapEntryClass):
    """Class to complete ultrasound tissue characterization analysis
    and generate a corresponding parametric map on a given region of interest (ROI).
    """
    
    def __init__(self, image_data: UltrasoundRfImage, config: RfAnalysisConfig, seg_data: BmodeSeg, 
                 results_class: type = FullResults, **kwargs):
        super().__init__(image_data, config, seg_data, results_class, **kwargs)
        
        self.ref_attenuation = kwargs.get("ref_attenuation", None)
        assert self.ref_attenuation is not None, "No reference attenuation coefficient was provided!"
        assert isinstance(self.ref_attenuation, float), "Reference attenuation coefficient must be a float!"
        
        self.ref_backscatter_coef = kwargs.get("ref_backscatter_coef", None)
        assert self.ref_backscatter_coef is not None, "No reference backscatter coefficient was provided!"
        assert isinstance(self.ref_backscatter_coef, float), "Reference backscatter coefficient must be a float!"
        
    def compute_window_vals(self, scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, window: Window):
        # Spectral analysis from before. Results saved in `window` object
        super().compute_window_vals(scan_rf_window, phantom_rf_window, window)
        
        # Additional analysis
        att_coef = self.compute_attenuation_coef(scan_rf_window, phantom_rf_window, window_depth=min(100, scan_rf_window.shape[0]//3))
        bsc = self.compute_backscatter_coefficient(window.results.f, window.results.ps, window.results.r_ps, 
                                                att_coef, self.config.center_frequency, roi_depth=scan_rf_window.shape[0])
        _, u_nakagami = self.compute_nakagami_params(scan_rf_window)
        window.results.att_coef = att_coef  # dB/cm/MHz
        window.results.bsc = bsc  # 1/cm-sr
        window.results.u_nakagami = u_nakagami
        
    ################# CUSTOM FUNCTIONS #################        
        
    def compute_attenuation_coef(self, rf_data: np.ndarray, ref_rf_data: np.ndarray, overlap=50, window_depth=100) -> float:
        """Compute the local attenuation coefficient of the ROI using the Spectral Difference
        Method for Local Attenuation Estimation. This method computes the attenuation coefficient
        for multiple frequencies and returns the slope of the attenuation as a function of frequency.

        Args:
            rf_data (np.ndarray): RF data of the ROI (n lines x m samples).
            ref_rf_data (np.ndarray): RF data of the phantom (n lines x m samples).
            overlap (float): Overlap percentage for analysis windows.
            window_depth (int): Depth of each window in samples.

        Returns:
            float: Local attenuation coefficient of the ROI for the central frequency (dB/cm/MHz).
            Updated and verified : Feb 2025 - IR
        """
        sampling_frequency = self.config.sampling_frequency
        start_frequency = self.config.analysis_freq_band[0]
        end_frequency = self.config.analysis_freq_band[1]

        # Initialize arrays for storing intensities (log of power spectrum for each frequency)
        ps_sample = []  # ROI power spectra
        ps_ref = []     # Phantom power spectra

        start_idx = 0
        end_idx = window_depth
        window_center_indices = []
        counter = 0

        # Loop through the windows in the RF data
        while end_idx < rf_data.shape[0]:
            sub_window_rf = rf_data[start_idx:end_idx]
            f, ps = compute_hanning_power_spec(sub_window_rf, start_frequency, end_frequency, sampling_frequency)
            ps_sample.append(20 * np.log10(ps))  # Log scale intensity for the ROI

            ref_sub_window_rf = ref_rf_data[start_idx:end_idx]
            ref_f, ref_ps = compute_hanning_power_spec(ref_sub_window_rf, start_frequency, end_frequency, sampling_frequency)
            ps_ref.append(20 * np.log10(ref_ps))  # Log scale intensity for the phantom

            window_center_indices.append((start_idx + end_idx) // 2)

            start_idx += int(window_depth * (1 - (overlap / 100)))
            end_idx = start_idx + window_depth
            counter += 1

        # Convert window depths to cm
        axial_res_cm = self.image_data.axial_res / 10
        window_depths_cm = np.array(window_center_indices) * axial_res_cm

        attenuation_coefficients = []  # One coefficient for each frequency

        f = f / 1e6
        ps_sample = np.array(ps_sample)
        ps_ref = np.array(ps_ref)

        mid_idx = f.shape[0] // 2
        start_idx = max(0, mid_idx - 25)
        end_idx = min(f.shape[0], mid_idx + 25)

        # Compute attenuation for each frequency
        for f_idx in range(start_idx, end_idx):
            normalized_intensities = np.subtract(ps_sample[:, f_idx], ps_ref[:, f_idx])
            p = np.polyfit(window_depths_cm, normalized_intensities, 1)
            local_attenuation = self.ref_attenuation * f[f_idx] - (1 / 4) * p[0]  # dB/cm
            attenuation_coefficients.append(local_attenuation / f[f_idx])  # dB/cm/MHz

        attenuation_coef = np.mean(attenuation_coefficients)
        return attenuation_coef

    def compute_backscatter_coefficient(self, freq_arr: np.ndarray, scan_ps: np.ndarray, ref_ps: np.ndarray,
                                    att_coef: float, frequency: int, roi_depth: int) -> float:
        """Compute the backscatter coefficient of the ROI using the reference phantom method.
        Assumes instrumentation and beam terms have the same effect on the signal from both 
        image and phantom. 

        Source: Yao et al. (1990): https://doi.org/10.1177/016173469001200105. PMID: 2184569

        Args:
            freq_arr (np.ndarray): Frequency array of power spectra (Hz).
            scan_ps (np.ndarray): Power spectrum of the analyzed scan at the current region.
            ref_ps (np.ndarray): Power spectrum of the reference phantom at the current region.
            att_coef (float): Attenuation coefficient of the current region (dB/cm/MHz).
            frequency (int): Frequency on which to compute backscatter coefficient (Hz).
            roi_depth (int): Depth of the start of the ROI in samples.
            
        Returns:
            float: Backscatter coefficient of the ROI for the central frequency (1/cm-sr).
            Updated and verified : Feb 2025 - IR
        """
        index = np.argmin(np.abs(freq_arr - frequency))
        ps_sample = scan_ps[index]
        ps_ref = ref_ps[index]
        s_ratio = ps_sample / ps_ref

        np_conversion_factor = np.log(10) / 20 
        converted_att_coef = att_coef * np_conversion_factor  # dB/cm/MHz -> Np/cm/MHz
        converted_ref_att_coef = self.ref_attenuation * np_conversion_factor  # dB/cm/MHz -> Np/cm/MHz

        window_depth_cm = roi_depth * self.image_data.axial_res / 10  # cm
        converted_att_coef *= frequency / 1e6  # Np/cm
        converted_ref_att_coef *= frequency / 1e6  # Np/cm        

        att_comp = np.exp(4 * window_depth_cm * (converted_att_coef - converted_ref_att_coef)) 
        bsc = s_ratio * self.ref_backscatter_coef * att_comp

        return bsc


    def compute_nakagami_params(self, rf_data: np.ndarray) -> Tuple[float, float]:
        """Compute Nakagami parameters for the ROI.

        Source: Tsui, P. H., Wan, Y. L., Huang, C. C. & Wang, M. C. 
        Effect of adaptive threshold filtering on ultrasonic Nakagami 
        parameter to detect variation in scatterer concentration. Ultrason. 
        Imaging 32, 229–242 (2010). https://doi.org/10.1177%2F016173461003200403

        Args:
            rf_data (np.ndarray): RF data of the ROI (n lines x m samples).
            
        Returns:
            Tuple: Nakagami parameters (w, u) for the ROI.
        """
        r = np.abs(hilbert(rf_data, axis=1))
        w = np.nanmean(r ** 2, axis=1)
        u = (w ** 2) / np.var(r ** 2, axis=1)

        # Averaging to get single parameter values
        w = np.nanmean(w)
        u = np.nanmean(u)

        return w, u
