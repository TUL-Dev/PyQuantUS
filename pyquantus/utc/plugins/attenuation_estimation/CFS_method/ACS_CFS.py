###################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.optimize import curve_fit
import logging

class AttenuationCoefficientEstimationCFS:
    """
    Compute the local attenuation coefficient of the ROI using the Spectral Difference Method for Local Attenuation Estimation.
    Note this method cannot be used to estimate attenuation when the scattering properties change within the ROI. This method assumes a tissue-mimicking phantom with a sound speed similar to the expected sound speed of the analyzed tissue. Assuming linear dependency on depth, the local attenuation coefficient computed here is equivalent to the total attenuation coefficient.
    
    Source: Mamou & Oelze, page 70-71: https://doi.org/10.1007/978-94-007-6952-6

    Args:
        rfData (np.ndarray): RF data of the ROI (n lines x m samples).
        refRfData (np.ndarray): RF data of the phantom (n lines x m samples).
        roiDepth (int): Depth of the start of the ROI in samples.
        overlap (float): Overlap percentage for analysis windows.
        windowDepth (int): Depth of each window in samples.

    Returns:
        float: Local attenuation coefficient of the ROI for the central frequency (dB/cm/MHz).
    """
    def __init__(self, signal, fs, time_array, visualize=True, speed_of_sound=1540, nperseg=64, overlap=32):
        """
        Initialize the estimator with optional parameters.
        Args:
            signal (np.ndarray): The input signal.
            fs (float): Sampling frequency in Hz.
            time_array (np.ndarray): Time array corresponding to the signal.
            visualize (bool): Whether to plot intermediate results. Default is True.
            speed_of_sound (float): Speed of sound in m/s. Default is 1540.
            nperseg (int): Number of samples per STFT segment. Default is 64.
            overlap (int): Number of overlapping samples for STFT. Default is 32.
        """
        # input parameters
        self.signal = signal
        self.fs = fs
        self.time_array = time_array
        self.visualize = visualize
        
        # processing parameters
        self.speed_of_sound = speed_of_sound
        self.nperseg = nperseg
        self.overlap = overlap
        
        # output parameters
        self.estimated_alpha = None # dB/cm/MHz
        self.frequencies = None
        self.times = None
        self.amplitudes = None
        self.z_values = None
        self.estimated_alpha_raw = None
        
        # processing parameters
        self.center_frequencies = []
        self.gaussian_fits = []

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.__run()

    ###################################################################################

    def __run(self):
        
        self.logger.info('Starting attenuation coefficient estimation run.')
        
        self._compute_stft()
        self._calculate_depth()
        self._fit_gaussians()
        self._fit_linear_model()
        
        if self.visualize:
            self._visualize_spectrogram()
            self._visualize_fitting()
            self._visualize_segments()
            
        self.estimated_alpha = self.estimated_alpha_raw * 34.72 # dB/cm/MHz
        
        self.logger.info(f'Attenuation coefficient estimation complete. Estimated alpha: {self.estimated_alpha:.4f} dB/cm/MHz')

    ###################################################################################
    @staticmethod
    def _gaussian(x, a, x0, sigma):
        """Gaussian function for curve fitting."""
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    ###################################################################################
    
    def _compute_stft(self):
        self.logger.info('Computing STFT...')
        self.frequencies, self.times, Zxx = stft(self.signal, fs=self.fs, nperseg=self.nperseg, noverlap=self.overlap)
        self.amplitudes = np.abs(Zxx)
        self.logger.debug(f'STFT computed. Shape of amplitudes: {self.amplitudes.shape}')
        
    ###################################################################################

    def _calculate_depth(self):
        self.logger.info('Calculating depth values...')
        # Depth = (speed_of_sound * time) / 2 (to account for round-trip), then convert to cm
        self.z_values = (self.speed_of_sound * self.times) / 2 * 100  # in cm
        self.logger.debug(f'Depth values calculated. z_values shape: {self.z_values.shape}')

    ###################################################################################
    
    def _fit_gaussians(self):
        self.logger.info('Fitting Gaussians to each spectrum...')
        for i in range(self.amplitudes.shape[1]):
            spectrum = self.amplitudes[:, i]
            initial_guess = [np.max(spectrum), self.frequencies[np.argmax(spectrum)], 1e6]
            try:
                popt, _ = curve_fit(self._gaussian, self.frequencies, spectrum, p0=initial_guess)
                self.center_frequencies.append(popt[1])
                self.gaussian_fits.append(popt)
            except RuntimeError:
                self.logger.warning(f'Gaussian fit failed at index {i}, using max spectrum frequency.')
                self.center_frequencies.append(self.frequencies[np.argmax(spectrum)])
                self.gaussian_fits.append(None)
        self.center_frequencies = np.array(self.center_frequencies) / 1e6  # Convert to MHz
        self.logger.debug('Gaussian fitting complete.')

    ###################################################################################
    
    def _fit_linear_model(self):
        self.logger.info('Fitting linear model to center frequencies...')
        sigma_ws = [fit[2] for fit in self.gaussian_fits if fit is not None]
        self.sigma_w = np.mean(sigma_ws) / 1e6
        self.f0 = self.center_frequencies[0]
        def linear_model(z, alpha):
            return self.f0 - 4 * self.sigma_w**2 * alpha * z
        self.linear_model = linear_model
        popt, _ = curve_fit(self.linear_model, self.z_values, self.center_frequencies)
        self.estimated_alpha_raw = popt[0]
        self.logger.info(f'Linear model fit complete. Raw estimated alpha: {self.estimated_alpha_raw:.6f}')

    ###################################################################################

    def _visualize_spectrogram(self):
        self.logger.info('Visualizing STFT spectrogram...')
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(self.z_values, self.frequencies / 1e6, self.amplitudes, shading='gouraud', cmap='jet')
        plt.colorbar(label="Amplitude")
        plt.xlabel("Depth (cm)")
        plt.ylabel("Frequency (MHz)")
        plt.title("STFT Spectrogram of the Signal")
        plt.ylim([0, np.max(self.frequencies) / 1e6])
        plt.show()

    ###################################################################################
    
    def _visualize_fitting(self):
        self.logger.info('Visualizing linear fit to center frequencies...')
        plt.figure(figsize=(10, 6))
        plt.scatter(self.z_values, self.center_frequencies, label="Observed Center Frequencies", color="red", marker="o")
        plt.plot(self.z_values, self.linear_model(self.z_values, self.estimated_alpha_raw),
                 label=f"Fitted Line (α = {self.estimated_alpha_raw:.4f})", linestyle="--", color="blue")
        plt.xlabel("Depth (cm)")
        plt.ylabel("Center Frequency (MHz)")
        plt.title("Curve Fitting to Estimate Attenuation Coefficient")
        plt.legend()
        plt.grid()
        plt.show()

    ###################################################################################
    
    def _visualize_segments(self):
        self.logger.info('Visualizing signal segments and their spectra...')
        for i, t in enumerate(self.times):
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            segment_start = int(t * self.fs)
            segment_end = int((t + self.nperseg / self.fs) * self.fs)
            segment = self.signal[segment_start:segment_end]
            axes[0].plot(segment, color='blue')
            axes[0].set_title(f"Signal Segment at Depth {self.z_values[i]:.2f} cm")
            axes[0].set_xlabel("Sample Index")
            axes[0].set_ylabel("Amplitude")
            axes[1].plot(self.frequencies / 1e6, self.amplitudes[:, i], color='green', label="Observed Spectrum")
            if self.gaussian_fits[i] is not None:
                fitted_curve = self._gaussian(self.frequencies, *self.gaussian_fits[i])
                axes[1].plot(self.frequencies / 1e6, fitted_curve, color='red', linestyle="--", label="Gaussian Fit")
                axes[1].axvline(self.center_frequencies[i], color='blue', linestyle=":", label=f"Peak @ {self.center_frequencies[i]:.2f} MHz")
            axes[1].set_title(f"Frequency Spectrum at Depth {self.z_values[i]:.2f} cm")
            axes[1].set_xlabel("Frequency (MHz)")
            axes[1].set_ylabel("Amplitude")
            axes[1].legend()
            plt.tight_layout()
            plt.show()

    ###################################################################################
