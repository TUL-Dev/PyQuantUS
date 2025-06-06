import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft


class AttenuationCorrection:
    def __init__(self, signal, trimmed_time_1d, fs, speed_of_sound, alpha, window='hann', nperseg=64, noverlap=32, visualize=False):
        
        self.signal = signal
        self.trimmed_time_1d = trimmed_time_1d
        self.fs = fs
        self.speed_of_sound = speed_of_sound
        self.alpha = alpha
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.visualize = visualize

    def deattenuate_signal_with_constant_alpha_stft(self):
        """
        Applies deattenuation correction to a 1D signal using STFT.

        Returns:
            np.ndarray: Deattenuated signal.
        """
        logging.debug("Starting deattenuation_stft method.")

        # Step 1: Compute the STFT of the signal
        f, t, Zxx = stft(self.signal, fs=self.fs, window=self.window, nperseg=self.nperseg, noverlap=self.noverlap)
        logging.info("Computed STFT of the signal.")

        # Step 2: Calculate depth from the STFT time vector
        # Create a new time array with the same start and end points as self.trimmed_time_1d
        start_point = self.trimmed_time_1d[0]
        end_point = self.trimmed_time_1d[-1]
        new_trimmed_time_1d = np.linspace(start_point, end_point, num=len(t))
        depth = (self.speed_of_sound * new_trimmed_time_1d) / 2  # Depth array corresponding to time vector t in cm
        depth_cm = depth * 100  # Convert depth to centimeters
        logging.debug(f"Calculated depth: {depth_cm}")
        
        # Convert frequency to MHz
        f_MHz = f / 1e6

        # Step 3: Define the attenuation factors based on frequency and depth using alpha
        attenuation_factors = np.exp(- self.alpha * np.abs(f_MHz[:, None]) * depth_cm[None, :] / (20 * np.log10(math.e)))
        logging.info("Defined attenuation factors.")

        # Step 4: Calculate deattenuation factors as the inverse of the attenuation factors
        deattenuation_factors = 1 / attenuation_factors
        logging.info("Calculated deattenuation factors.")

        # Step 5: Apply deattenuation correction based on frequency content
        corrected_Zxx = Zxx * deattenuation_factors
        logging.debug("Applied deattenuation correction.")

        # Step 6: Reconstruct the signal using ISTFT
        _, corrected_signal = istft(corrected_Zxx, fs=self.fs, window=self.window, nperseg=self.nperseg, noverlap=self.noverlap)
        logging.info("Reconstructed the signal using ISTFT.")

        # Step 7: Trim the corrected signal to match the original signal size
        trim_size = (len(corrected_signal) - len(self.signal)) // 2
        corrected_signal = corrected_signal[trim_size:len(corrected_signal) - trim_size]
        
        # Visualize Zxx and corrected Zxx if visualize is True
        if self.visualize:
            plt.figure(figsize=(12, 6))

            # Plot original STFT magnitude
            plt.subplot(2, 1, 1)
            plt.title('STFT Magnitude of Original Signal')
            plt.pcolormesh(depth_cm, f_MHz, np.log(np.abs(Zxx)), shading='gouraud')
            plt.ylabel('Frequency (MHz)')
            plt.xlabel('Depth (cm)')
            plt.colorbar(label='Magnitude')

            # Plot corrected STFT magnitude
            plt.subplot(2, 1, 2)
            plt.title('STFT Magnitude of Corrected Signal')
            plt.pcolormesh(depth_cm, f_MHz, np.log(np.abs(corrected_Zxx)), shading='gouraud')
            plt.ylabel('Frequency (MHz)')
            plt.xlabel('Depth (cm)')
            plt.colorbar(label='Magnitude')

            plt.tight_layout()
            plt.show()
            logging.info("Visualized STFT magnitudes.")

        logging.debug("Finished deattenuation_stft method.")
        return corrected_signal