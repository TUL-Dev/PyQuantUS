# Standard library imports
import logging
import os
from typing import Tuple, List

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.signal import hilbert, stft
from scipy.optimize import curve_fit
from scipy.special import hermite, factorial
from scipy.fft import fft
from tqdm import tqdm

# Local application imports
from pyquantus.utc.objects import UltrasoundImage, AnalysisConfig, Window
from pyquantus.utc.transforms import computeHanningPowerSpec, computeSpectralParams

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


###################################################################################
# Gaussian Hermite Wavelet
###################################################################################
class GaussinaHermiteWavelet:
    """Class for computing the Gaussian Hermite wavelet."""
    
    ###################################################################################
    # Constructor method (initializer)
    ###################################################################################
    def __init__(self, order, fs, sigma, wavelet_duration, visualize) -> None:
                
        self.order = order
        self.fs = fs 
        self.sigma = sigma
        self.wavelet_duration = wavelet_duration
        self.visualize = visualize
    
        self.time = None
        self.wavelet = None

        self.__run()
        
    ###################################################################################
    # Run method
    ###################################################################################
    def __run(self):
        
        self.build_time_array()
        self.build_wavelet()
        self.get_central_freq_of_wavelet()
        self.plot()
    
    ###################################################################################
    # Build time array
    ###################################################################################
    def build_time_array(self):
        logging.info("Building time array...")

        # Get the sampling frequency rate
        time_step_second = 1 / (self.fs)
        logging.debug(f"Calculated time step (seconds): {time_step_second}")

        # Create a time array for the wavelet
        # Ensure the wavelet duration is a float for np.arange
        half_duration = self.wavelet_duration / 2  # Use float division
        self.time = np.arange(-half_duration, half_duration, time_step_second)
                    
        logging.info(f"Time array created with length: {len(self.time)}")
        
    ###################################################################################
    # Build wavelet
    ###################################################################################
    def build_wavelet(self):
        logging.debug(f"Starting normalized_hermite_polynomial with order={self.order}, time={self.time}, sigma={self.sigma}")

        # Physicists' Hermite polynomial
        H_n = hermite(self.order)
        logging.debug(f"Hermite polynomial of order {self.order} generated.")

        # Normalization factor (assuming custom energy calculation)
        energy = np.sqrt(np.pi / 2) * (factorial(2 * self.order) / (2**self.order * factorial(self.order)))
        normalization_factor = 1.0 / np.sqrt(energy)
        logging.debug(f"Calculated energy={energy} and normalization_factor={normalization_factor}")

        # Gaussian window
        gaussian_window = np.exp(-(self.time**2) / (1 * self.sigma**2))
        logging.debug(f"Computed Gaussian window with values: {gaussian_window}")

        # Construct the wavelet
        self.wavelet = normalization_factor * H_n(self.time / self.sigma) * gaussian_window
        logging.debug(f"Generated wavelet with values: {self.wavelet}")

    ###################################################################################
    # Get central frequency of wavelet
    ###################################################################################
    def get_central_freq_of_wavelet(self):
        logging.info("Starting central frequency calculation.")
        
        # Perform Fourier Transform
        fft_wavelet = np.fft.fft(self.wavelet)
        logging.debug(f"FFT of wavelet: {fft_wavelet}")

        # Convert sampling rate to Hz and calculate dt
        dt = 1 / self.fs

        # Compute the frequency bins
        n = len(self.wavelet)
        freq_bins = np.fft.fftfreq(n, d=dt)
        logging.debug(f"Frequency bins: {freq_bins}")

        # Compute the Power Spectrum
        power_spectrum = np.abs(fft_wavelet) ** 2
        logging.debug(f"Power Spectrum: {power_spectrum}")

        # Limit to positive frequencies
        positive_freqs = freq_bins > 0
        positive_freq_bins = freq_bins[positive_freqs]
        positive_power_spectrum = power_spectrum[positive_freqs]

        # Find the frequency with maximum amplitude in the positive spectrum
        max_amplitude_freq = positive_freq_bins[np.argmax(positive_power_spectrum)]
        logging.info(f"Frequency with maximum amplitude: {max_amplitude_freq / 1e6:.2f} MHz")

        # Replace print with logging
        logging.info(f"Frequency with Maximum Amplitude: {max_amplitude_freq / 1e6:.2f} MHz")

    ###################################################################################
    # Plot
    ###################################################################################
    def plot(self, show_negative_freqs=False, shift_time_positive=True):
        logging.info("Starting the plot process.")
        
        if self.visualize:
            logging.info("Visualization is enabled.")
            
            plt.figure(figsize=(12, 4))
            logging.info("Figure created with size 12x4.")

            # Plot the Hermite wavelet
            plt.subplot(1, 2, 1)
            
            # Shift time to positive if the option is enabled
            time_to_plot = self.time * 1e6
            if shift_time_positive:
                time_to_plot = (self.time - self.time.min()) * 1e6
                logging.info("Time shifted to positive values.")

            plt.plot(time_to_plot, self.wavelet, label=f'GH{self.order}')  # Time in µs
            
            # Shift vertical lines if time is shifted
            sigma_shift = self.time.min() if shift_time_positive else 0
            plt.axvline(x=(3 * self.sigma - sigma_shift) * 1e6, color='blue', linestyle='--', label='+3\u03C3')
            plt.axvline(x=(-3 * self.sigma - sigma_shift) * 1e6, color='blue', linestyle='--', label='-3\u03C3')
            
            plt.title('Hermite Wavelets')
            plt.xlabel('Time [µs]')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            logging.info("Hermite wavelet plot created with time scaled to microseconds.")

            # FFT and frequency response
            logging.info("Starting FFT calculation.")
            wavelet_fft = fft(self.wavelet)
            freqs = np.fft.fftfreq(len(wavelet_fft), d=(self.time[1] - self.time[0]))  # Frequency vector
            fft_magnitude = np.abs(wavelet_fft)  # Magnitude of FFT
            logging.info("FFT calculation completed.")

            plt.subplot(1, 2, 2)
            if show_negative_freqs:
                plt.plot(freqs / 1e6, fft_magnitude, label=f'FFT of GH{self.order}')  # Full spectrum
                plt.title('Full Frequency Spectrum of Hermite Wavelets')
            else:
                positive_mask = freqs >= 0
                plt.plot(freqs[positive_mask] / 1e6, fft_magnitude[positive_mask], label=f'FFT of GH{self.order}')  # Positive only
                plt.title('Positive Frequency Spectrum of Hermite Wavelets')
            
            plt.xlabel('Frequency [MHz]')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.grid()
            logging.info("Frequency response plot created.")

            plt.tight_layout()
            plt.show()
            logging.info("Plots displayed successfully.")
        else:
            logging.info("Visualization is disabled; no plots will be shown.")


###################################################################################
# H-scan
###################################################################################
class Hscan:
    """Class for computing the H-scan parameters."""
    

    ###################################################################################
    # Constructor
    ###################################################################################
    def __init__(self,
                 signal_nd: np.ndarray,
                 row_axis: int,
                 signal_axis: int,
                 frame_axis: int,
                 wavelet_GHx_params_1: dict,
                 wavelet_GHx_params_2: dict,
                 ) -> None:
        
        # input arguments
        self.signal_nd = signal_nd
        self.signal_axis = signal_axis
        self.row_axis = row_axis
        self.frame_axis = frame_axis
        self.wavelet_GHx_params_1 = wavelet_GHx_params_1
        self.wavelet_GHx_params_2 = wavelet_GHx_params_2
        self.wavelet_GH1 = self.GaussinaHermiteWavelet(**wavelet_GHx_params_1)
        self.wavelet_GH2 = self.GaussinaHermiteWavelet(**wavelet_GHx_params_2)
                                
        self.convolved_signal_with_ghx_1_nd = None
        self.convolved_signal_with_ghx_2_nd = None
        self.convolved_signal_with_ghx_1_envelope_nd = None
        self.convolved_signal_with_ghx_2_envelope_nd = None
        
        self.envelope_nd = None

        self.__run()

    ###################################################################################
    # Compute H-scan
    ###################################################################################
    def __run(self):
                
        # convolved signals
        self.convolve_signal_with_wavelets_nd()
        self.set_envelope_of_convolved_signals_nd() 
        self.set_envelope_of_signals_nd()
        
    ###################################################################################
    # Convolve 3D signal
    ###################################################################################
    def _convolve_signal_nd(self,
                            signal_nd: np.ndarray,
                            wavelet_1d: np.ndarray,
                            signal_axis: int) -> np.ndarray:
        """
        Convolve an n-dimensional signal with a 1D wavelet along the specified axis.
        This method is dimension-independent and works for any signal dimension.
        
        Args:
            signal_nd (np.ndarray): N-dimensional signal array to be convolved
            wavelet_1d (np.ndarray): 1D wavelet array to convolve with
            signal_axis (int): Axis along which to perform convolution
            
        Returns:
            np.ndarray: Convolved signal with same dimensions as input signal
            
        Raises:
            ValueError: If signal_axis is invalid or signal/wavelet dimensions are incompatible
        """
        try:
                
            if signal_axis >= signal_nd.ndim:
                raise ValueError(f"Axis {signal_axis} is out of bounds for signal with {signal_nd.ndim} dimensions")
            
            logging.info(f"Starting convolution of {signal_nd.ndim}D signal along axis {signal_axis}")
            
            # Create the convolution function for a single 1D signal
            def convolve_1d(signal_1d):
                return np.convolve(signal_1d, wavelet_1d, mode='same')
            
            # Apply the convolution along the specified axis using np.apply_along_axis
            # This automatically handles any number of dimensions
            convolved_signal = np.apply_along_axis(
                func1d=convolve_1d,
                axis=signal_axis,
                arr=signal_nd
            )
            
            logging.info(f"Successfully convolved {signal_nd.ndim}D signal with shape {signal_nd.shape}")
            return convolved_signal
            
        except Exception as e:
            logging.error(f"Error during convolution: {str(e)}")
            raise

    ###################################################################################
    # Convolve signal with wavelets 3D
    ###################################################################################
    def convolve_signal_with_wavelets_nd(self):
        logging.info("Starting convolution with wavelets")

        # Convolve with the first wavelet
        self.convolved_signal_with_ghx_1_nd = self._convolve_signal_nd(
            signal_nd=self.signal_nd,
            wavelet_1d=self.wavelet_GH1.wavelet,
            signal_axis=self.signal_axis
        )
        logging.info("Convolution with wavelet 1 completed")
        
        # Convolve with the second wavelet
        self.convolved_signal_with_ghx_2_nd = self._convolve_signal_nd(
            signal_nd=self.signal_nd,
            wavelet_1d=self.wavelet_GH2.wavelet,
            signal_axis=self.signal_axis
        )
        logging.info("Convolution with wavelet 2 completed")
                
    ###################################################################################
    # Get envelope of convolved signal 3D
    ###################################################################################
    def set_envelope_of_convolved_signals_nd(self):
        """Get the envelope of the convolved signal 3D using Hilbert transform.
        
        This method computes the envelope of both convolved signals (GH1 and GH2) using the Hilbert transform.
        The envelope is computed along the specified signal axis.
        """
        logging.info("Starting to compute the envelope of convolved signals in 3D.")

        try:
            # Compute envelope for first convolved signal (GH1)
            logging.info("Calculating envelope for the first convolved signal (GH1).")
            self.convolved_signal_with_ghx_1_envelope_nd = np.abs(hilbert(self.convolved_signal_with_ghx_1_nd,
                                                                          axis=self.signal_axis))
            logging.info("Envelope for the first convolved signal computed successfully.")

            # Compute envelope for second convolved signal (GH2)
            logging.info("Calculating envelope for the second convolved signal (GH2).")
            self.convolved_signal_with_ghx_2_envelope_nd = np.abs(hilbert(self.convolved_signal_with_ghx_2_nd,
                                                                          axis=self.signal_axis))
            logging.info("Envelope for the second convolved signal computed successfully.")

        except Exception as e:
            logging.error(f"An error occurred while computing the envelopes: {e}")
            raise  # Re-raise the exception after logging it
        finally:
            logging.info("Envelope computation process completed.")

    ###################################################################################
    # Get envelope of signals 3D
    ###################################################################################
    def set_envelope_of_signals_nd(self):
        """Get the envelope of the signals 3D."""
        self.envelope_nd = np.abs(hilbert(self.signal_nd, axis=self.signal_axis))


