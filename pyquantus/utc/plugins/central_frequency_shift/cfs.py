# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.optimize import curve_fit

###################################################################################
# Central frequency shift
###################################################################################
class CentralFrequencyShift:
    """Class for computing the central frequency shift for the ROI.
    
    This class implements methods to analyze the central frequency shift of ultrasound signals
    using Short-Time Fourier Transform (STFT) and Gaussian fitting. It can be used to estimate
    attenuation coefficients based on frequency shifts with depth.
    
    Attributes:
        signal (np.ndarray): Input signal to analyze
        fs (float): Sampling frequency in Hz
        time_array (np.ndarray): Time values corresponding to the signal
        speed_of_sound (float): Speed of sound in m/s (default: 1540)
        stft_params (dict): Parameters for STFT computation
    """
    
    ###################################################################################
    # Constructor
    ###################################################################################
    def __init__(self,
                 signal_1d: np.ndarray,
                 sampling_frequency_MHz: float,
                 time_array_s: np.ndarray, 
                 depth_array_cm: np.ndarray,
                 speed_of_sound_m_s: float = 1540,
                 stft_params: dict = None,
                 use_fft: bool = False):
        """Initialize the CentralFrequencyShift class.
        
        Args:
            signal (np.ndarray): Input signal to analyze
            fs (float): Sampling frequency in Hz
            time_array (np.ndarray): Time values corresponding to the signal
            speed_of_sound (float, optional): Speed of sound in m/s. Defaults to 1540.
            stft_params (dict, optional): Parameters for STFT computation. If None, uses defaults.
            use_fft (bool, optional): Whether to use FFT instead of STFT. Defaults to False.
        """
        # input arguments
        self.signal_1d = signal_1d
        self.fs = sampling_frequency_MHz * 1e6
        self.time_array_s = time_array_s
        self.depth_array_cm = depth_array_cm
        self.speed_of_sound_m_s = speed_of_sound_m_s
        self.stft_params = stft_params
        self.use_fft = use_fft
            
        # Initialize computed attributes
        self.frequencies_stft = None
        self.times_stft = None
        self.Zxx_stft = None
        self.amplitudes_stft = None
        self.center_frequencies = None
        self.fitted_sigmas = None
        self.gaussian_fits = None
        
        self.__run()
        
    ###################################################################################
    # Main method to compute central frequency shift and estimate attenuation coefficient.
    ###################################################################################
    def __run(self):

        self.set_signal_zero_mean()
        self.compute_stft()
        self.adjust_depth_array()
        self.fit_gaussian_peaks()    
        self.plot_stft_spectrogram()
        self.plot_signal_segments(use_gaussian_fit=False)
        self.plot_frequency_shift()

    ###################################################################################
    # Set signal zero mean
    ###################################################################################
    def set_signal_zero_mean(self):
        """Set the signal zero mean."""
        self.signal_1d = self.signal_1d - np.mean(self.signal_1d)

    ###################################################################################
    # Compute STFT
    ###################################################################################
    def compute_stft(self):
        """Compute the STFT of the signal."""
        self.frequencies_stft, self.times_stft, self.Zxx_stft = stft(self.signal_1d, 
                                        fs=self.fs,
                                        nperseg=self.stft_params['nperseg'],
                                        noverlap=self.stft_params['overlap']
                                        )
        
        # Compute the amplitudes of the STFT
        self.amplitudes_stft = np.abs(self.Zxx_stft)

    ###################################################################################
    # Compute depth values
    ###################################################################################
    def adjust_depth_array(self):
        """Adjust depth array to match the number of STFT time segments."""
        
        # Interpolate depth values to align with STFT time segments
        if len(self.depth_array_cm) != len(self.times_stft):
            self.depth_array_cm = np.interp(
                np.linspace(0, len(self.depth_array_cm) - 1, len(self.times_stft)),
                np.arange(len(self.depth_array_cm)),
                self.depth_array_cm
            )
        
        # Remove the last depth value to avoid edge effects
        self.depth_array_cm = self.depth_array_cm[:-1]

    ###################################################################################
    # Fit Gaussian peaks
    ###################################################################################
    def fit_gaussian_peaks(self):
        """Fit Gaussian curves to frequency spectra at each depth."""
        
        center_frequencies = []
        fitted_sigmas = []
        gaussian_fits = []

        for i in range(len(self.depth_array_cm)):
            spectrum = self.amplitudes_stft[:, i]
                        
            # Initial guesses for Gaussian fit
            initial_guess = [
                np.max(spectrum),
                self.frequencies_stft[np.argmax(spectrum)],
                1e6
            ]
            
            try:
                # Fit Gaussian to the frequency spectrum
                popt, _ = curve_fit(
                    self.gaussian,
                    self.frequencies_stft,
                    spectrum,
                    p0=initial_guess
                )
                fitted_amplitude, fitted_center, fitted_sigma = popt
                center_frequencies.append(fitted_center)
                fitted_sigmas.append(fitted_sigma)
                gaussian_fits.append(popt)
            except RuntimeError:
                # Fallback to max frequency if fit fails
                center_frequencies.append(self.frequencies_stft[np.argmax(spectrum)])
                fitted_sigmas.append(None)
                gaussian_fits.append(None)

        # Convert center frequencies to MHz
        self.center_frequencies = np.array(center_frequencies) / 1e6
        self.fitted_sigmas = fitted_sigmas
        self.gaussian_fits = gaussian_fits

    ###################################################################################
    # Gaussian function
    ###################################################################################
    def gaussian(self, x, a, x0, sigma):
        """Gaussian function for peak fitting."""
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    ###################################################################################
    # Plot STFT spectrogram
    ###################################################################################
    def plot_stft_spectrogram(self, log_scale=True):
        """Plot the STFT spectrogram."""
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(
            self.times_stft,
            self.frequencies_stft / 1e6,
            np.log(self.amplitudes_stft) if log_scale else self.amplitudes_stft,
            shading='gouraud',
            cmap='jet'
        )
        plt.colorbar(label="Amplitude")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (MHz)")
        plt.title("STFT Spectrogram of the Signal")
        plt.ylim([0, np.max(self.frequencies_stft) / 1e6])
        plt.show()

    ###################################################################################
    # Plot signal segments
    ###################################################################################
    def plot_signal_segments(self, use_gaussian_fit: bool = False):
        """Plot individual signal segments with their frequency spectra.
        
        Args:
            use_gaussian_fit (bool): Whether to use Gaussian fitting before finding maximum frequency.
                If False, directly finds the frequency with maximum amplitude.
        """
        for i, t in enumerate(self.times_stft[:-1]):
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            
            # Extract signal segment in time domain
            segment = self.signal_1d[int(t * self.fs): 
                                int((t + self.stft_params['nperseg'] / self.fs) * self.fs)]
            
            # Make segment zero mean
            segment = segment - np.mean(segment)
            
            # Plot time domain signal
            axes[0].plot(segment, color='blue')
            axes[0].set_title(f"Signal Segment at Depth {self.depth_array_cm[i]:.2f} cm")
            axes[0].set_xlabel("Sample Index")
            axes[0].set_ylabel("Amplitude")
            
            # Get positive frequencies only for STFT
            positive_mask = self.frequencies_stft >= 0
            frequencies = self.frequencies_stft[positive_mask]
            amplitudes = self.amplitudes_stft[positive_mask, i]
            
            # Plot frequency spectrum from STFT
            axes[1].plot(
                frequencies / 1e6,
                amplitudes,
                color='green',
                label="STFT Spectrum"
            )
            
            if use_gaussian_fit and self.gaussian_fits[i] is not None:
                # Ensure we're using positive frequencies for the fit
                fitted_curve = self.gaussian(
                    frequencies,
                    *self.gaussian_fits[i]
                )
                axes[1].plot(
                    frequencies / 1e6,
                    fitted_curve,
                    color='red',
                    linestyle="--",
                    label="Gaussian Fit"
                )
                axes[1].axvline(
                    self.center_frequencies[i],
                    color='blue',
                    linestyle=":",
                    label=f"Peak @ {self.center_frequencies[i]:.2f} MHz"
                )
            else:
                # Directly find maximum frequency without Gaussian fit
                peak_freq = frequencies[np.argmax(amplitudes)]
                axes[1].axvline(
                    peak_freq / 1e6,
                    color='blue',
                    linestyle=":",
                    label=f"Peak @ {peak_freq/1e6:.2f} MHz"
                )
                self.center_frequencies[i] = peak_freq / 1e6
                self.fitted_sigmas[i] = None
                self.gaussian_fits[i] = None
            
            axes[1].set_title(f"Frequency Spectrum at Depth {self.depth_array_cm[i]:.2f} cm")
            axes[1].set_xlabel("Frequency (MHz)")
            axes[1].set_ylabel("Amplitude")
            axes[1].legend()
            
            plt.tight_layout()
            plt.show()

    ###################################################################################
    # Plot frequency shift
    ###################################################################################
    def plot_frequency_shift(self, show_linear_fit: bool = False):
        """Plot frequency shift results.
        
        Args:
            show_linear_fit (bool): Whether to show the linear fit line. Defaults to False.
        """
        plt.figure(figsize=(10, 6))
        
        # Filter out invalid data points
        valid_indices = [i for i in range(len(self.center_frequencies)) 
                        if self.center_frequencies[i] is not None]
        z_values = self.depth_array_cm[valid_indices]
        center_frequencies = self.center_frequencies[valid_indices]
        
        # Plot observed center frequencies
        plt.scatter(
            z_values,
            center_frequencies,
            label="Observed Center Frequencies",
            color="red",
            marker="o"
        )
        
        # Fit and plot linear model if requested
        if show_linear_fit and len(z_values) > 1:  # Only fit if we have enough points
            slope, intercept = np.polyfit(z_values, center_frequencies, 1)
            linear_fit = slope * z_values + intercept
            
            # Plot linear fit
            plt.plot(
                z_values,
                linear_fit,
                linestyle="--",
                color="blue",
                label=f"Linear Fit (slope: {slope:.3f} MHz/cm)"
            )
        
        plt.xlabel("Depth (cm)")
        plt.ylabel("Center Frequency (MHz)")
        plt.title("Frequency Shift with Depth")
        plt.legend()
        plt.grid(True)
        plt.show()
