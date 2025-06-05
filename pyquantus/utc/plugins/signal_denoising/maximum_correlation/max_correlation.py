###################################################################################
# attenuation coefficient estimation
###################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

class MaximumCorrelationDenoising:
    """Class for computing the attenuation coefficient based on central frequency shift for the ROI.
    
    This class implements a signal denoising method using STFT-based correlation analysis and ISTFT reconstruction.
    """
    
    ###################################################################################
    # Constructor
    ###################################################################################
    def __init__(self, 
                 speed_of_sound: float = 1540,
                 window_type: str = 'hann',
                 nperseg: int = 32,
                 noverlap: int = 16,
                 lower_threshold: int = 20):
        """
        Initialize the MaximumCorrelationDenoising class.
        
        Args:
            speed_of_sound (float, optional): Speed of sound in m/s. Defaults to 1540.
            window_type (str, optional): Window function for STFT. Defaults to 'hann'.
            nperseg (int, optional): Number of points per segment for STFT. Defaults to 32.
            noverlap (int, optional): Number of points to overlap between segments. Defaults to 16.
            lower_threshold (int, optional): Minimum number of points for correlation analysis. Defaults to 20.
        """
        self.speed_of_sound = speed_of_sound
        self.window_type = window_type
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.lower_threshold = lower_threshold

    ###################################################################################
    # Plotting
    ###################################################################################
    def plot_frequency_signal(self, depths_cm, log_amplitude, stored_signal, max_corr_segment, best_slope,
                            freq, correlation_values, segment_sizes, lower_threshold, max_corr_size):
        """Plots signal amplitude and correlation analysis for a specific frequency."""
        plt.figure(figsize=(10, 4))
        plt.plot(depths_cm, log_amplitude, label=f'{freq:.2f} MHz Signal', color='blue')
        plt.xlabel("Depth (cm)")
        plt.ylabel("Log Amplitude")
        plt.title(f"Frequency-Specific Signal ({freq:.2f} MHz)")
        plt.grid()
        
        if max_corr_segment:
            X_max, y_max, y_pred_max = max_corr_segment
            plt.plot(X_max, y_pred_max, label="Best Fit Regression", color='red', linestyle='dashed')
            plt.text(0.05, 0.85, f"Slope: {best_slope:.4f}", transform=plt.gca().transAxes,
                    fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='red'))
        
        plt.legend()
        plt.show()
        
        # Plot correlation coefficient evolution
        plt.figure(figsize=(10, 4))
        plt.plot(segment_sizes, correlation_values, label=f"Correlation for {freq:.2f} MHz", color='red')
        plt.axvline(x=lower_threshold, color='black', linestyle='dotted', label=f"Lower Limit ({lower_threshold} points)")
        if max_corr_size:
            plt.axvline(x=max_corr_size, color='green', linestyle='dashed', label="Best Correlation Segment")
        
        plt.xlabel("Number of Data Points")
        plt.ylabel("Correlation Coefficient (R)")
        plt.title(f"Correlation Evolution ({freq:.2f} MHz)")
        plt.legend()
        plt.grid()
        plt.show()
        
        # Plot stored signal up to max correlation segment
        if stored_signal is not None and len(stored_signal) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(depths_cm, stored_signal, label=f"Stored Signal ({freq:.2f} MHz)", color='green')
            plt.xlabel("Depth (cm)")
            plt.ylabel("Log Amplitude")
            plt.title(f"Stored Signal Up to Best Correlation ({freq:.2f} MHz)")
            plt.legend()
            plt.grid()
            plt.show()

    ###################################################################################
    # Denoising
    ###################################################################################
    def denoise_signal(self, signal, sampling_rate_MHz, trimmed_time_1d, visualize=False):
        """
        Denoises a signal using STFT-based correlation analysis and ISTFT reconstruction.
        
        Args:
            signal (np.ndarray): Input signal to denoise
            sampling_rate_MHz (float): Sampling rate in MHz
            trimmed_time_1d (np.ndarray): Time array for the signal
            visualize (bool, optional): Whether to show visualization plots. Defaults to False.
            
        Returns:
            np.ndarray: Denoised signal
        """
        fs = sampling_rate_MHz * 1e6  # Convert sampling rate to Hz
        
        # Compute STFT for the signal
        frequencies, times, stft_matrix = stft(signal, fs=fs, window=self.window_type, 
                                             nperseg=self.nperseg, noverlap=self.noverlap)
        
        frequencies_MHz = frequencies / 1e6  # Convert to MHz
        depths_cm = (self.speed_of_sound * np.linspace(trimmed_time_1d[0], trimmed_time_1d[-1], len(times))) / 2 * 100
        slopes = []
        
        # Initialize denoised STFT matrix
        denoised_stft_matrix = np.zeros_like(stft_matrix, dtype=np.complex128)

        for idx, freq in enumerate(frequencies_MHz):
            freq_amplitude = np.abs(stft_matrix[idx])
            phase = np.angle(stft_matrix[idx])  # Preserve phase
            
            amplitude_log = np.log10(freq_amplitude + 1)
            segment_sizes = np.arange(2, len(depths_cm) + 1)
            correlation_values = np.full_like(segment_sizes, np.nan, dtype=np.float64)
            
            max_corr, best_slope, max_corr_segment, max_corr_size = None, None, None, None
            denoised_amplitude_log = np.zeros_like(amplitude_log)  # Initialize with zeros
            
            for i in segment_sizes:
                X = depths_cm[:i].reshape(-1, 1)
                y = amplitude_log[:i]
                
                if np.std(y) == 0:
                    continue  # Avoid division by zero
                
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                corr, _ = pearsonr(y, y_pred)
                correlation_values[i - 2] = corr
                
                if i >= self.lower_threshold and (max_corr is None or abs(corr) > abs(max_corr)):
                    max_corr, best_slope = corr, -model.coef_[0]
                    max_corr_segment = (X, y, y_pred)
                    max_corr_size = i
                    denoised_amplitude_log[:i] = amplitude_log[:i]  # Store up to this point
            
            slopes.append(best_slope if best_slope is not None else 0)
            
            # Convert log amplitude back to normal scale and restore phase
            denoised_amplitude = 10**denoised_amplitude_log - 1
            denoised_stft_matrix[idx] = denoised_amplitude * np.exp(1j * phase)  # Restore phase information

            # Keep the visualization for each frequency
            if visualize:
                self.plot_frequency_signal(depths_cm, amplitude_log, denoised_amplitude_log, max_corr_segment, best_slope,
                                      freq, correlation_values, segment_sizes, self.lower_threshold, max_corr_size)

        # Reconstruct the time-domain signal using ISTFT
        _, reconstructed_signal = istft(denoised_stft_matrix, fs=fs, window=self.window_type, 
                                      nperseg=self.nperseg, noverlap=self.noverlap)
        
        if visualize:
            plt.figure(figsize=(10, 4))
            plt.plot(signal, label='Original Signal', color='blue')
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            plt.title("Original Time-Domain Signal")
            plt.legend()
            plt.grid()
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.plot(reconstructed_signal, label='Reconstructed Signal', color='purple')
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            plt.title("Reconstructed Time-Domain Signal using ISTFT")
            plt.legend()
            plt.grid()
            plt.show()

        return reconstructed_signal
