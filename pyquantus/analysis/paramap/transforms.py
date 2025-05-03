import numpy as np
from typing import Tuple
from numpy.matlib import repmat

NUM_FOURIER_POINTS = 8192

def compute_hanning_power_spec(rf_data: np.ndarray, start_frequency: int, end_frequency: int, 
                            sampling_frequency: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the power spectrum of the RF data using a Hanning window.

    Args:
        rf_data (np.ndarray): RF data from the ultrasound image (n lines x m samples).
        start_frequency (int): lower bound of the frequency range (Hz).
        end_frequency (int): upper bound of the frequency range (Hz).
        sampling_frequency (int): sampling frequency of the RF data (Hz).

    Returns:
        Tuple: frequency range and power spectrum.
    """
    # Create Hanning Window Function
    unrm_wind = np.hanning(rf_data.shape[0])
    wind_func_computations = unrm_wind * np.sqrt(len(unrm_wind) / sum(np.square(unrm_wind)))
    wind_func = repmat(
        wind_func_computations.reshape((rf_data.shape[0], 1)), 1, rf_data.shape[1]
    )

    # Frequency Range
    frequency = np.linspace(0, sampling_frequency, NUM_FOURIER_POINTS)
    f_low = round(start_frequency * (NUM_FOURIER_POINTS / sampling_frequency))
    f_high = round(end_frequency * (NUM_FOURIER_POINTS / sampling_frequency))
    freq_chop = frequency[f_low:f_high]

    # Get PS
    fft = np.square(
        abs(np.fft.fft(np.transpose(np.multiply(rf_data, wind_func)), NUM_FOURIER_POINTS) * rf_data.size)
    )
    full_ps = np.mean(fft, axis=0)

    ps = full_ps[f_low:f_high]

    return freq_chop, ps
