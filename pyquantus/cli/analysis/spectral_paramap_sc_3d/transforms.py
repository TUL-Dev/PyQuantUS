import numpy as np
from typing import Tuple
from numpy.matlib import repmat

NUM_FOURIER_POINTS = 8192

def map_1d_to_3d(coord, xDim, yDim, zDim):
    x = coord // (zDim * yDim)
    coord -= x * (zDim * yDim)
    y = coord // zDim
    z = coord - (y * zDim)
    
    return x, y, z

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


def compute_spectral_params(nps: np.ndarray, f: np.ndarray, 
                               low_f: int, high_f: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Perform spectral analysis on the normalized power spectrum.
    source: Lizzi et al. https://doi.org/10.1016/j.ultrasmedbio.2006.09.002
    
    Args:
        nps (np.ndarray): normalized power spectrum.
        f (np.ndarray): frequency array (Hz).
        low_f (int): lower bound of the frequency window for analysis (Hz).
        high_f (int): upper bound of the frequency window for analysis (Hz).
        
    Returns:
        Tuple: midband fit, frequency range, linear fit, and linear regression coefficients.
    """
    # 1. in one scan / run-through of data file's f array, find the data points on
    # the frequency axis closest to reference file's analysis window's LOWER bound and UPPER bounds
    smallest_diff_low_f = 999999999
    smallest_diff_high_f = 999999999

    for i in range(len(f)):
        current_diff_low_f = abs(low_f - f[i])
        current_diff_high_f = abs(high_f - f[i])

        if current_diff_low_f < smallest_diff_low_f:
            smallest_diff_low_f = current_diff_low_f
            smallest_diff_index_low_f = i

        if current_diff_high_f < smallest_diff_high_f:
            smallest_diff_high_f = current_diff_high_f
            smallest_diff_index_high_f = i

    # 2. compute linear regression within the analysis window
    f
    f_band = f[
        smallest_diff_index_low_f:smallest_diff_index_high_f
    ]  # transpose row vector f in order for it to have same dimensions as column vector nps
    p = np.polyfit(
        f_band, nps[smallest_diff_index_low_f:smallest_diff_index_high_f], 1
    )
    nps_linfit = np.polyval(p, f_band)  # y_linfit is a column vecotr

    mbfit = p[0] * f_band[round(f_band.shape[0] / 2)] + p[1]

    return mbfit, f_band, nps_linfit, p
