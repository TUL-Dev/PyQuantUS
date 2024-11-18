import numpy as np
from typing import Tuple
from numpy.matlib import repmat

NUM_FOURIER_POINTS = 8192

def computeHanningPowerSpec(rfData: np.ndarray, startFrequency: int, endFrequency: int, 
                            samplingFrequency: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the power spectrum of the RF data using a Hanning window.

    Args:
        rfData (np.ndarray): RF data from the ultrasound image.
        startFrequency (int): lower bound of the frequency range (Hz).
        endFrequency (int): upper bound of the frequency range (Hz).
        samplingFrequency (int): sampling frequency of the RF data (Hz).

    Returns:
        Tuple: frequency range and power spectrum.
    """
    # Create Hanning Window Function
    unrmWind = np.hanning(rfData.shape[0])
    windFuncComputations = unrmWind * np.sqrt(len(unrmWind) / sum(np.square(unrmWind)))
    windFunc = repmat(
        windFuncComputations.reshape((rfData.shape[0], 1)), 1, rfData.shape[1]
    )

    # Frequency Range
    frequency = np.linspace(0, samplingFrequency, NUM_FOURIER_POINTS)
    fLow = round(startFrequency * (NUM_FOURIER_POINTS / samplingFrequency))
    fHigh = round(endFrequency * (NUM_FOURIER_POINTS / samplingFrequency))
    freqChop = frequency[fLow:fHigh]

    # Get PS
    fft = np.square(
        abs(np.fft.fft(np.transpose(np.multiply(rfData, windFunc)), NUM_FOURIER_POINTS) * rfData.size)
    )
    fullPS = 20 * np.log10(np.mean(fft, axis=0))

    ps = fullPS[fLow:fHigh]

    return freqChop, ps


def spectralAnalysisDefault6db(npsNormalized: np.ndarray, f: np.ndarray, 
                               db6LowF: int, db6HighF: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Perform spectral analysis on the normalized power spectrum.
    
    Args:
        npsNormalized (np.ndarray): normalized power spectrum.
        f (np.ndarray): frequency array (Hz).
        db6LowF (int): lower bound of the 6dB window (Hz).
        db6HighF (int): upper bound of the 6dB window (Hz).
        
    Returns:
        Tuple: midband fit, frequency range, linear fit, and linear regression coefficients.
    """
    # 1. in one scan / run-through of data file's f array, find the data points on
    # the frequency axis closest to reference file's 6dB window's LOWER bound and UPPER bounds
    smallestDiffDb6LowF = 999999999
    smallestDiffDb6HighF = 999999999

    for i in range(len(f)):
        currentDiffDb6LowF = abs(db6LowF - f[i])
        currentDiffDb6HighF = abs(db6HighF - f[i])

        if currentDiffDb6LowF < smallestDiffDb6LowF:
            smallestDiffDb6LowF = currentDiffDb6LowF
            smallestDiffIndexDb6LowF = i

        if currentDiffDb6HighF < smallestDiffDb6HighF:
            smallestDiffDb6HighF = currentDiffDb6HighF
            smallestDiffIndexDb6HighF = i

    # 2. compute linear regression within the 6dB window
    fBand = f[
        smallestDiffIndexDb6LowF:smallestDiffIndexDb6HighF
    ]  # transpose row vector f in order for it to have same dimensions as column vector nps
    p = np.polyfit(
        fBand, npsNormalized[smallestDiffIndexDb6LowF:smallestDiffIndexDb6HighF], 1
    )
    npsLinfit = np.polyval(p, fBand)  # y_linfit is a column vecotr

    mbfit = p[0] * fBand[round(fBand.shape[0] / 2)] + p[1]

    return mbfit, fBand, npsLinfit, p #, rsqu, ib

def int32torgb(color):
    """Convert int32 to rgb tuple"""
    rgb = []
    for _ in range(3):
        rgb.append(color&0xff)
        color = color >> 8
    return rgb

def condenseArr(image: np.ndarray) -> np.ndarray:
    """Condense (M,N,3) arr to (M,N) with uint32 data to preserve info"""
    assert len(image.shape) == 3
    assert image.shape[-1] == 3
    
    return np.dstack((image,np.zeros(image.shape[:2], 'uint8'))).view('uint32').squeeze(-1)

def expandArr(image: np.ndarray) -> np.ndarray:
    """Inverse of condenseArr"""
    assert len(image.shape) == 2
    
    fullArr = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            fullArr[i,j] = int32torgb(image[i,j])

    return fullArr.astype('uint8')
