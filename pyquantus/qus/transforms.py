import numpy as np
from numpy.matlib import repmat

NUM_FOURIER_POINTS = 8192

def computeHanningPowerSpec(rfData, startFrequency, endFrequency, samplingFrequency):
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


def spectralAnalysisDefault6db(npsNormalized, f, db6LowF, db6HighF):
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

    # # Compute linear regression residuals
    # npsResid = (
    #     npsNormalized[smallestDiffIndexDb6LowF:smallestDiffIndexDb6HighF] - npsLinfit
    # )
    # npsSsResid = sum(np.square(npsResid))
    # npsSsTotal = (len(npsNormalized - 1)) * np.var(npsNormalized)
    # rsqu = 1 - (npsSsResid / npsSsTotal)

    # # Compute spectral parameters
    # ib = 0
    # for i in range(smallestDiffIndexDb6LowF, smallestDiffIndexDb6HighF):
    #     ib += npsNormalized[i] * i

    mbfit = p[0] * fBand[round(fBand.shape[0] / 2)] + p[1]

    return mbfit, fBand, npsLinfit, p #, rsqu, ib

def int32torgb(color):
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
