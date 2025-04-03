from typing import List

import numpy as np

class SpectralResults:
    """Class to store analysis results from UTC analysis of an image.
    """
    def __init__(self):
        self.mbf: float  # midband fit (dB)
        self.ss: float  # spectral slope (dB/MHz)
        self.si: float  # spectral intercept (dB)
        self.nps: np.ndarray  # normalized power spectrum
        self.ps: np.ndarray  # image power spectrum
        self.rPs: np.ndarray  # phantom power spectrum
        self.f: np.ndarray  # frequency array (Hz)

class Window:
    """Class to store window data for UTC analysis.
    """
    def __init__(self):
        self.left: int 
        self.right: int 
        self.top: int 
        self.bottom: int 
        self.results = SpectralResults()
        
class Window3d:
    """Class to store window data for 3D UTC analysis.
    """
    def __init__(self):
        self.axMin: int
        self.axMax: int
        self.latMin: int
        self.latMax: int
        self.corMin: int
        self.corMax: int
        self.results = SpectralResults()

class AnalysisConfig:
    """Class to store configuration data for UTC analysis.
    """
    def __init__(self):
        self.transducerFreqBand: List[int]  # [min, max] (Hz)
        self.analysisFreqBand: List[int]  # [lower, upper] (Hz)
        self.samplingFrequency: int  # Hz
        self.axWinSize: float  # axial length per window (mm)
        self.latWinSize: float  # lateral width per window (mm)
        self.windowThresh: float  # % of window area required to be in ROI
        self.axialOverlap: float  # % of ax window length to move before next window
        self.lateralOverlap: float  # % of lat window length to move before next window
        self.centerFrequency: float  # Hz
        
class AnalysisConfig3d(AnalysisConfig):
    """Class to store configuration data for 3D UTC analysis.
    """
    def __init__(self):
        super().__init__()
        self.corWinSize: float  # coronal height per window (mm)
        self.coronalOverlap: float  # % of cor window height to move before next window

class UltrasoundImage:
    """Class to store ultrasound image and RF data.
    """
    def __init__(self):
        self.scBmode: np.ndarray # rgb
        self.bmode: np.ndarray # rgb
        self.rf: np.ndarray
        self.phantomRf: np.ndarray
        self.axialResRf: float # mm/pix
        self.lateralResRf: float # mm/pix
        self.xmap: np.ndarray # maps (y,x) in SC coords to x preSC coord
        self.ymap: np.ndarray # maps (y,x) in SC coords to y preSC coord

class UltrasoundImage3d:
    """Class to store ultrasound image and RF data.
    """
    def __init__(self):
        self.scBmode: np.ndarray # rgb
        self.bmode: np.ndarray # rgb
        self.rf: np.ndarray
        self.phantomRf: np.ndarray
        self.axialResRf: float # mm/pix
        self.lateralResRf: float # mm/pix
        self.coronalResRf: float # mm/pix
        self.coordMap3d: np.ndarray # maps (z,y,x) in SC coords to (x,y) preSC coord