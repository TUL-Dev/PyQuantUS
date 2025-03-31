from typing import Any, List

import numpy as np

from pyquantus.parse.objects import ScConfig

class Window:
    """Class to store window data for UTC analysis.
    
    Args:
        ResultsClass (type): Class type to store analysis results.
    """
    def __init__(self, ResultsClass: type):
        self.left = 0
        self.right = 0
        self.top = 0
        self.bottom = 0
        self.results = ResultsClass()

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

class UltrasoundImage:
    """Class to store ultrasound image and RF data.
    """
    def __init__(self):
        self.scBmode: np.ndarray
        self.bmode: np.ndarray
        self.rf: np.ndarray
        self.phantomRf: np.ndarray
        self.axialResRf: float
        self.lateralResRf: float
        self.xmap: np.ndarray
        self.ymap: np.ndarray
        self.scConfig: ScConfig 
