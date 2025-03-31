from dataclasses import dataclass
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

@dataclass
class AnalysisConfig:
    """Class to store configuration data for UTC analysis.
    """
    transducerFreqBand: List[int]  # [min, max] (Hz)
    analysisFreqBand: List[int]  # [lower, upper] (Hz)
    samplingFrequency: int  # Hz
    axWinSize: float  # axial length per window (mm)
    latWinSize: float  # lateral width per window (mm)
    windowThresh: float  # % of window area required to be in ROI
    axialOverlap: float  # % of ax window length to move before next window
    lateralOverlap: float  # % of lat window length to move before next window
    centerFrequency: float  # Hz

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
