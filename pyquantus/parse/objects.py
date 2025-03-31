from dataclasses import dataclass

import numpy as np

@dataclass
class OutImStruct():
    """Output image structure for scan converted images."""
    scArr: np.ndarray
    xmap: np.ndarray # sc (y,x) --> preSC x
    ymap: np.ndarray # sc (y,x) --> preSC y
        
class DataOutputStruct():
    """Data output structure for general RF/IQ data."""
    def __init__(self):
        self.scBmodeStruct: OutImStruct
        self.scBmode: np.ndarray
        self.rf: np.ndarray
        self.bMode: np.ndarray
        self.widthPixels: int
        self.depthPixels: int
        
@dataclass
class ScConfig:
    """Scan conversion configuration."""
    width: int # deg
    tilt: int
    startDepth: float # mm
    endDepth: float # mm
    numSamplesDrOut: int
        
class InfoStruct():
    """Metadata structure for RF/IQ data."""
    def __init__(self):
        self.maxFrequency: int # transducer freq (Hz)
        self.lowBandFreq: int # analysis freq (Hz)
        self.upBandFreq: int # analysis freq (Hz)
        self.centerFrequency: int #Hz
        self.clipFact: float # %
        self.dynRange: int
        self.samples: int
        self.lines: int
        self.depthOffset: float
        self.depth: float
        self.width: float
        self.samplingFrequency: int
        self.lineDensity: int
        self.lateralRes: float
        self.axialRes: float
        self.yResRF: float
        self.xResRF: float
        self.quad2x: float
        self.numSonoCTAngles: int
        self.numSamplesDrOut: int
        
        # For scan conversion
        self.tilt1: float
        self.width1: float
        self.startDepth1: float
        self.endDepth1: float
        self.endHeight: float
