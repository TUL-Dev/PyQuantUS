import numpy as np

from .._obj.objects import OutImStruct
        
class DataOutputStruct():
    """Data output structure for general RF/IQ data."""
    def __init__(self):
        self.scBmodeStruct: OutImStruct
        self.scBmode: np.ndarray
        self.rf: np.ndarray
        self.bMode: np.ndarray
        self.widthPixels: int
        self.depthPixels: int
        
class InfoStruct():
    """Metadata structure for RF/IQ data."""
    def __init__(self):
        self.minFrequency: int # transducer freq (Hz)
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
