import numpy as np

class OutImStruct():
    """Output image structure for scan converted images."""
    def __init__(self):
        self.scArr: np.ndarray
        self.xmap: np.ndarray # sc (y,x) --> preSC x
        self.ymap: np.ndarray # sc (y,x) --> preSC y
        
class DataOutputStruct():
    """Data output structure for general RF/IQ data."""
    def __init__(self):
        self.scBmodeStruct: OutImStruct
        self.scBmode: np.ndarray
        self.rf: np.ndarray
        self.bMode: np.ndarray
        self.widthPixels: int
        self.depthPixels: int
        self.coordMap3d: np.ndarray # maps sc (x,y,z) to preSC (x,y,z)
        
class ScConfig:
    """Scan conversion configuration."""
    def __init__(self):
        self.width: int # deg
        self.tilt: int
        self.startDepth: float # mm
        self.endDepth: float # mm
        self.numSamplesDrOut: int
        
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
        
class ScParams():
    def __init__(self):
        self.NUM_PLANES: int
        self.pixPerMm: float
        self.VDB_2D_ECHO_APEX_TO_SKINLINE: float
        self.VDB_2D_ECHO_START_WIDTH_GC: float
        self.VDB_2D_ECHO_STOP_WIDTH_GC: float
        self.VDB_THREED_START_ELEVATION_ACTUAL: float
        self.VDB_THREED_STOP_ELEVATION_ACTUAL: float
        self.VDB_2D_ECHO_STOP_DEPTH_SIP: float
        self.VDB_2D_ECHO_START_DEPTH_SIP: float
        self.VDB_2D_ECHO_SLACK_TIME_MM: float
        self.VDB_THREED_RT_VOLUME_RATE: float

        self.NumXmtCols: int
        self.NumRcvCols: int