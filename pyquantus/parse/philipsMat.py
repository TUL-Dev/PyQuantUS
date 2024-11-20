from pathlib import Path
from typing import Tuple

from scipy.io import loadmat
from scipy.signal import hilbert
import numpy as np

from pyquantus.parse.objects import DataOutputStruct, InfoStruct
from pyquantus.parse.transforms import scanConvert


def philips2dRfMatParser(filepath: str, refpath: str, frame: int) \
        -> Tuple[DataOutputStruct, InfoStruct, DataOutputStruct, InfoStruct]:
    """Parse Philips 2D RF data from a .mat file.
    
    Args:
        filepath (str): The file path of the Philips 2D RF data.
        refpath (str): The file path of the reference data.
        frame (int): The frame number.
        
    Returns:
        Tuple: The image data, image metadata, reference data, and reference metadata.
    """

    input = loadmat(filepath.__str__())
    ImgInfo = readFileInfo()
    ImgData, ImgInfo = readFileImg(ImgInfo, frame, input)

    input = loadmat(refpath.__str__())
    RefInfo = readFileInfo()
    RefData, RefInfo = readFileImg(RefInfo, frame, input)

    return ImgData, ImgInfo, RefData, RefInfo

def readFileInfo() -> InfoStruct:  
    """Read default metadata values for a Philips 2D RF data.""" 
    Info = InfoStruct()
    Info.minFrequency = 2000000
    Info.maxFrequency = 8000000
    Info.lowBandFreq = 3000000
    Info.upBandFreq = 7000000
    Info.centerFrequency = 9000000 #Hz
    Info.depthOffset = 0.04 # probeStruct.transmitoffset
    Info.depth = 0.16 #m
    Info.width = 70 #deg
    Info.samplingFrequency = 20000000

    # Scan Conversion
    Info.tilt1 = 0
    Info.width1 = 70
    Info.startDepth1 = 0.04
    Info.endDepth1 = 0.16
    Info.endHeight = 500
    Info.clipFact = 0.95
    Info.dynRange = 255
    
    Info.yResRF = 1
    Info.xResRF = 1
    Info.quad2x = 1

    return Info

def readFileImg(Info: InfoStruct, frame: int, input) -> Tuple[DataOutputStruct, InfoStruct]:
    """Read Philips 2D RF data, parse it, and complete scan conversion.
    
    Args:
        Info (InfoStruct): Philips 2D RF data metadata.
        frame (int): The frame number.
        input: The .mat file data.
        
    Returns:
        Tuple: The image data and image metadata.
    """
    echoData = input["rf_data_all_fund"]# indexing works by frame, angle, image
    while not(len(echoData[0].shape) > 1 and echoData[0].shape[0]>40 and echoData[0].shape[1]>40):
        echoData = echoData[0]
    echoData = np.array(echoData[frame]).astype(np.int32)

    bmode = np.zeros(echoData.shape).astype(np.int32)

    # Do Hilbert Transform on each column
    for i in range(echoData.shape[1]):
        bmode[:,i] = 20*np.log10(abs(hilbert(echoData[:,i]))) # type: ignore

    ModeIM = echoData

    scBmodeStruct, hCm1, wCm1 = scanConvert(bmode, Info.width1, Info.tilt1, Info.startDepth1, Info.endDepth1, Info.endHeight.__int__())
    Info.depth = hCm1
    Info.width = wCm1
    Info.lateralRes = wCm1*10/scBmodeStruct.scArr.shape[1]
    Info.axialRes = hCm1*10/scBmodeStruct.scArr.shape[0]

    Data = DataOutputStruct()
    Data.scBmodeStruct = scBmodeStruct
    Data.scBmode = scBmodeStruct.scArr
    Data.rf = ModeIM
    Data.bMode = bmode
    
    clippedMax = Info.clipFact*np.amax(Data.scBmode)
    scBmode = np.clip(Data.scBmode, clippedMax-Info.dynRange, clippedMax) * (255/clippedMax)
    Data.scBmodeStruct.scArr = scBmode
    Data.scBmode = scBmode
        
    clippedMax = Info.clipFact*np.amax(Data.bMode)
    bmode = np.clip(Data.bMode, clippedMax-Info.dynRange, clippedMax) * (255/clippedMax)
    Data.bMode = bmode

    return Data, Info

# if __name__ == "__main__":
    # getImage('FatQuantData1.mat', '/Users/davidspector/Documents/MATLAB/Data/', 'FatQuantPhantom1.mat', '/Users/davidspector/Documents/MATLAB/Data/', 0)