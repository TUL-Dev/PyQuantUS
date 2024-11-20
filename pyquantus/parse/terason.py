import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert
from typing import Tuple

from pyquantus.parse.objects import DataOutputStruct, InfoStruct

def readFileInfo(path: str) -> Tuple[InfoStruct, float, np.ndarray, np.ndarray]:
    """Reads the metadata and RF data from a Terason .mat file.
    
    Args:
        path (str): Path to the .mat file.
        
    Returns:
        Tuple[InfoStruct, float, np.ndarray, np.ndarray]: Metadata, focal depth, and RF data.
    """
    input = loadmat(path)
    b_data_Zone1 = np.array(input["b_data_Zone1"])
    b_data_Zone2 = np.array(input["b_data_Zone2"])

    Info = InfoStruct()
    Info.minFrequency = 3000000
    Info.maxFrequency = 15000000
    Info.lowBandFreq = 5000000
    Info.upBandFreq = 13000000
    Info.centerFrequency = 9000000 #Hz
    
    Info.lines = input["NumberOfLines"][0][0]
    Info.depth = input["SectorDepthCM"][0][0] * 10 #mm
    Info.samplingFrequency = input["FPS"][0][0] * 1000000 #Hz

    focalDepthZone0 = input["FocalDepthZone0"][0][0]

    return Info, focalDepthZone0, b_data_Zone1, b_data_Zone2

def readFileImg(b_data_Zone1: np.ndarray, b_data_Zone2: np.ndarray, focalDepthZone0: float, 
                OmniOn: int, Info: InfoStruct) -> Tuple[DataOutputStruct, InfoStruct]:
    """Reads the RF data and adds to scan metadata.
    
    Args:
        b_data_Zone1 (np.ndarray): RF data from zone 1.
        b_data_Zone2 (np.ndarray): RF data from zone 2.
        focalDepthZone0 (float): Focal depth of zone 0.
        OmniOn (int): Whether the Omni is on.
        Info (InfoStruct): RF Scan Metadata.
        
    Returns:
        Tuple[DataOutputStruct, InfoStruct]: RF data and metadata.
    """   
    # Blend the zones overlap
    M, N = b_data_Zone1.shape
    if OmniOn == 1:
        b_data1 = (b_data_Zone1[:M, :int(N/2)] + b_data_Zone1[:M, int(N/2):])/2
        b_data2 = (b_data_Zone1[:M, :int(N/2)] + b_data_Zone2[:M, int(N/2):])/2
    else:
        b_data1 = b_data_Zone1[:M, :int(N/2)]
        b_data2 = b_data_Zone2[:M, :int(N/2)]
    
    # Hardcoded vals are specific to Terason images used for a study
    endScanOneIndex = round(focalDepthZone0/(Info.depth/10)*M)

    rfData = np.zeros(b_data1.shape)
    b_data_av = (b_data1[endScanOneIndex:2*endScanOneIndex,:]+b_data2[endScanOneIndex:2*endScanOneIndex,:])/2
    rfData[:endScanOneIndex, :] = b_data1[:endScanOneIndex,:]
    rfData[endScanOneIndex:2*endScanOneIndex,:] = b_data_av
    rfData[2*endScanOneIndex:,:] = b_data2[2*endScanOneIndex:,:]

    bmode = np.zeros(rfData.shape)
    for i in range(rfData.shape[1]):
        bmode[:,i] = 20*np.log10(abs(hilbert(rfData[:,i]))) # type: ignore

    # dynrange of 40
    bmode = np.clip(bmode, np.amax(bmode)-40, np.amax(bmode))
    bmode -= np.amin(bmode)
    bmode *= (255/np.amax(bmode))

    Data = DataOutputStruct()
    Data.rf = rfData
    Data.bMode = bmode
    Data.widthPixels = bmode.shape[1]
    Data.depthPixels = bmode.shape[0]

    Info.width = Info.depth*2
    Info.axialRes = Info.depth/bmode.shape[0]
    Info.lateralRes = Info.width/bmode.shape[1]
    
    return Data, Info


def terasonRfParser(filePath: str, phantomPath: str, OmniOn=1) \
    -> Tuple[DataOutputStruct, InfoStruct, DataOutputStruct, InfoStruct]:
    """Parses Terason RF data and metadata take with two focal zones.
    
    Args:
        filePath (str): Path to the RF data.
        phantomPath (str): Path to the phantom data.
        OmniOn (int): Whether the Omni is on.
    
    Returns:
        Tuple: RF data and metadata for image and phantom.
    """
    # Credit: Steven R. Broadstone, D.Sc., Teratech Corporation dba Terason
    # Parser inspired by MATLAB code writen by Steven
    # Written to parse RF data taken with two focal zones

    imgInfoStruct, focalDepthZone0, b_data_Zone1, b_data_Zone2 = readFileInfo(filePath)
    imgDataStruct, imgInfoStruct = readFileImg(b_data_Zone1, b_data_Zone2, focalDepthZone0, OmniOn, imgInfoStruct)

    refInfoStruct, focalDepthZone0, b_data_Zone1, b_data_Zone2 = readFileInfo(phantomPath)
    refDataStruct, refInfoStruct = readFileImg(b_data_Zone1, b_data_Zone2, focalDepthZone0, OmniOn, refInfoStruct)

    return imgDataStruct, imgInfoStruct, refDataStruct, refInfoStruct