from pathlib import Path

import numpy as np
from tqdm import tqdm
from typing import Tuple
from scipy.io import loadmat
from scipy.signal import hilbert

from pyquantus.parse.objects import DataOutputStruct, InfoStruct
from pyquantus.parse.transforms import scanConvert

def readScan(scanPath: Path, paramPath: Path):
    """Reads the metadata and RF data from a Verasonics .mat file.
    Note this parser assumes it is working with RF data. It also 
    assumes the scan must be scan converted.
    
    Args:
        scanPath (Path): Path to the scan .mat file.
        paramPath (Path): Path to the parameter .mat file.
    
    Returns:
        Tuple[]
    """
    # Read RF metadata
    scanFile = loadmat(scanPath)
    paramFile = loadmat(paramPath)
    
    txBand = paramFile['SeqParam']['Trans'][0][0][0][0][5][0] # MHz
    txFreq = scanFile['param']['TWFrequency'][0][0][0][0] # MHz
    # frameRate = scanFile['param']['Bmode'][0][0][0][0][3][0][0]
    # focus = scanFile['param']['Bmode'][0][0][0][0][1][0][0]*10 # cm
    samplingFreq = scanFile['param']['fs'][0][0][0][0] # Hz
    depth = scanFile['param']['TransRadiusmm'][0][0][0][0]*10 # cm
    # speedOfSound = scanFile['param']['c'][0][0][0][0] # m/s
    theta = scanFile['param']['theta'][0][0][0]
    radii = scanFile['param']['r'][0][0][0]
    endDepth = max(radii) # m (GUESS)
    startDepth = min(radii) # m (GUESS)
    scanWidth = np.rad2deg(abs(max(theta) - min(theta)))
    tilt = 0 # ASSUMED
    
    info = InfoStruct()
    info.minFrequency = txBand[0]*1e6
    info.maxFrequency = txBand[1]*1e6
    info.lowBandFreq = txBand[0]*1e6
    info.upBandFreq = txBand[1]*1e6
    info.centerFrequency = txFreq*1e6
    info.depth = depth
    info.samplingFrequency = samplingFreq
    info.clipFact = 0.95
    info.dynRange = 60
    info.width1 = scanWidth
    info.tilt1 = tilt
    info.startDepth1 = startDepth
    info.endDepth1 = endDepth
    
    # Read RF data and convert to RF
    rfData = scanFile['RFmig']
    bmode = np.zeros(rfData.shape)
    for i in range(rfData.shape[2]):
        bmode[:,:,i] = 20*np.log10(abs(hilbert(rfData[:,:,i], axis=1)))
        
    # Clip and scale B-mode
    clippedMax = info.clipFact*np.amax(bmode)
    bmode = np.clip(bmode, clippedMax-info.dynRange, clippedMax)
    bmode -= np.amin(bmode)
    bmode *= (255/np.amax(bmode))
    
    # Scan convert B-mode
    scBmodeStruct, hCm1, wCm1 = scanConvert(bmode[:,:,0], info.width1, info.tilt1, info.startDepth1, 
                                            info.endDepth1, desiredHeight=500)
    scBmodes = np.array([scanConvert(bmode[:,:,i], info.width1, info.tilt1, info.startDepth1, 
                                    info.endDepth1, desiredHeight=500)[0].scArr for i in tqdm(range(bmode.shape[2]))])

    info.yResRF =  info.endDepth1*1000 / scBmodeStruct.scArr.shape[0] # mm/pix
    info.xResRF = info.yResRF * (scBmodeStruct.scArr.shape[0]/scBmodeStruct.scArr.shape[1]) # placeholder
    info.axialRes = hCm1*10 / scBmodeStruct.scArr.shape[0] # mm/pix
    info.lateralRes = wCm1*10 / scBmodeStruct.scArr.shape[1] # mm/pix
    info.depth = hCm1*10 #mm
    info.width = wCm1*10 #mm
    
    data = DataOutputStruct()
    data.scBmodeStruct = scBmodeStruct
    data.scBmode = scBmodes
    data.bMode = bmode
    data.rf = rfData
    
    data.bMode = np.transpose(bmode, (2, 0, 1))
    data.rf = np.transpose(rfData, (2, 0, 1))
    
    return data, info

def verasonicsRfParser(scanPath: Path, phantomPath: Path, paramPath: Path) \
    -> Tuple[DataOutputStruct, InfoStruct, DataOutputStruct, InfoStruct]:
    """Parses Verasonics RF data and metadata take with two focal zones.
    
    Args:
        filePath (str): Path to the RF data.
        phantomPath (str): Path to the phantom data.
        OmniOn (int): Whether the Omni is on.
    
    Returns:
        Tuple: RF data and metadata for image and phantom.
    """
    imgData, imgInfo = readScan(scanPath, paramPath)
    phantomData, phantomInfo = readScan(phantomPath, paramPath)
    return imgData, imgInfo, phantomData, phantomInfo