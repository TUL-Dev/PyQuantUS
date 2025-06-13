from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.signal import firwin, lfilter, hilbert
from scipy.ndimage import correlate

from pyquantus.parse.objects import DataOutputStruct, InfoStruct
from pyquantus.parse.philipsRf import Rfdata, parseRF
from src.Parsers.philipsSipVolumeParser import ScParams, readSIPscVDBParams, scanConvert3dVolumeSeries, formatVolumePix

def QbpFilter(rfData: np.ndarray, Fc1: float, Fc2: float, FiltOrd: int) -> Tuple[np.ndarray, np.ndarray]:
    FiltCoef = firwin(FiltOrd+1, [Fc1*2, Fc2*2], window="hamming", pass_zero="bandpass") # type: ignore
    FiltRfDat = np.transpose(lfilter(np.transpose(FiltCoef),1,np.transpose(rfData)))

    # Do Hilbert Transform on each column
    IqDat = np.zeros(FiltRfDat.shape).astype(np.complex128)
    DbEnvDat = np.zeros(FiltRfDat.shape)
    for i in range(FiltRfDat.shape[1]):
        IqDat[:,i] = hilbert(FiltRfDat[:,i])
        DbEnvDat[:,i] = 20*np.log10(abs(IqDat[:,i])+1)
    
    return IqDat, DbEnvDat

def bandpassFilterEnvLog(rfData: np.ndarray, scParams: ScParams) -> Tuple[np.ndarray, np.ndarray]:
    # Below params are from Philips trial & error
    QbpFiltOrd = 80
    QbpFcA1 = 0.026
    QbpFcA2 = 0.068
    QbpFcB1 = 0.030
    QbpFcB2 = 0.072
    QbpFcC1 = 0.020
    QbpFcC2 = 0.064

    R, M, C = rfData.shape
    rfDat2 = rfData.reshape(R, -1, order='F')
    IqDatA, DbEnvDatA = QbpFilter(rfDat2, QbpFcA1, QbpFcA2, QbpFiltOrd)
    IqDatB, DbEnvDatB = QbpFilter(rfDat2, QbpFcB1, QbpFcB2, QbpFiltOrd)
    IqDatC, DbEnvDatC = QbpFilter(rfDat2, QbpFcC1, QbpFcC2, QbpFiltOrd)
    DbEnvDat = (DbEnvDatA + DbEnvDatB + DbEnvDatC)/3
    QbpDecimFct = int(np.ceil(DbEnvDat.shape[0]/512))
    DbEnvDat = correlate(DbEnvDat, np.ones((QbpDecimFct,1))/QbpDecimFct, mode='nearest')
    DbEnvDat = DbEnvDat[np.arange(0, DbEnvDat.shape[0],QbpDecimFct)]
    NumSamples = DbEnvDat.shape[0]
    NumPlanes = scParams.NUM_PLANES

    # Format RF data to match B-MODE (DbEnvDat)
    formattedRf = rfDat2[np.arange(0, DbEnvDatA.shape[0],QbpDecimFct)]
    rfFullVol = formattedRf[:,:scParams.NumRcvCols*NumPlanes].reshape(NumSamples,scParams.NumRcvCols,NumPlanes, order='F')

    # Keep first full volume
    DbEnvDat_FullVol = DbEnvDat[:,:scParams.NumRcvCols*NumPlanes].reshape(NumSamples,scParams.NumRcvCols,NumPlanes, order='F')
    return DbEnvDat_FullVol, rfFullVol


def sort3DData(dataIn: Rfdata, scParams: ScParams) -> Tuple[np.ndarray, ScParams]:
    dataOut = dataIn.echoData[0]

    # Compute the number of columns and receive beams for use later
    OutML_Azim = dataIn.dbParams.azimuthMultilineFactorXbrOut[0]
    scParams.NumXmtCols = int(max(dataIn.headerInfo.Line_Index))+1
    scParams.NumRcvCols = int(OutML_Azim*scParams.NumXmtCols)
    
    return dataOut, scParams

def getVolume(rfPath: Path, sipNumOutBits: int = 8, DRlowerdB: int = 20, DRupperdB: int = 40):
    scParamFname = f"{rfPath.name[:-3]}_Extras.txt"
    scParamPath = rfPath.parent / Path(scParamFname)

    # #Read in parameter data (primarily for scan conversion)
    scParams = readSIPscVDBParams(scParamPath)
    scParams.pixPerMm=2.5; #for scan conversion grid
    # TODO: implement handling for IQ data (see scParams.removeGapsFlag in Dave Duncan MATLAB code)

    #Read in the interleaved SIP volume data time series (both linear/non-linear parts) 
    rawData = parseRF(f"{rfPath.absolute()}", 0, 2000)

    rfDataArr, scParams = sort3DData(rawData, scParams)

    #Bandpass Filtering + Envelope Det + Log Compression
    dBEnvData_vol, rfVol = bandpassFilterEnvLog(rfDataArr,scParams)

    #Scan Conversion of 3D volume time series (Only doing 1 volume here)
    SC_Vol, bmodeDims = scanConvert3dVolumeSeries(dBEnvData_vol, scParams, scale=False)
    # SC_rfVol, rfDims = scanConvert3dVolumeSeries(rfVol, scParams, normalize=False)

    #Parameters for basic visualization of volume
    slope = (2**sipNumOutBits)/(20*np.log10(2**sipNumOutBits))
    upperLim = slope * DRupperdB
    lowerLim = slope * DRlowerdB

    # Format image for output
    SC_Vol = formatVolumePix(SC_Vol)
    SC_Vol = np.clip(SC_Vol, lowerLim, upperLim)
    SC_Vol = (SC_Vol - lowerLim)/(upperLim - lowerLim) * 255
    bmodeDims = [bmodeDims[2], bmodeDims[0], bmodeDims[1]]
    # rfDims = [rfDims[2], rfDims[0], rfDims[1]]

    Data = DataOutputStruct()
    Data.rf = rfVol
    Data.bMode = SC_Vol
    Data.widthPixels = SC_Vol.shape[2]
    Data.depthPixels = SC_Vol.shape[1]

    Info = InfoStruct()
    Info.minFrequency = 1000000
    Info.maxFrequency = 6000000
    Info.lowBandFreq = 5000000
    Info.upBandFreq = 13000000
    Info.centerFrequency = 9000000 #Hz
    Info.samplingFrequency = 50000000 # TODO: currently a guess
    Info.width = bmodeDims[2]
    Info.depth = bmodeDims[1]
    Info.lateralRes = Info.width/SC_Vol.shape[2] # mm/pix
    Info.axialRes = Info.depth/SC_Vol.shape[1] # mm/pix

    return Data, Info