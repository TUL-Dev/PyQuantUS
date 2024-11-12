import struct

import numpy as np
from scipy.signal import hilbert

from pyquantus.parse.objects import DataOutputStruct, InfoStruct
from pyquantus.parse.transforms import scanConvert, iqToRf

def findPreset(filename):
    headersize = 16

    file_obj = open(filename, 'rb')
    hdr = [int.from_bytes(file_obj.read(2), byteorder='little', signed=False) for i in range(2)]
    numAcquiredRxBeams = hdr[0]

    hdr = [int.from_bytes(file_obj.read(2), byteorder='little', signed=False) for i in range(2)]
    numParallelAcquisitions = hdr[1]

    hdr = [int.from_bytes(file_obj.read(2), byteorder='little', signed=False) for i in range(2)]
    numSamplesDrOut = hdr[0]
    file_obj.close()

    return numSamplesDrOut

def readIQ(filename):
    headersize = 16

    file_obj = open(filename, 'rb')
    hdr = [int.from_bytes(file_obj.read(2), byteorder='little', signed=False) for i in range(2)]
    numAcquiredRxBeams = hdr[0]

    hdr = [int.from_bytes(file_obj.read(2), byteorder='little', signed=False) for i in range(2)]
    numParallelAcquisitions = hdr[1]

    hdr = [int.from_bytes(file_obj.read(2), byteorder='little', signed=False) for i in range(2)]
    numSamplesDrOut = hdr[0]
    numSamplesRbfOut = hdr[1]

    hdr = [int.from_bytes(file_obj.read(1), byteorder='little', signed=False) for i in range(4)]
    isPhaseInvertEn = hdr[0]

    hdr = [struct.unpack('f', file_obj.read(4))[0] for i in range(5)]
    decimationFactor = hdr[0]
    rbfDecimationFactor = hdr[1]
    rbfBeMixerFrequency = hdr[2]
    propagationVelCmPerSec = hdr[3]
    digitizingRateHz = hdr[4] 

    # read IQ data
    file_obj.seek(0)
    numSamplesIQAcq = numSamplesDrOut*2
    dataA = np.zeros((numSamplesDrOut*2, numAcquiredRxBeams))
    dataB = np.zeros((numSamplesDrOut*2, numAcquiredRxBeams))
    allData = np.zeros((numSamplesDrOut*2, numAcquiredRxBeams*(1+isPhaseInvertEn)))

    # IQ Acquisition, following parameter always zero
    isPhaseInvertEn = 0
    isRxFreqCompoundEn = 0
    isDiffplusEn = 0

    for ii in range(int(numAcquiredRxBeams/numParallelAcquisitions)):
        for jj in range(numParallelAcquisitions):
            hdr = [int.from_bytes(file_obj.read(4), byteorder='little', signed=False) for i in range(headersize)]
            allData[:headersize, (ii*numParallelAcquisitions) + jj] = hdr

            dat = np.array([int.from_bytes(file_obj.read(4), byteorder='little', signed=False) for i in range(numSamplesIQAcq)])
            dat[dat >= (2**23)] -= (2**24)

            dataA[:,ii*numParallelAcquisitions+jj] = dat[:numSamplesDrOut*2]

            allData[headersize:headersize+(numSamplesDrOut*2)+1, \
                    ii*numParallelAcquisitions+jj] = dat[:min(numSamplesDrOut*2, allData.shape[0]-headersize)]
            
            

    iq = dataA[np.arange(0, numSamplesDrOut*2, 2)] + 1j*dataA[np.arange(1,numSamplesDrOut*2,2)]
    bmode = 20*np.log10(abs(iq))

    return bmode, iq, digitizingRateHz, numSamplesDrOut, decimationFactor

def canonIqParser(imgPath: str, refPath: str):
    imgInfo, refInfo, imgData, refData = getData(imgPath, refPath)
    return imgData, imgInfo, refData, refInfo

def getData(imgPath: str, refPath: str):
    imgInfo = readFileInfo()
    imgData, imgInfo = readFileImg(imgInfo, imgPath)

    refInfo = readFileInfo()
    refData, refInfo = readFileImg(refInfo, refPath)

    return imgInfo, refInfo, imgData, refData

def readFileInfo():    
    Info = InfoStruct()
    Info.minFrequency = 0
    Info.maxFrequency = 8000000
    Info.lowBandFreq = 3000000 #Hz
    Info.upBandFreq = 5000000 #Hz
    Info.centerFrequency = 4000000 #Hz

    # Scan Convert Settings
    Info.tilt1 = 0
    Info.width1 = 70 #degrees

    return Info

def readFileImg(Info: InfoStruct, filePath: str):
    bmode, iqData, Info.samplingFrequency, Info.numSamplesDrOut, decimationFactor = readIQ(filePath)
    if Info.numSamplesDrOut == 1400: #Preset 1
        Info.depth = 150 #mm
        # print("Preset 1 found!")
    elif Info.numSamplesDrOut == 1496: #Preset 2
        Info.depth = 200 #mm
        # print("Preset 2 found!")
    else:
        print("ERROR: No preset found!")
        exit()

    rfData = iqToRf(iqData, Info.samplingFrequency, decimationFactor, Info.centerFrequency)
    bmode = np.zeros(rfData.shape)
    for i in range(rfData.shape[1]):
        bmode[:,i] = 20*np.log10(abs(hilbert(rfData[:,i])))           

    Info.endDepth1 = Info.depth/1000 #m
    Info.startDepth1 = Info.endDepth1/4 #m

    scBmodeStruct, hCm1, wCm1 = scanConvert(bmode, Info.width1, Info.tilt1, Info.startDepth1, Info.endDepth1)
    Info.depth = hCm1*10 #mm
    Info.width = wCm1*10 #mm
    Info.lateralRes = Info.width/scBmodeStruct.scArr.shape[1]
    Info.axialRes = Info.depth/scBmodeStruct.scArr.shape[0]

    Data = DataOutputStruct()
    Data.scBmodeStruct = scBmodeStruct
    Data.scBmode = scBmodeStruct.scArr * (255/np.amax(scBmodeStruct.scArr))
    Data.rf = rfData
    Data.bMode = bmode * (255/np.amax(bmode))

    return Data, Info
