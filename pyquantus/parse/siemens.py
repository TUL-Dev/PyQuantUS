import struct
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from pyquantus.parse.objects import DataOutputStruct, InfoStruct

class TimestampStruct():
    def __init__(self):
        self.year = None
        self.month = None
        self.dayOfWeek = None
        self.day = None
        self.hour = None
        self.minute = None
        self.second = None
        self.millisecond = None

class Frh0Struct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.frameTimeStamp = None
        self.isTriggeredFrame = None
        self.futureParameter7 = None
        self.futureParameter8 = None

class RfsdStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.isDynamicFocusEn = None
        self.rxFNum = None
        self.rxFocusRangeCm = None
        self.isAperGrowthEn = None
        self.rxApodization = None
        self.txFocusRangeCm = None
        self.txFNum = None
        self.txFrequencyMhz = None
        self.numTxCycles = None
        self.txWaveformStyle = None
        self.prfHz = None
        self.txApodization = None
        self.pulseAmplitude = None
        self.analogGain = None

class RfcoStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.acousticFrameRateHz = None
        self.numParallelAcquisitions = None
        self.fovShape = None
        self.apexLateralCm = None
        self.apexVerticalCm = None
        self.displayedLateralMin = None
        self.displayedLateralSpan = None
        self.displayedAxialMinCm = None
        self.displayedAxialSpanCm = None
        self.steeringAngleRad = None
        self.lineDensity = None
        self.interleaveFactor = None
        self.alternateInterleaveFactor = None
        self.isFrameInterleaveEn = None
        self.ensembleSize = None

class RfdoStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.fovShape = None
        self.gateStartCm = None
        self.gateSizeCm = None
        self.apexLateralCm = None
        self.apexVerticalCm = None
        self.roiLateralMin = None
        self.steeringAngleRad = None
        self.sampleVolumeWidth = None
        self.dopplerType = None

class RfmmStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.fovShape = None
        self.displayedAxialMinCm = None
        self.displayedAxialSpanCm = None
        self.apexLateralCm = None
        self.apexVerticalCm = None
        self.roiLateralMin = None
        self.steeringAngleRad = None

class RfbmStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.numFocalZones = None
        self.acousticFrameRateHz = None
        self.numParallelAcquisitions = None
        self.fovShape = None
        self.apexLateralCm = None
        self.apexVerticalCm = None
        self.displayedLateralMin = None
        self.displayedLateralSpan = None
        self.displayedAxialMinCm = None
        self.displayedAxialSpanCm = None
        self.steeringAngleRad = None
        self.lineDensity = None
        self.phaseInvertMode = None

class RfsiStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.notes = None
        self.script = None

class RfamStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.examIndex = None
        self.probeName = None
        self.probeRadiusCm = None
        self.isTrigger10n = None
        self.isTrigger20n = None
        self.trigger1DelaySec = None
        self.trigger2DelaySec = None
        self.trigger1WaveCount = None
        self.trigger2WaveCount = None
        self.futureParameter1 = None
        self.futureParameter2 = None
        self.futureParameter3 = None
        self.futureParameter4 = None
        self.futureParameter5 = None
        self.futureParameter6 = None

class StriStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.startStreamTimestamp = None
        self.endStreamTimestamp = None
        self.streamType = None

class Db00Struct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.startOfFrameData = None

class FrameStruct():
    def __init__(self):
        self.ckid = None
        self.dwFlags = None
        self.chunkOffset = None
        self.chunkLength = None

class Idx1Struct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.frame = []
        self.startOffset = None

class RfbdStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.numSamplesPerVector = None
        self.numVectorsPerFrame = None
        self.mode = None
        self.set = None
        self.positionX = None
        self.positionZ = None
        self.thetaRad = None

class StrfStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.samplingRate = None
        self.bitCountPerSample = None
        self.sampleMask = None
        self.vectorHeaderLengthBytes = None
        self.compression = None

class StrhStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.fccType = None
        self.fccHandler = None
        self.dwFlags = None
        self.dwPriority = None
        self.dwInitialFrames = None
        self.dwScale = None
        self.dwRate = None
        self.dwStart = None
        self.dwLength = None
        self.dwSuggestedBufferSize = None
        self.quality = None
        self.dwSampleSize = None
        self.rcFrame_bottom = None
        self.rcFrame_left = None
        self.rcFrame_right = None
        self.rcFrame_top = None

class Csh0Struct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.frameClockAtStartStream = None
        self.requestForDataTimeStamp = None
        self.numCustomStreamHeaders = None
        self.numCustomFrameHeaders = None
        self.numFramesInStream = None
        self.numVectorsPerStreamFrame = None

class Cfh0Struct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.sequencingMode = None
        self.userName = None
        self.versionURI = None
        self.versionURIFileFormat = None
        self.versionRtcFirmware = None
        self.versionSipFirmware = None
        self.versionPciFirmware = None
        self.suppressPatientID = None

class SffmStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.sffVersion = None
        self.platformName = None
        self.platformVersion = None
        self.operatingSystem = None
        self.operatingSystemVersion = None
        self.cpu = None

class AvihStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.dwMicroSecPerFrame = None
        self.dwMaxBytesPerSec = None
        self.resesrved1 = None
        self.dwFlags = None
        self.dwTotalFrames = None
        self.dwInitialFrames = None
        self.dwStreams = None
        self.dwSuggestedBufferSize = None
        self.dwWidth = None
        self.dwHeight = None
        self.dwReserved = None

class RfgiStruct():
    def __init__(self):
        self.tagName = None
        self.dataSize = None
        self.headerVersion = None
        self.rfAcqGain = None
        self.minDataVectorRange = None
        self.numFramesAcquired = None
        self.rfAxialMinCm = None
        self.rfAxialSpanCm = None
        self.isBmodeInfoFrameDep = None
        self.isMmodeInfoFrameDep = None
        self.isColorInfoFrameDep = None
        self.isDopplerInfoFrameDep = None
        self.isSetDepInfoFrameDep = None
        self.isBeamDepInfoFrameDep = None

class FileHeaderStruct():
    def __init__(self):
        self.rfgi = None
        self.idx1 = None
        self.rfbd = None
        self.avih = None
        self.sffm = None
        self.strh = None
        self.cfh0 = None
        self.csh0 = None
        self.stri = None
        self.rfam = None
        self.rfsi = None
        self.rfbm = None
        self.rfmm = None
        self.rfdo = None
        self.rfco = None
        self.rfsd = None
        self.strf = None
        self.frh0 = None


def siemensParser(imgPath: str, phantomPath: str):

    imgInfo, refInfo, imgData, refData = getData(imgPath, phantomPath)

    clippedMax = imgInfo.clipFact*np.amax(imgData.bMode)
    bmode = np.clip(imgData.bMode, clippedMax-imgInfo.dynRange, clippedMax) * (255/clippedMax)

    return bmode, imgData, imgInfo, refData, refInfo

def getData(imgPath: str, phantomPath: str):
    focus = 0 # hard-coded for now
    imgInfo = readFileInfo(imgPath)
    [imgData, imgInfo] = readFileImg(imgInfo, focus, imgPath)

    refInfo = readFileInfo(phantomPath)
    [refData, refInfo] = readFileImg(refInfo, focus, phantomPath)
    
    return imgInfo, refInfo, imgData, refData

def readFileImg(Info: InfoStruct, focus, imgPath: str):
    file_obj = open(imgPath, 'rb')
    FileHeader = readHeader(imgPath)
    if Path(imgPath).name == 'uri_SpV2232_VpF512_FpA90_20210129103529.rfd':
        # Custom phantom optimization for Thyroid project
        data, _ = extractFrameData(file_obj, FileHeader, 51)
        data = data.reshape((1, data.shape[0], data.shape[1]))
    else:
        data = np.zeros([len(FileHeader.idx1.frame), int(FileHeader.rfbd.numSamplesPerVector-FileHeader.strf.vectorHeaderLengthBytes/2), \
                        int(FileHeader.csh0.numVectorsPerStreamFrame)]).astype(np.int16)
        for frame in range(len(FileHeader.idx1.frame)):
            [data[frame], _]  = extractFrameData(file_obj, FileHeader, frame)
    echoData = splitData(data.astype(np.double), focus)
    
    bmode = np.zeros(echoData.shape).astype(np.int32)

    # Do Hilbert Transform on each column
    for frame in range(echoData.shape[0]):
        for i in range(echoData.shape[2]):
            bmode[frame,:,i] = 20*np.log10(abs(hilbert(echoData[frame,:,i])))

    modeIM = echoData
    Info.axialRes = (1/(Info.samplingFrequency*2))*1540 # speed of sound in tissue
    Info.axialRes *= 1000 # m -> mm
    Info.lateralRes = 10/Info.lineDensity
    Info.width = Info.lateralRes*bmode.shape[2] # mm
    Info.depth = Info.axialRes*bmode.shape[1] # mm
    Info.clipFact = 0.95
    Info.dynRange = 55

    Data = DataOutputStruct()
    Data.rf = modeIM
    Data.bMode = bmode
    Data.widthPixels = bmode.shape[2]
    Data.depthPixels = bmode.shape[1]
    return Data, Info


def splitData(imgData, focus):
    newData = np.zeros((imgData.shape[0], imgData.shape[1], int(imgData.shape[2]/2)))
    if focus == 1:
        for i in range(0, int(imgData.shape[2]), 2):
            a = i/2
            newData[:,:,a] = imgData[:,:,i]
    else:
        for i in range(int(imgData.shape[2]/2)):
            newData[:,:,i] = imgData[:,:,(2*i)+1]

    return newData

def extractFrameData(file_obj, FileHeader, frameNum):
    tmp = np.zeros((int(FileHeader.rfbd.numSamplesPerVector-FileHeader.strf.vectorHeaderLengthBytes/2), \
                    int(FileHeader.csh0.numVectorsPerStreamFrame))).astype(np.int16)
    tmp2 = np.zeros((int(FileHeader.strf.vectorHeaderLengthBytes), int(FileHeader.csh0.numVectorsPerStreamFrame))).astype(np.uint8)

    file_obj.seek(FileHeader.idx1.frame[frameNum].chunkOffset+FileHeader.idx1.startOffset)
    tagName = file_obj.read(4).decode()
    dataSize = int.from_bytes(file_obj.read(4), 'little')
    if tagName != FileHeader.idx1.frame[frameNum].ckid:
        print("Warning: File Read Unsuccessful: Frame Location does not match Index Chunk")
    elif dataSize != FileHeader.idx1.frame[frameNum].chunkLength:
        print("Warning: File Read Unsuccessful: Frame size does not match Index Chunk")
    else:
        for i in range(FileHeader.csh0.numVectorsPerStreamFrame):
            vectorTmp = [int.from_bytes(file_obj.read(2), 'little', signed=True) for j in range(FileHeader.rfbd.numSamplesPerVector)]
            tmp[:,i] = vectorTmp[int(FileHeader.strf.vectorHeaderLengthBytes/2):]
            vectorTmp = vectorTmp[:int(FileHeader.strf.vectorHeaderLengthBytes/2)] + \
                                  65536*np.where((np.array(vectorTmp[:int(FileHeader.strf.vectorHeaderLengthBytes/2)]) < 0), 1, 0)
            tmp2[np.arange(0, FileHeader.strf.vectorHeaderLengthBytes, 2), i] = np.array(vectorTmp % 256).astype(np.uint8)
            tmp2[np.arange(1, FileHeader.strf.vectorHeaderLengthBytes, 2), i] = np.array(np.floor(vectorTmp/256)).astype(np.uint8)

    return tmp, tmp2

def readFileInfo(filepath):
    FileHeader = readHeader(filepath)

    # Add final parameters to info struct
    info = InfoStruct()
    info.samples = FileHeader.rfbd.numSamplesPerVector
    info.lines = FileHeader.rfbd.numVectorsPerFrame
    info.width = 1/FileHeader.rfbm.lineDensity
    info.samplingFrequency = FileHeader.strf.samplingRate
    info.centerFrequency = FileHeader.rfsd[0].txFrequencyMhz
    info.lineDensity = FileHeader.rfbm.lineDensity

    return info


def readHeader(filepath):
    file_obj = open(filepath, 'rb')
    worstCaseHeaderSize = 100000
    fileAsChars = [int.from_bytes(file_obj.read(1), 'little') for i in range(worstCaseHeaderSize)]
    FileHeader = FileHeaderStruct()

    # Do a temporary read to load the idx1 table
    FileHeader.rfgi = tmpRead('rfgi', file_obj, fileAsChars)
    FileHeader.idx1 = tmpRead('idx1', file_obj, fileAsChars)
    FileHeader.idx1.startOffset = tmpRead('movi', file_obj, fileAsChars)

    # Using the idx1 table, we know where the data starts, and we only need part
    # of the data up to that point into headers, so the searches in tmpRead can be faster.
    # the FRH0 (frame headers) are handled based on the idx1 chunk.
    # The 'if' statement is only for older URI data, can be removed once header format finalized
    if (FileHeader.idx1.frame[0].chunkOffset + FileHeader.idx1.startOffset) < worstCaseHeaderSize:
        fileAsChars = fileAsChars[:(FileHeader.idx1.frame[0].chunkOffset + FileHeader.idx1.startOffset)+1]
    
    FileHeader.rfbd = tmpRead('rfbd', file_obj, fileAsChars)
    FileHeader.avih = tmpRead('avih', file_obj, fileAsChars)
    FileHeader.sffm = tmpRead('sffm', file_obj, fileAsChars)
    FileHeader.strh = tmpRead('strh', file_obj, fileAsChars)
    FileHeader.strf = tmpRead('strf', file_obj, fileAsChars)
    FileHeader.cfh0 = tmpRead('cfh0', file_obj, fileAsChars)
    FileHeader.csh0 = tmpRead('csh0', file_obj, fileAsChars)
    FileHeader.stri = tmpRead('stri', file_obj, fileAsChars)
    FileHeader.rfam = tmpRead('rfam', file_obj, fileAsChars)
    FileHeader.rfsi = tmpRead('rfsi', file_obj, fileAsChars)
    FileHeader.rfbm = tmpRead('rfbm', file_obj, fileAsChars)
    FileHeader.rfmm = tmpRead('rfmm', file_obj, fileAsChars)
    FileHeader.rfdo = tmpRead('rfdo', file_obj, fileAsChars)
    FileHeader.rfco = tmpRead('rfco', file_obj, fileAsChars)
    FileHeader.rfsd = [[] for i in range(8)]
    for i in range(8):
        fourCC = str('RFS' + str(i))
        FileHeader.rfsd[i] = tmpRead(fourCC, file_obj, fileAsChars)

    FileHeader.frh0 = [[] for i in range(FileHeader.avih.dwTotalFrames)]
    for i in range(FileHeader.avih.dwTotalFrames):
        FileHeader.frh0[i] = idxRelativeRead('frh0', file_obj, i, FileHeader.idx1)

    return FileHeader

def idxRelativeRead(fourCC, file_obj, frameNum, idx1):
    file_obj.seek(0)
    out = readHeader_frh0(file_obj, frameNum, idx1)
    file_obj.seek(0)
    return out

def readHeader_frh0(file_obj, frameNum, idx1):
    frh0Size = 22
    file_obj.seek((idx1.frame[frameNum].chunkOffset)-frh0Size+idx1.startOffset)
    frh0 = Frh0Struct()

    frh0.tagName = file_obj.read(4).decode()
    if frh0.tagName != 'frh0':
        print("Warning: FRH0 Header Not Read Correctly")
        return
    
    frh0.dataSize = int.from_bytes(file_obj.read(4), 'little')
    frh0.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    frh0.frameTimeStamp = int.from_bytes(file_obj.read(4), 'little')
    frh0.isTriggeredFrame = int.from_bytes(file_obj.read(1), 'little')
    frh0.futureParameter7 = int.from_bytes(file_obj.read(4), 'little')
    frh0.futureParameter8 = int.from_bytes(file_obj.read(1), 'little')

    return frh0


def tmpRead(fourCC, file_obj, fileAsChars):
    # Credit Shelby Brunke, Jerome Mai, Siemens Ultrasound, 2003

    out = []
    if fourCC == 'rfgi':
        file_obj.seek(0)
        out = readHeader_rfgi(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'movi':
        file_obj.seek(0)
        out = getStartOffset(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'avih':
        file_obj.seek(0)
        out = readHeader_avih(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'sffm':
        file_obj.seek(0)
        out = readHeader_sffm(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'cfh0':
        file_obj.seek(0)
        out = readHeader_cfh0(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'csh0':
        file_obj.seek(0)
        out = readHeader_csh0(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'strh':
        file_obj.seek(0)
        out = readHeader_strh(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'strf':
        file_obj.seek(0)
        out = readHeader_strf(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'rfbd':
        file_obj.seek(0)
        out = readHeader_rfbd(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'idx1':
        file_obj.seek(0)
        avih = readHeader_avih(file_obj, fileAsChars)
        out = readHeader_idx1(file_obj, avih.dwTotalFrames)
        file_obj.seek(0)
    elif fourCC == '00db':
        file_obj.seek(0)
        out = readHeader_00db(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'stri':
        file_obj.seek(0)
        out = readHeader_stri(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'rfam':
        file_obj.seek(0)
        out = readHeader_rfam(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'rfsi':
        file_obj.seek(0)
        out = readHeader_rfsi(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'rfbm':
        file_obj.seek(0)
        out = readHeader_rfbm(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'rfmm':
        file_obj.seek(0)
        out = readHeader_rfmm(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'rfdo':
        file_obj.seek(0)
        out = readHeader_rfdo(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'rfco':
        file_obj.seek(0)
        out = readHeader_rfco(file_obj, fileAsChars)
        file_obj.seek(0)
    elif fourCC == 'RFS0':
        file_obj.seek(0)
        out = readHeader_rfsd(file_obj, fileAsChars, fourCC)
        file_obj.seek(0)
    elif fourCC == 'RFS1':
        file_obj.seek(0)
        out = readHeader_rfsd(file_obj, fileAsChars, fourCC)
        file_obj.seek(0)
    elif fourCC == 'RFS2':
        file_obj.seek(0)
        out = readHeader_rfsd(file_obj, fileAsChars, fourCC)
        file_obj.seek(0)
    elif fourCC == 'RFS3':
        file_obj.seek(0)
        out = readHeader_rfsd(file_obj, fileAsChars, fourCC)
        file_obj.seek(0)
    elif fourCC == 'RFS4':
        file_obj.seek(0)
        out = readHeader_rfsd(file_obj, fileAsChars, fourCC)
        file_obj.seek(0)
    elif fourCC == 'RFS5':
        file_obj.seek(0)
        out = readHeader_rfsd(file_obj, fileAsChars, fourCC)
        file_obj.seek(0)
    elif fourCC == 'RFS6':
        file_obj.seek(0)
        out = readHeader_rfsd(file_obj, fileAsChars, fourCC)
        file_obj.seek(0)
    elif fourCC == 'RFS7':
        file_obj.seek(0)
        out = readHeader_rfsd(file_obj, fileAsChars, fourCC)
        file_obj.seek(0)
    else:
        print("fourCC found no matches")
    
    return out

def getStartOffset(file_obj, fileAsChars):
    # Seems like redundant function of getFourCCByteLocation. Delete if found to be during testing
    topOfHeader = getFourCCByteLocation(fileAsChars, 'movi')
    file_obj.seek(topOfHeader)
    startOffset = file_obj.tell()
    
    tagName = file_obj.read(4).decode()
    
    if tagName != 'movi':
        print("Warning: Start Offset Location Not Found")
    
    return startOffset

def readHeader_rfsd(file_obj, fileAsChars, fourCC):
    topOfHeader = getFourCCByteLocation(fileAsChars, fourCC)
    file_obj.seek(topOfHeader)
    rfsd = RfsdStruct()

    rfsd.tagName = file_obj.read(4).decode()
    if rfsd.tagName != fourCC:
        print("Warning: {0} Header Not Read Correctly".format(fourCC))
        return
    
    rfsd.dataSize = int.from_bytes(file_obj.read(4), 'little')
    rfsd.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    rfsd.isDynamicFocusEn = int.from_bytes(file_obj.read(1), 'little')
    rfsd.rxFNum = struct.unpack('f', file_obj.read(4))[0]
    rfsd.rxFocusRangeCm = struct.unpack('f', file_obj.read(4))[0]
    rfsd.isAperGrowthEn = int.from_bytes(file_obj.read(1), 'little')
    rfsd.rxApodization = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfsd.txFocusRangeCm = struct.unpack('f', file_obj.read(4))[0]
    rfsd.txFNum = struct.unpack('f', file_obj.read(4))[0]
    rfsd.txFrequencyMhz = struct.unpack('f', file_obj.read(4))[0]
    rfsd.numTxCycles = struct.unpack('f', file_obj.read(4))[0]
    rfsd.txWaveformStyle = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfsd.prfHz = struct.unpack('f', file_obj.read(4))[0]
    rfsd.txApodization = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfsd.pulseAmplitude = struct.unpack('f', file_obj.read(4))[0]
    rfsd.analogGain = [struct.unpack('f', file_obj.read(4))[0] for i in range(512)]

    return rfsd

def readHeader_rfco(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'rfco')
    file_obj.seek(topOfHeader)
    rfco = RfcoStruct()

    rfco.tagName = file_obj.read(4).decode()
    if rfco.tagName != 'rfco':
        print("Warning: RFCO Header Not Read Correctly")
        return
    
    rfco.dataSize = int.from_bytes(file_obj.read(4), 'little')
    rfco.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    rfco.acousticFrameRateHz = struct.unpack('f', file_obj.read(4))[0]
    rfco.numParallelAcquisitions = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfco.fovShape = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfco.apexLateralCm = struct.unpack('f', file_obj.read(4))[0]
    rfco.apexVerticalCm = struct.unpack('f', file_obj.read(4))[0]
    rfco.displayedLateralMin = struct.unpack('f', file_obj.read(4))[0]
    rfco.displayedLateralSpan = struct.unpack('f', file_obj.read(4))[0]
    rfco.displayedAxialMinCm = struct.unpack('f', file_obj.read(4))[0]
    rfco.displayedAxialSpanCm = struct.unpack('f', file_obj.read(4))[0]
    rfco.lineDensity = struct.unpack('f', file_obj.read(4))[0]
    rfco.interleaveFactor = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfco.alternateInterleaveFactor = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfco.isFrameInterleaveEn = int.from_bytes(file_obj.read(1), 'little')
    rfco.ensembleSize = int.from_bytes(file_obj.read(4), 'little')

    return rfco

def readHeader_rfdo(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'rfdo')
    file_obj.seek(topOfHeader)
    rfdo = RfdoStruct()

    rfdo.tagName = file_obj.read(4).decode()
    if rfdo.tagName != 'rfdo':
        print("Warning: RFDO Header Not Read Correctly")
        return

    rfdo.dataSize = int.from_bytes(file_obj.read(4), 'little')
    rfdo.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    rfdo.fovShape = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfdo.gateStartCm = struct.unpack('f', file_obj.read(4))[0]
    rfdo.gateSizeCm = struct.unpack('f', file_obj.read(4))[0]
    rfdo.apexLateralCm = struct.unpack('f', file_obj.read(4))[0]
    rfdo.apexVerticalCm = struct.unpack('f', file_obj.read(4))[0]
    rfdo.roiLateralMin = struct.unpack('f', file_obj.read(4))[0]
    rfdo.steeringAngleRad = struct.unpack('f', file_obj.read(4))[0]
    rfdo.sampleVolumeWidth = struct.unpack('f', file_obj.read(4))[0]
    rfdo.dopplerType = int.from_bytes(file_obj.read(4), 'little', signed=True)

    return rfdo

def readHeader_rfmm(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'rfmm')
    file_obj.seek(topOfHeader)
    rfmm = RfmmStruct()

    rfmm.tagName = file_obj.read(4).decode()
    if rfmm.tagName != 'rfmm':
        print("Warning: RFMM Header Not Read Correctly")
        return
    
    rfmm.dataSize = int.from_bytes(file_obj.read(4), 'little')
    rfmm.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    rfmm.fovShape = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfmm.displayedAxialMinCm = struct.unpack('f', file_obj.read(4))[0]
    rfmm.displayedAxialSpanCm = struct.unpack('f', file_obj.read(4))[0]
    rfmm.apexLateralCm = struct.unpack('f', file_obj.read(4))[0]
    rfmm.apexVerticalCm = struct.unpack('f', file_obj.read(4))[0]
    rfmm.roiLateralMin = struct.unpack('f', file_obj.read(4))[0]
    rfmm.steeringAngleRad = struct.unpack('f', file_obj.read(4))[0]

    return rfmm

def readHeader_rfbm(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'rfbm')
    file_obj.seek(topOfHeader)
    rfbm = RfbmStruct()

    rfbm.tagName = file_obj.read(4).decode()
    if rfbm.tagName != 'rfbm':
        print("Warning: RFBM Header Not Read Correctly")
        return
    
    rfbm.dataSize = int.from_bytes(file_obj.read(4), 'little')
    rfbm.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    rfbm.numFocalZones = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfbm.acousticFrameRateHz = struct.unpack('f', file_obj.read(4))[0]
    rfbm.numParallelAcquisitions = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfbm.fovShape = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfbm.apexLateralCm = struct.unpack('f', file_obj.read(4))[0]
    rfbm.apexVerticalCm = struct.unpack('f', file_obj.read(4))[0]
    rfbm.displayedLateralMin = struct.unpack('f', file_obj.read(4))[0]
    rfbm.displayedLateralSpan = struct.unpack('f', file_obj.read(4))[0]
    rfbm.displayedAxialMinCm = struct.unpack('f', file_obj.read(4))[0]
    rfbm.displayedAxialSpanCm = struct.unpack('f', file_obj.read(4))[0]
    rfbm.steeringAngleRad = struct.unpack('f', file_obj.read(4))[0]
    rfbm.lineDensity = struct.unpack('f', file_obj.read(4))[0]
    rfbm.phaseInvertMode = int.from_bytes(file_obj.read(4), 'little', signed=True)

    return rfbm

def readHeader_rfsi(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'rfsi')
    file_obj.seek(topOfHeader)
    rfsi = RfsiStruct()

    rfsi.tagName = file_obj.read(4).decode()
    if rfsi.tagName != 'rfsi':
        print("Warning: RFSI Header Not Read Correctly")
        return

    rfsi.dataSize = int.from_bytes(file_obj.read(4), 'little')
    rfsi.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    rfsi.notes = file_obj.read(10).decode()
    rfsi.script = file_obj.read(12).decode()

    return rfsi

def readHeader_rfam(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'rfam')
    file_obj.seek(topOfHeader)
    rfam = RfamStruct()

    rfam.tagName = file_obj.read(4).decode()
    if rfam.tagName != 'rfam':
        print("Wanring: RFAM Header Not Read Correctly")
        return
    
    rfam.dataSize = int.from_bytes(file_obj.read(4), 'little')
    rfam.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    rfam.examIndex = int.from_bytes(file_obj.read(4), 'little', signed=True)
    probeTmp = np.array([int.from_bytes(file_obj.read(1), 'little') for i in range(32)])
    probeTmp = probeTmp[np.arange(0,32,2)]
    rfam.probeName = ''.join(chr(i) for i in probeTmp)
    rfam.probeRadiusCm = struct.unpack('f', file_obj.read(4))[0]
    rfam.isTrigger10n = int.from_bytes(file_obj.read(1), 'little')
    rfam.isTrigger20n = int.from_bytes(file_obj.read(1), 'little')
    rfam.trigger1DelaySec = struct.unpack('f', file_obj.read(4))[0]
    rfam.trigger2DelaySec = struct.unpack('f', file_obj.read(4))[0]
    rfam.trigger1WaveCount = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfam.trigger2WaveCount = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfam.futureParameter1 = int.from_bytes(file_obj.read(1), 'little')
    rfam.futureParameter2 = int.from_bytes(file_obj.read(1), 'little')
    rfam.futureParameter3 = struct.unpack('f', file_obj.read(4))[0]
    rfam.futureParameter4 = struct.unpack('f', file_obj.read(4))[0]
    rfam.futureParameter5 = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfam.futureParameter6 = struct.unpack('f', file_obj.read(4))[0]

    return rfam

def readHeader_stri(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'stri')
    file_obj.seek(topOfHeader)
    stri = StriStruct()
    stri.startStreamTimestamp = TimestampStruct()
    stri.endStreamTimestamp = TimestampStruct()

    stri.tagName = file_obj.read(4).decode()
    if stri.tagName != 'stri':
        print("Warning: STRI Header Not Read Correctly")
        return
    
    stri.dataSize = int.from_bytes(file_obj.read(4), 'little')

    stri.startStreamTimestamp.year = int.from_bytes(file_obj.read(2), 'little')
    stri.startStreamTimestamp.month = int.from_bytes(file_obj.read(2), 'little')
    stri.startStreamTimestamp.dayOfWeek = int.from_bytes(file_obj.read(2), 'little')
    stri.startStreamTimestamp.day = int.from_bytes(file_obj.read(2), 'little')
    stri.startStreamTimestamp.hour = int.from_bytes(file_obj.read(2), 'little')
    stri.startStreamTimestamp.minute = int.from_bytes(file_obj.read(2), 'little')
    stri.startStreamTimestamp.second = int.from_bytes(file_obj.read(2), 'little')
    stri.startStreamTimestamp.millisecond = int.from_bytes(file_obj.read(2), 'little')

    stri.endStreamTimestamp.year = int.from_bytes(file_obj.read(2), 'little')
    stri.endStreamTimestamp.month = int.from_bytes(file_obj.read(2), 'little')
    stri.endStreamTimestamp.dayOfWeek = int.from_bytes(file_obj.read(2), 'little')
    stri.endStreamTimestamp.day = int.from_bytes(file_obj.read(2), 'little')
    stri.endStreamTimestamp.hour = int.from_bytes(file_obj.read(2), 'little')
    stri.endStreamTimestamp.minute = int.from_bytes(file_obj.read(2), 'little')
    stri.endStreamTimestamp.second = int.from_bytes(file_obj.read(2), 'little')
    stri.endStreamTimestamp.millisecond = int.from_bytes(file_obj.read(2), 'little')

    stri.streamType = int.from_bytes(file_obj.read(4), 'little', signed=True)
    
    return stri



def readHeader_00db(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, '00db')
    file_obj.seek(topOfHeader)
    ffdb = Db00Struct()

    ffdb.tagName = file_obj.read(4)
    if ffdb.tagName != '00db':
        print("Warning: 00DB Header Not Read Correctly")
        return 
    
    ffdb.dataSize = int.from_bytes(file_obj.read(4), 'little')
    ffdb.startOfFrameData = file_obj.tell()

    return ffdb

def readHeader_idx1(file_obj, numFramesAcquired):
    offsetIdx1ToEofInBytes = 16*numFramesAcquired+8
    file_obj.seek((-1*offsetIdx1ToEofInBytes), 2)
    idx1 = Idx1Struct()

    idx1.tagName = file_obj.read(4).decode()
    if idx1.tagName != 'idx1':
        print("Warning: IDX1 Header Not Read Correctly")
        return

    idx1.dataSize = int.from_bytes(file_obj.read(4), 'little')
    idx1.frame = [FrameStruct() for i in range(numFramesAcquired)]
    for i in range(numFramesAcquired):
        idx1.frame[i].ckid = file_obj.read(4).decode()
        idx1.frame[i].dwFlags = int.from_bytes(file_obj.read(4), 'little')
        idx1.frame[i].chunkOffset = int.from_bytes(file_obj.read(4), 'little')
        idx1.frame[i].chunkLength = int.from_bytes(file_obj.read(4), 'little')
    
    return idx1


def readHeader_rfbd(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'rfbd')
    file_obj.seek(topOfHeader)
    rfbd = RfbdStruct()

    rfbd.tagName = file_obj.read(4).decode()
    if rfbd.tagName != 'rfbd':
        print("Warning: RFBD Header Not Read Correctly")
        return
    
    rfbd.dataSize = int.from_bytes(file_obj.read(4), 'little')
    rfbd.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    rfbd.numSamplesPerVector = int.from_bytes(file_obj.read(4), 'little') + 32
    rfbd.numVectorsPerFrame = int.from_bytes(file_obj.read(4), 'little')
    modeTmp = np.array([int.from_bytes(file_obj.read(1), 'little') for i in range(2*rfbd.numVectorsPerFrame)])
    modeTmp = modeTmp[np.arange(0, 2*rfbd.numVectorsPerFrame, 2)]
    rfbd.mode = ''.join(chr(i) for i in modeTmp)
    rfbd.set = [int.from_bytes(file_obj.read(4), 'little') for i in range(rfbd.numVectorsPerFrame)]
    rfbd.positionX = [struct.unpack('f', file_obj.read(4))[0] for i in range(rfbd.numVectorsPerFrame)]
    rfbd.positionZ = [struct.unpack('f', file_obj.read(4))[0] for i in range(rfbd.numVectorsPerFrame)]
    rfbd.thetaRad = [struct.unpack('f', file_obj.read(4))[0] for i in range(rfbd.numVectorsPerFrame)]

    return rfbd

def readHeader_strf(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'strf')
    file_obj.seek(topOfHeader)
    strf = StrfStruct()

    strf.tagName = file_obj.read(4).decode()
    if strf.tagName != 'strf':
        print("Warning: STRF Header Not Read Correctly")
        return
    
    strf.dataSize = int.from_bytes(file_obj.read(4), 'little')
    strf.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    strf.samplingRate = struct.unpack('f', file_obj.read(4))[0]
    strf.bitCountPerSample = int.from_bytes(file_obj.read(4), 'little')
    strf.sampleMask = int.from_bytes(file_obj.read(4), 'little')
    strf.vectorHeaderLengthBytes = int.from_bytes(file_obj.read(4), 'little')
    strf.compression = int.from_bytes(file_obj.read(4), 'little')
    
    return strf

def readHeader_strh(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'strh')
    file_obj.seek(topOfHeader)
    strh = StrhStruct()

    strh.tagName = file_obj.read(4).decode()
    if strh.tagName != 'strh':
        print("Warning: STRH Header Not Read Correctly")
        return
    
    strh.dataSize = int.from_bytes(file_obj.read(4), 'little')
    strh.fccType = file_obj.read(4).decode()
    strh.fccHandler = file_obj.read(4).decode()
    strh.dwFlags = int.from_bytes(file_obj.read(4), 'little')
    strh.dwPriority = int.from_bytes(file_obj.read(4), 'little')
    strh.dwInitialFrames = int.from_bytes(file_obj.read(4), 'little')
    strh.dwScale = int.from_bytes(file_obj.read(4), 'little')
    strh.dwRate = int.from_bytes(file_obj.read(4), 'little')
    strh.dwStart = int.from_bytes(file_obj.read(4), 'little')
    strh.dwLength = int.from_bytes(file_obj.read(4), 'little')
    strh.dwSuggestedBufferSize = int.from_bytes(file_obj.read(4), 'little')
    strh.quality = int.from_bytes(file_obj.read(4), 'little')
    strh.dwSampleSize = int.from_bytes(file_obj.read(4), 'little')
    strh.rcFrame_bottom = int.from_bytes(file_obj.read(4), 'little')
    strh.rcFrame_left = int.from_bytes(file_obj.read(4), 'little')
    strh.rcFrame_right = int.from_bytes(file_obj.read(4), 'little')
    strh.rcFrame_top = int.from_bytes(file_obj.read(4), 'little')
    
    return strh

def readHeader_csh0(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'csh0')
    file_obj.seek(topOfHeader)
    csh0 = Csh0Struct()

    csh0.tagName = file_obj.read(4).decode()
    if csh0.tagName != 'csh0':
        print("Warning: CSH0 Header Not Read Correctly")
        return
    
    csh0.dataSize = int.from_bytes(file_obj.read(4), 'little')
    csh0.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    csh0.frameClockAtStartStream = int.from_bytes(file_obj.read(4), 'little')
    csh0.requestForDataTimeStamp = int.from_bytes(file_obj.read(4), 'little')
    csh0.numCustomStreamHeaders = int.from_bytes(file_obj.read(4), 'little')
    csh0.numCustomFrameHeaders = int.from_bytes(file_obj.read(4), 'little')
    csh0.numFramesInStream = int.from_bytes(file_obj.read(4), 'little')
    csh0.numVectorsPerStreamFrame = int.from_bytes(file_obj.read(4), 'little')
    
    return csh0

def readHeader_cfh0(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'cfh0')
    file_obj.seek(topOfHeader)
    cfh0 = Cfh0Struct()

    cfh0.tagName = file_obj.read(4).decode()
    if cfh0.tagName != 'cfh0':
        print("Warning: CFH0 Header not Read Correctly")
        return
    
    cfh0.dataSize = int.from_bytes(file_obj.read(4), 'little')
    cfh0.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    cfh0.sequencingMode = int.from_bytes(file_obj.read(4), 'little', signed=True)
    cfh0.userName = file_obj.read(14).decode()
    cfh0.versionURI = int.from_bytes(file_obj.read(4), 'little', signed=True)
    cfh0.versionURIFileFormat = int.from_bytes(file_obj.read(4), 'little', signed=True)
    cfh0.versionRtcFirmware = int.from_bytes(file_obj.read(4), 'little', signed=True)
    cfh0.versionSipFirmware = int.from_bytes(file_obj.read(4), 'little', signed=True)
    cfh0.versionPciFirmware = int.from_bytes(file_obj.read(4), 'little', signed=True)
    cfh0.suppressPatientID = int.from_bytes(file_obj.read(1), 'little', signed=True)
    
    return cfh0


def readHeader_sffm(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'sffm')
    file_obj.seek(topOfHeader)
    sffm = SffmStruct()

    sffm.tagName = file_obj.read(4).decode()
    if sffm.tagName != 'sffm':
        print("Warning: SFFM Header Not Read Correctly")
        return
    
    sffm.dataSize = int.from_bytes(file_obj.read(4), 'little')
    sffm.sffVersion = int.from_bytes(file_obj.read(4), 'little')
    sffm.platformName = int.from_bytes(file_obj.read(4), 'little', signed=True)
    sffm.platformVersion = int.from_bytes(file_obj.read(4), 'little', signed=True)
    sffm.operatingSystem = int.from_bytes(file_obj.read(4), 'little', signed=True)
    sffm.operatingSystemVersion = int.from_bytes(file_obj.read(4), 'little', signed=True)
    sffm.cpu = int.from_bytes(file_obj.read(4), 'little', signed=True)
    
    return sffm


def readHeader_avih(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'avih')
    file_obj.seek(topOfHeader)
    avih = AvihStruct()

    avih.tagName = file_obj.read(4).decode()
    if avih.tagName != 'avih':
        print("Warning: AVIH Header Not Read Correctly")
        return
    
    avih.dataSize = int.from_bytes(file_obj.read(4), 'little')
    avih.dwMicroSecPerFrame = int.from_bytes(file_obj.read(4), 'little')
    avih.dwMaxBytesPerSec = int.from_bytes(file_obj.read(4), 'little')
    avih.resesrved1 = int.from_bytes(file_obj.read(4), 'little')
    avih.dwFlags = int.from_bytes(file_obj.read(4), 'little')
    avih.dwTotalFrames = int.from_bytes(file_obj.read(4), 'little')
    avih.dwInitialFrames = int.from_bytes(file_obj.read(4), 'little')
    avih.dwStreams = int.from_bytes(file_obj.read(4), 'little')
    avih.dwSuggestedBufferSize = int.from_bytes(file_obj.read(4), 'little')
    avih.dwWidth = int.from_bytes(file_obj.read(4), 'little')
    avih.dwHeight = int.from_bytes(file_obj.read(4), 'little')
    avih.dwReserved = [int.from_bytes(file_obj.read(4), 'little') for i in range(4)]
    
    return avih


def readHeader_rfgi(file_obj, fileAsChars):
    topOfHeader = getFourCCByteLocation(fileAsChars, 'rfgi')
    file_obj.seek(topOfHeader)
    rfgi = RfgiStruct()

    rfgi.tagName = file_obj.read(4).decode()
    if rfgi.tagName != 'rfgi':
        print("Warning: RFGI Header Not Read Correctly")
        return
    
    rfgi.dataSize = int.from_bytes(file_obj.read(4), 'little')
    rfgi.headerVersion = int.from_bytes(file_obj.read(4), 'little')
    rfgi.rfAcqGain = int.from_bytes(file_obj.read(4), 'little', signed=True)
    rfgi.minDataVectorRange = int.from_bytes(file_obj.read(4), 'little')
    rfgi.numFramesAcquired = int.from_bytes(file_obj.read(4), 'little')
    rfgi.rfAxialMinCm = struct.unpack('f', file_obj.read(4))[0]
    rfgi.rfAxialSpanCm = struct.unpack('f', file_obj.read(4))[0]
    rfgi.isBmodeInfoFrameDep = int.from_bytes(file_obj.read(1), 'little')
    rfgi.isMmodeInfoFrameDep = int.from_bytes(file_obj.read(1), 'little')
    rfgi.isColorInfoFrameDep = int.from_bytes(file_obj.read(1), 'little')
    rfgi.isDopplerInfoFrameDep = int.from_bytes(file_obj.read(1), 'little')
    rfgi.isSetDepInfoFrameDep = int.from_bytes(file_obj.read(1), 'little')
    rfgi.isBeamDepInfoFrameDep = int.from_bytes(file_obj.read(1), 'little')

    return rfgi


def getFourCCByteLocation(fileAsChars, fourCC):
    fourCC = [ord(fourCC[i]) for i in range(len(fourCC))]
    if len(fourCC) < len(fileAsChars):
        shorter = fourCC
        longer = fileAsChars
    else:
        shorter = fileAsChars
        longer = fourCC
    return findStartSubstring(shorter, longer)

def findStartSubstring(shortString, longString):
    startIndex = 0
    shortIndex = 0
    i = 0
    while shortIndex < len(shortString) and i < len(longString):
        if shortString[shortIndex] == longString[i]:
            if shortIndex == 0:
                startIndex = i
            shortIndex += 1
        else:
            shortIndex = 0
        i += 1
    if shortIndex < len(shortString):
        return 0
    return startIndex


if __name__ == "__main__":
    siemensParser('uri_SpV2232_VpF512_FpA90_20210129103529.rfd', '/Volumes/CREST Data/David_S_Data/Thyroid_Data_RFD/Phantom/Preset_2/', 'uri_SpV2232_VpF512_FpA90_20210129103529.rfd', '/Volumes/CREST Data/David_S_Data/Thyroid_Data_RFD/Phantom/Preset_2/')