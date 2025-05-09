import os
import platform
from pathlib import Path
from datetime import datetime
import warnings
import ctypes as ct
import logging
import numpy as np
from scipy.io import savemat
from philipsRfParser import getPartA, getPartB

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

###################################################################################
class PhilipsRFParser:
    """Main class for parsing Philips RF data files."""
    
    class HeaderInfoStruct:
        """Philips-specific structure containing information from the headers."""
        def __init__(self):
            self.RF_CaptureVersion: list
            self.Tap_Point: list
            self.Data_Gate: list
            self.Multilines_Capture: list
            self.Steer: list
            self.elevationPlaneOffset: list
            self.PM_Index: list
            self.Pulse_Index: list
            self.Data_Format: list
            self.Data_Type: list
            self.Header_Tag: list
            self.Threed_Pos: list
            self.Mode_Info: list
            self.Frame_ID: list
            self.CSID: list
            self.Line_Index: list
            self.Line_Type: list
            self.Time_Stamp: int
            self.RF_Sample_Rate: list

    class dbParams:
        """Philips-specific structure containing signal properties of the scan."""
        def __init__(self):
            self.acqNumActiveScChannels2d: list
            self.azimuthMultilineFactorXbrOut: list
            self.azimuthMultilineFactorXbrIn: list
            self.numOfSonoCTAngles2dActual: list
            self.elevationMultilineFactor: list
            self.numPiPulses: list
            self.num2DCols: list
            self.fastPiEnabled: list
            self.numZones2d: list
            self.numSubVols: list
            self.numPlanes: list
            self.zigZagEnabled: list
            self.azimuthMultilineFactorXbrOutCf: list
            self.azimuthMultilineFactorXbrInCf: list
            self.multiLineFactorCf: list
            self.linesPerEnsCf: list
            self.ensPerSeqCf: list
            self.numCfCols: list
            self.numCfEntries: list
            self.numCfDummies: list
            self.elevationMultilineFactorCf: list
            self.Planes: list
            self.tapPoint: list

    class Rfdata:
        """Philips-specific structure containing constructed RF data."""
        def __init__(self):
            self.lineData: np.ndarray  # Array containing interleaved line data (Data x XmitEvents)
            self.lineHeader: np.ndarray  # Array containing qualifier bits of the interleaved line data (Qualifiers x XmitEvents)
            self.headerInfo = PhilipsRFParser.HeaderInfoStruct()  # Structure containing information from the headers
            self.echoData: np.ndarray  # Array containing echo line data
            self.dbParams = PhilipsRFParser.dbParams()  # Structure containing dbParameters
            self.echoMModeData: list
            self.miscData: list

###################################################################################

    def __init__(self):
        """Initialize the PhilipsRFParser."""
        self.VHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 160, 160]
        self.FHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 11, 11]
        self.fileHeaderSize = len(self.VHeader)

###################################################################################

    def find_signature(self, filepath: Path) -> list:
        """Find the signature in the RF file."""
        logging.info(f"Attempting to open file: {filepath}")
        try:
            with open(filepath, 'rb') as file:
                sig = list(file.read(8))
                logging.info(f"Signature read: {sig}")
                return sig
        except Exception as e:
            logging.error(f"Failed to read signature from {filepath}: {e}")
            raise

###################################################################################

    def call_get_part_a(self, numClumps: int, filename: str, offset: int) -> np.ndarray:
        """Call getPartA to retrieve raw partA data."""
        logging.info(f"Calling getPartA with numClumps={numClumps}, filename='{filename}', offset={offset}")
        partA = getPartA(numClumps, filename, offset)
        partA = np.array(partA, dtype=int)
        partA = partA.reshape((12, numClumps), order='F')
        return partA

###################################################################################

    def call_get_part_b(self, numClumps: int, filename: str, offset: int) -> np.ndarray:
        """Call getPartB to retrieve raw partB data."""
        logging.info(f"Calling getPartB with numClumps={numClumps}, filename='{filename}', offset={offset}")
        partB = getPartB(numClumps, filename, offset)
        partB = np.array([partB], dtype=int)
        return partB

###################################################################################

    def prune_data(self, lineData, lineHeader, ML_Capture):
        """Remove false gate data and align samples."""
        logging.info("Starting pruneData function.")
        numSamples = lineData.shape[0]
        referenceLine = int(np.ceil(lineData.shape[1] * 0.2)) - 1
        startPoint = int(np.ceil(numSamples * 0.015)) - 1

        indicesFound = np.where(lineHeader[startPoint:numSamples + 1, referenceLine] == 3)
        if not len(indicesFound[0]):
            iFirstSample = 1
            logging.warning("No valid gate found, starting from sample 1.")
        else:
            iFirstSample = indicesFound[0][0] + startPoint

        alignment = np.arange(0, numSamples, np.double(ML_Capture))
        diff = alignment - iFirstSample
        iFirstSample = int(alignment[np.where(diff >= 0)[0][0]])

        prunedData = lineData[iFirstSample:numSamples + 1, :]
        lineHeader = lineHeader[iFirstSample:numSamples + 1, :]

        numSamples = prunedData.shape[0]
        startPoint = int(np.floor(numSamples * 0.99)) - 1

        indicesFound = np.where(lineHeader[startPoint:numSamples + 1, referenceLine] == 0)
        if not len(indicesFound[0]):
            iLastSample = numSamples
            logging.warning("No zero data found near the end, keeping full length.")
        else:
            iLastSample = indicesFound[0][0] + startPoint
            alignment = np.arange(0, numSamples, np.double(ML_Capture))
            diff = alignment - iLastSample
            iLastSample = int(alignment[np.where(diff >= 0)[0][0]]) - 1

        prunedData = prunedData[:iLastSample + 1, :]
        return prunedData

###################################################################################

    def sort_rf(self, RFinput, Stride, ML, CRE=1, isVoyager=True):
        """Sort RF data based on configuration."""
        logging.info(f"Starting SortRF with Stride={Stride}, ML={ML}, CRE={CRE}")
        N = RFinput.shape[0]
        xmitEvents = RFinput.shape[1]
        depth = int(np.floor(N / Stride))
        MLs = np.arange(0, ML)

        # Preallocate output arrays
        out0 = np.zeros((depth, ML, xmitEvents))
        out1 = np.zeros((depth, ML, xmitEvents)) if CRE >= 2 else None
        out2 = np.zeros((depth, ML, xmitEvents)) if CRE >= 3 else None
        out3 = np.zeros((depth, ML, xmitEvents)) if CRE == 4 else None

        # Determine ML_SortList based on Stride and CRE
        ML_SortList = self._get_ml_sort_list(Stride, CRE)
        if ML_SortList is None:
            return None, None, None, None

        # Sort the RF input into outputs
        for k in range(ML):
            iML = np.where(np.array(ML_SortList) == MLs[k])[0]
            if iML.size == 0:
                logging.warning(f"No entries found for ML {MLs[k]}")
                continue

            out0[:depth, k, :] = RFinput[np.arange(iML[0], (depth * Stride), Stride), :]
            if CRE >= 2 and iML.size > 1:
                out1[:depth, k, :] = RFinput[np.arange(iML[1], (depth * Stride), Stride), :]
            if CRE >= 3 and iML.size > 2:
                out2[:depth, k, :] = RFinput[np.arange(iML[2], (depth * Stride), Stride), :]
            if CRE == 4 and iML.size > 3:
                out3[:depth, k, :] = RFinput[np.arange(iML[3], (depth * Stride), Stride), :]

        return out0, out1, out2, out3

###################################################################################

    def _get_ml_sort_list(self, Stride, CRE):
        """Get the ML sort list based on stride and CRE."""
        if Stride == 128:
            return list(range(128))
        elif Stride == 32:
            if CRE == 4:
                return [4, 4, 5, 5, 6, 6, 7, 7, 4, 4, 5, 5, 6, 6, 7, 7,
                       0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]
            return list(range(16)) * 2
        elif Stride == 16:
            if CRE == 1:
                return [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
            elif CRE == 2:
                return list(range(8)) * 2
            elif CRE == 4:
                return [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]
        elif Stride == 12:
            if CRE == 1:
                return [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
            elif CRE == 2:
                return [0, 1, 2, 3, 4, 5] * 2
            elif CRE == 4:
                return [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
        elif Stride == 8:
            if CRE == 1:
                return [0, 2, 4, 6, 1, 3, 5, 7]
            elif CRE == 2:
                return [0, 1, 2, 3] * 2
            elif CRE == 4:
                return [0, 0, 1, 1, 0, 0, 1, 1]
        elif Stride == 4:
            if CRE == 1:
                return [0, 2, 1, 3]
            elif CRE == 2:
                return [0, 1, 0, 1]
            elif CRE == 4:
                return [0, 0, 0, 0]
        elif Stride == 2:
            if CRE == 1:
                return [0, 1]
            elif CRE in (2, 4):
                return [0, 0]
        logging.error("No sort list for this stride value.")
        return None

###################################################################################

    def parse_data_f(self, rawrfdata, headerInfo):
        """Parse data in Fusion format."""
        logging.info("Starting parseDataF function.")
        minNeg = 2**18

        iHeader = np.array(np.where((rawrfdata[0, :] & 1572864) == 524288))[0]
        numHeaders = len(iHeader) - 1

        maxNumSamples = 0
        for m in range(numHeaders):
            tempMax = iHeader[m+1] - iHeader[m] - 1
            if tempMax > maxNumSamples:
                maxNumSamples = tempMax

        numSamples = maxNumSamples * 12
        lineData = np.zeros((numSamples, numHeaders), dtype=np.int32)
        lineHeader = np.zeros((numSamples, numHeaders), dtype=np.uint8)

        for m in range(numHeaders):
            iStartData = iHeader[m] + 2
            iStopData = iHeader[m+1] - 1

            if headerInfo.Data_Type[m] == float(0x5A):
                iStopData = iStartData + 10000

            lineData_u32 = rawrfdata[:12, iStartData:iStopData + 1]
            lineData_s32 = np.int32(lineData_u32 & 524287)

            iNeg = np.where(lineData_s32 >= minNeg)
            lineData_s32[iNeg] -= (2 * minNeg)

            lineHeader_u8 = (lineData_u32 & 1572864) >> 19

            lineData[:lineData_s32.size, m] = lineData_s32.ravel(order='F')
            lineHeader[:lineHeader_u8.size, m] = lineHeader_u8.ravel(order='F')

        return lineData, lineHeader

###################################################################################

    def parse_header_v(self, rawrfdata):
        """Parse header in Voyager format."""
        logging.info("Starting parseHeaderV function.")
        temp_headerInfo = self.HeaderInfoStruct()

        iHeader = np.where(np.uint8(rawrfdata[2, 0, :]) & 224)[0]
        numHeaders = len(iHeader) - 1

        for m in range(numHeaders):
            packedHeader = ''
            for k in np.arange(11, 0, -1):
                temp = ''
                for i in np.arange(2, 0, -1):
                    byte_value = np.uint8(rawrfdata[i, k, iHeader[m]])
                    temp += bin(byte_value)[2:].zfill(8)
                packedHeader += temp[3:24]

            iBit = 0
            temp_headerInfo.RF_CaptureVersion[m] = int(packedHeader[iBit:iBit + 4], 2)
            iBit += 4
            temp_headerInfo.Tap_Point[m] = int(packedHeader[iBit:iBit + 3], 2)
            iBit += 3
            temp_headerInfo.Data_Gate[m] = int(packedHeader[iBit], 2)
            iBit += 1
            temp_headerInfo.Multilines_Capture[m] = int(packedHeader[iBit:iBit + 4], 2)
            iBit += 4
            temp_headerInfo.RF_Sample_Rate[m] = int(packedHeader[iBit], 2)
            iBit += 1
            temp_headerInfo.Steer[m] = int(packedHeader[iBit:iBit + 6], 2)
            iBit += 6
            temp_headerInfo.elevationPlaneOffset[m] = int(packedHeader[iBit:iBit + 8], 2)
            iBit += 8
            temp_headerInfo.PM_Index[m] = int(packedHeader[iBit:iBit + 2], 2)
            iBit += 2
            temp_headerInfo.Line_Index[m] = int(packedHeader[iBit:iBit + 16], 2)
            iBit += 16
            temp_headerInfo.Pulse_Index[m] = int(packedHeader[iBit:iBit + 16], 2)
            iBit += 16
            temp_headerInfo.Data_Format[m] = int(packedHeader[iBit:iBit + 16], 2)
            iBit += 16
            temp_headerInfo.Data_Type[m] = int(packedHeader[iBit:iBit + 16], 2)
            iBit += 16
            temp_headerInfo.Header_Tag[m] = int(packedHeader[iBit:iBit + 16], 2)
            iBit += 16
            temp_headerInfo.Threed_Pos[m] = int(packedHeader[iBit:iBit + 16], 2)
            iBit += 16
            temp_headerInfo.Mode_Info[m] = int(packedHeader[iBit:iBit + 16], 2)
            iBit += 16
            temp_headerInfo.Frame_ID[m] = int(packedHeader[iBit:iBit + 32], 2)
            iBit += 32
            temp_headerInfo.CSID[m] = int(packedHeader[iBit:iBit + 16], 2)
            iBit += 16
            temp_headerInfo.Line_Type[m] = int(packedHeader[iBit:iBit + 16], 2)
            iBit += 16
            temp_headerInfo.Time_Stamp = int(packedHeader[iBit:iBit + 32], 2)

        return temp_headerInfo

###################################################################################

    def parse_data_v(self, rawrfdata, headerInfo):
        """Parse data in Voyager format."""
        logging.info("Starting parseDataV function.")
        minNeg = 16 * (2**16)

        iHeader = np.where((rawrfdata[2, 0, :] & 224) == 64)[0]
        numHeaders = len(iHeader) - 1

        if numHeaders <= 0:
            logging.error("Not enough headers found! Exiting.")
            return None, None

        numSamples = (iHeader[1] - iHeader[0] - 1) * 12
        lineData = np.zeros((numSamples, numHeaders), dtype=np.int32)
        lineHeader = np.zeros((numSamples, numHeaders), dtype=np.uint8)

        for m in range(numHeaders):
            iStartData = iHeader[m] + 1
            iStopData = iHeader[m + 1] - 1

            if headerInfo.Data_Type[m] == float(0x5A):
                iStopData = iStartData + 10000

            lineData_u8 = rawrfdata[:, :, iStartData:iStopData]
            lineData_s32 = (np.int32(lineData_u8[0, :, :]) +
                           (np.int32(lineData_u8[1, :, :]) << 8) +
                           (np.int32(lineData_u8[2, :, :] & np.uint8(31)) << 16))

            iNeg = np.where(lineData_s32 >= minNeg)
            lineData_s32[iNeg] -= 2 * minNeg

            lineHeader_u8 = (lineData_u8[2, :, :] & 224) >> 6

            flatData = lineData_s32.ravel(order='F')
            flatHeader = lineHeader_u8.ravel(order='F')

            lineData[:flatData.size, m] = flatData
            lineHeader[:flatHeader.size, m] = flatHeader

        return lineData, lineHeader

###################################################################################

    def parse_file_header(self, file_obj, endianness):
        """Parse the file header."""
        logging.info("Starting parseFileHeader.")

        fileVersion = int.from_bytes(file_obj.read(4), endianness, signed=False)
        numFileHeaderBytes = int.from_bytes(file_obj.read(4), endianness, signed=False)
        logging.info(f"File Version: {fileVersion}")
        logging.info(f"Header Size: {numFileHeaderBytes} bytes")

        temp_dbParams = self.dbParams()

        if fileVersion in {2, 3, 4, 5, 6}:
            # Common fields
            temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]
            temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]
            temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]

            if fileVersion >= 4:
                temp_dbParams.azimuthMultilineFactorXbrOutCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3)]
                temp_dbParams.azimuthMultilineFactorXbrInCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3)]

            temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]
            temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]

            if fileVersion >= 4:
                temp_dbParams.elevationMultilineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3)]

            if fileVersion >= 5:
                temp_dbParams.multiLineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3)]

            temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]
            temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(14 * 11)], (14, 11), order='F')
            temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]
            temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]

            temp_dbParams.numSubVols = int.from_bytes(file_obj.read(4), endianness, signed=False)
            temp_dbParams.numPlanes = int.from_bytes(file_obj.read(4), endianness, signed=False)
            temp_dbParams.zigZagEnabled = int.from_bytes(file_obj.read(4), endianness, signed=False)

            if fileVersion in {3, 4, 5, 6}:
                temp_dbParams.linesPerEnsCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]
                temp_dbParams.ensPerSeqCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]
                temp_dbParams.numCfCols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(14)]
                temp_dbParams.numCfEntries = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]
                temp_dbParams.numCfDummies = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(3 if fileVersion >= 4 else 4)]

            if fileVersion == 6:
                temp_dbParams.tapPoint = int.from_bytes(file_obj.read(4), endianness, signed=False)

        else:
            logging.error(f"Unknown file version: {fileVersion}")
            raise ValueError(f"Unknown file version: {fileVersion}")

        return temp_dbParams, numFileHeaderBytes

###################################################################################

    def parse_rf(self, filepath: str, readOffset: int, readSize: int) -> Rfdata:
        """Main method to parse an RF data file."""
        logging.info(f"Starting parseRF for file: {filepath}")
        start_time = datetime.now()

        rfdata = self.Rfdata()
        logging.info(f"Initialized Rfdata structure")

        # Determine file format and read initial header info
        isVoyager, hasFileHeader, totalHeaderSize = self._analyze_header(filepath)
        logging.info(f"File analysis results - isVoyager: {isVoyager}, hasFileHeader: {hasFileHeader}, totalHeaderSize: {totalHeaderSize}")
        
        endianness = 'big' if isVoyager else 'little'
        logging.info(f"Using {endianness} endianness for parsing")

        # Align file read boundaries based on format
        readOffset, readSize = self._align_offsets(filepath, readOffset, readSize, totalHeaderSize, isVoyager)
        logging.info(f"Aligned read parameters - offset: {readOffset}, size: {readSize}")

        # Load raw data from file
        logging.info("Loading raw data from file...")
        rawrfdata = self._load_raw_data(filepath, readOffset, readSize, totalHeaderSize, isVoyager)
        logging.info(f"Loaded raw data of size: {len(rawrfdata) if isinstance(rawrfdata, bytes) else rawrfdata.shape}")

        # Parse metadata headers from raw data
        logging.info("Parsing metadata headers...")
        headerInfo = self.parse_header_v(rawrfdata) if isVoyager else self.parse_header_f(rawrfdata)
        logging.info("Header parsing completed")

        # Parse actual signal data
        logging.info("Parsing RF signal data...")
        lineData, lineHeader, Tap_Point = self._parse_rf_data(rawrfdata, headerInfo, isVoyager)
        logging.info(f"Signal data parsed - lineData shape: {lineData.shape}, lineHeader shape: {lineHeader.shape}")

        # Store parsed values in rfdata structure
        rfdata.lineData = lineData
        rfdata.lineHeader = lineHeader
        rfdata.headerInfo = headerInfo
        del rawrfdata  # Free memory
        logging.info("Stored parsed data in rfdata structure")

        # Determine multi-line capture configuration and correct Tap Point
        ML_Capture, Tap_Point = self._determine_capture_info(headerInfo, Tap_Point)
        logging.info(f"Capture configuration determined - ML_Capture: {ML_Capture}, Tap_Point: {Tap_Point}")

        # Log parsed capture metadata
        self._log_capture_info(Tap_Point, ML_Capture)

        # Final organization of RF data
        logging.info("Organizing RF data into different types...")
        self._organize_rfdata(rfdata, ML_Capture, isVoyager)
        logging.info("RF data organization completed")

        elapsed_time = datetime.now() - start_time
        logging.info(f"Completed parseRF(). Total elapsed time: {elapsed_time}")
        return rfdata

###################################################################################

    def _analyze_header(self, filepath):
        """Analyze the file header to determine file type and header size."""
        with open(filepath, 'rb') as file_obj:
            fileHeader = list(file_obj.read(self.fileHeaderSize))
            isVoyager = fileHeader == self.VHeader
            hasFileHeader = fileHeader in (self.VHeader, self.FHeader)

            if hasFileHeader:
                logging.info("Header information found.")
                logging.info("Parsing Voyager RF capture file..." if isVoyager else "Parsing Fusion RF capture file...")
                endianness = 'big' if isVoyager else 'little'
                rfdata = self.Rfdata()
                rfdata.dbParams, numFileHeaderBytes = self.parse_file_header(file_obj, endianness)
                totalHeaderSize = self.fileHeaderSize + 8 + numFileHeaderBytes
            else:
                logging.info("Parsing legacy Voyager RF capture file (no standard header).")
                isVoyager = True
                totalHeaderSize = 0

        return isVoyager, hasFileHeader, totalHeaderSize

###################################################################################

    def _align_offsets(self, filepath, readOffsetMB, readSizeMB, totalHeaderSize, isVoyager):
        """Align read offset and size to block boundaries."""
        readOffset = readOffsetMB * (2**20)
        readSize = readSizeMB * (2**20)
        remainingSize = os.stat(filepath).st_size - totalHeaderSize

        alignment = 36 if isVoyager else 32
        aligned = np.arange(0, remainingSize + 1, alignment)

        readOffset = aligned[np.searchsorted(aligned, readOffset)]
        readSize = aligned[np.searchsorted(aligned, readSize)]

        return readOffset, readSize

###################################################################################

    def _load_raw_data(self, filepath, readOffset, readSize, totalHeaderSize, isVoyager):
        """Load raw data from file."""
        with open(filepath, 'rb') as f:
            f.seek(totalHeaderSize + readOffset)
            rawrfdata = f.read(readSize)

        if isVoyager:
            return rawrfdata
        else:
            numClumps = int(np.floor(readSize / 32))
            offset = totalHeaderSize + readOffset
            partA = self.call_get_part_a(numClumps, filepath, offset)
            partB = self.call_get_part_b(numClumps, filepath, offset)
            return np.concatenate((partA, partB))

###################################################################################

    def _parse_rf_data(self, rawrfdata, headerInfo, isVoyager):
        """Parse RF data based on file format."""
        Tap_Point = headerInfo.Tap_Point[0]

        if isVoyager:
            lineData, lineHeader = self.parse_data_v(rawrfdata, headerInfo)
        else:
            lineData, lineHeader = self.parse_data_f(rawrfdata, headerInfo)
            if Tap_Point == 0:
                lineData <<= 2

        return lineData, lineHeader, Tap_Point

###################################################################################

    def _determine_capture_info(self, headerInfo, Tap_Point):
        """Determine capture configuration."""
        if Tap_Point == 7:
            ML_Capture = 128
        else:
            ML_Capture = np.double(headerInfo.Multilines_Capture[0])

        if ML_Capture == 0:
            SAMPLE_RATE = np.double(headerInfo.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32

        if Tap_Point == 7:
            Tap_Point = 4

        return ML_Capture, Tap_Point

###################################################################################

    def _log_capture_info(self, Tap_Point, ML_Capture):
        """Log capture configuration."""
        namePoint = ['PostShepard', 'PostAGNOS', 'PostXBR', 'PostQBP', 'PostADC']
        logging.info(f"Capture Tap Point: {namePoint[Tap_Point]}")
        logging.info(f"ML Capture: {ML_Capture}x")

###################################################################################

    def _organize_rfdata(self, rfdata, ML_Capture, isVoyager):
        """Organize RF data into different types."""
        DataType_ECHO = np.arange(1, 15)
        DataType_EchoMMode = 26
        DataType_COLOR = [17, 21, 22, 23, 24]
        DataType_ColorMMode = [27, 28]
        DataType_CW = 16
        DataType_PW = [18, 19]
        DataType_Dummy = [20, 25, 29, 30, 31]
        DataType_SWI = [90, 91]
        DataType_Misc = [15, 88, 89]

        def find_indices(dataTypes):
            xmitEvents = len(rfdata.headerInfo.Data_Type)
            mask = np.zeros(xmitEvents, dtype=bool)
            for dt in dataTypes:
                mask = np.bitwise_or(mask, rfdata.headerInfo.Data_Type == dt)
            return mask

        # Process each data type
        for data_type, name in [
            (DataType_ECHO, 'echoData'),
            ([DataType_EchoMMode], 'echoMModeData'),
            (DataType_COLOR, 'colorData'),
            (DataType_ColorMMode, 'colorMModeData'),
            ([DataType_CW], 'cwData'),
            (DataType_PW, 'pwData'),
            (DataType_Dummy, 'dummyData'),
            (DataType_SWI, 'swiData'),
            (DataType_Misc, 'miscData')
        ]:
            mask = find_indices(data_type)
            if np.sum(mask) > 0:
                data = self.prune_data(rfdata.lineData[:, mask], rfdata.lineHeader[:, mask], ML_Capture)
                setattr(rfdata, name, self.sort_rf(data, ML_Capture, ML_Capture, 1, isVoyager))

###################################################################################

    def parse(self, filepath: str, ML_out=2, ML_in=32, used_os=2256) -> np.ndarray:
        """Parse Philips RF data file and save to .mat file."""
        logging.info(f"Starting philipsRfParser for file: {filepath}")
        logging.info(f"Parameters - ML_out: {ML_out}, ML_in: {ML_in}, used_os: {used_os}")

        # Parse RF file
        logging.info("Parsing RF file...")
        rf = self.parse_rf(filepath, 0, 2000)
        logging.info("RF file parsing completed")

        # Check even/odd line index to remove duplicate lines
        logging.info("Checking for duplicate lines...")
        if rf.headerInfo.Line_Index[249] == rf.headerInfo.Line_Index[250]:
            rf.lineData = rf.lineData[:, np.arange(2, rf.lineData.shape[1], 2)]
            logging.info("Detected even-indexed duplicate, skipping even lines")
        else:
            rf.lineData = rf.lineData[:, np.arange(1, rf.lineData.shape[1], 2)]
            logging.info("Detected odd-indexed duplicate, skipping odd lines")

        txBeamperFrame = np.array(rf.dbParams.num2DCols).flat[0]
        NumSonoCTAngles = rf.dbParams.numOfSonoCTAngles2dActual[0]
        logging.info(f"Beam parameters - txBeamperFrame: {txBeamperFrame}, NumSonoCTAngles: {NumSonoCTAngles}")

        # Calculate parameters
        numFrame = int(np.floor(rf.lineData.shape[1] / (txBeamperFrame * NumSonoCTAngles)))
        multilinefactor = ML_in
        pt = int(np.floor((rf.lineData.shape[0] - used_os) / multilinefactor))
        logging.info(f"Calculated parameters - numFrame: {numFrame}, multilinefactor: {multilinefactor}, points: {pt}")

        # Initialize arrays
        logging.info("Initializing data arrays...")
        rftemp_all_harm = np.zeros((pt, ML_out * txBeamperFrame))
        rftemp_all_fund = np.zeros((pt, ML_out * txBeamperFrame))
        rf_data_all_harm = np.zeros((numFrame, NumSonoCTAngles, pt, ML_out * txBeamperFrame))
        rf_data_all_fund = np.zeros((numFrame, NumSonoCTAngles, pt, ML_out * txBeamperFrame))
        logging.info("Array initialization completed")

        # Fill data arrays
        logging.info("Filling data arrays...")
        for k0 in range(numFrame):
            logging.info(f"Processing frame {k0+1}/{numFrame}")
            for k1 in range(NumSonoCTAngles):
                for k2 in range(txBeamperFrame):
                    bi = k0 * txBeamperFrame * NumSonoCTAngles + k1 * txBeamperFrame + k2
                    temp = np.transpose(
                        np.reshape(rf.lineData[used_os + np.arange(pt * multilinefactor), bi],
                                 (multilinefactor, pt), order='F')
                    )
                    rftemp_all_harm[:, np.arange(ML_out) + (k2 * ML_out)] = temp[:, [0, 2]]
                    rftemp_all_fund[:, np.arange(ML_out) + (k2 * ML_out)] = temp[:, [9, 11]]

                rf_data_all_harm[k0, k1] = rftemp_all_harm
                rf_data_all_fund[k0, k1] = rftemp_all_fund
        logging.info("Data array filling completed")

        # Prepare contents for .mat saving
        logging.info("Preparing data for .mat file...")
        contents = {
            'echoData': rf.echoData[0],
            'lineData': rf.lineData,
            'lineHeader': rf.lineHeader,
            'headerInfo': rf.headerInfo,
            'dbParams': rf.dbParams,
            'rf_data_all_fund': rf_data_all_fund,
            'rf_data_all_harm': rf_data_all_harm,
            'NumFrame': numFrame,
            'NumSonoCTAngles': NumSonoCTAngles,
            'pt': pt,
            'multilinefactor': multilinefactor,
        }

        # Add optional fields
        if len(rf.echoData) > 1 and len(rf.echoData[1]):
            contents['echoData1'] = rf.echoData[1]
            logging.info("Added echoData1 to contents")
        if len(rf.echoData) > 2 and len(rf.echoData[2]):
            contents['echoData2'] = rf.echoData[2]
            logging.info("Added echoData2 to contents")
        if len(rf.echoData) > 3 and len(rf.echoData[3]):
            contents['echoData3'] = rf.echoData[3]
            logging.info("Added echoData3 to contents")
        if hasattr(rf, 'echoMModeData'):
            contents['echoMModeData'] = rf.echoMModeData
            logging.info("Added echoMModeData to contents")
        if hasattr(rf, 'miscData'):
            contents['miscData'] = rf.miscData
            logging.info("Added miscData to contents")

        # Save to .mat
        destination = filepath.rsplit('.', 1)[0] + '.mat'
        if os.path.exists(destination):
            logging.info(f"Removing existing file: {destination}")
            os.remove(destination)

        logging.info(f"Saving data to {destination}")
        savemat(destination, contents)
        logging.info(f"Successfully saved parsed data to {destination}")

        return rf_data_all_fund.shape

###################################################################################

if __name__ == "__main__":
    parser = PhilipsRFParser()
    parser.parse(r"d:\Omid\0_samples\Philips\UKDFIBEPIC003\UKDFIBEPIC003INTER3D_20250424_094008.rf")

###################################################################################