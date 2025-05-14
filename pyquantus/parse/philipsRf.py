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

###################################################################################
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

###################################################################################
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

###################################################################################
class Rfdata:
    """Philips-specific structure containing donstructed RF data."""
    def __init__(self):
        self.lineData: np.ndarray # Array containing interleaved line data (Data x XmitEvents)
        self.lineHeader: np.ndarray # Array containing qualifier bits of the interleaved line data (Qualifiers x XmitEvents)
        self.headerInfo = HeaderInfoStruct() # Structure containing information from the headers
        self.echoData: np.ndarray # Array containing echo line data
        self.dbParams = dbParams() # Structure containing dbParameters. Should match feclib::RFCaptureDBInfo
        self.echoMModeData: list
        self.miscData: list

###################################################################################
class PhilipsRfParser:
    """Main class for parsing Philips RF data files."""
    ###################################################################################
    
    def __init__(self, ML_out=2, ML_in=32, used_os=None):
        """Initialize the parser with default parameters."""
        logging.info(f"Initializing PhilipsRfParser with ML_out={ML_out}, ML_in={ML_in}, used_os={used_os}")
        self.ML_out = ML_out
        self.ML_in = ML_in
        self.used_os = used_os
        self.rfdata = None
        self.txBeamperFrame = None
        self.NumSonoCTAngles = None
        self.numFrame = None
        self.multilinefactor = None
        self.pt = None
        
    ###################################################################################
    
    def _find_signature(self, filepath: Path):
        """Find file signature."""
        logging.debug(f"Finding file signature for {filepath}")
        file = open(filepath, 'rb')
        sig = list(file.read(8))
        logging.debug(f"File signature: {sig}")
        return sig

    ###################################################################################
    
    def _call_get_part_a(self, numClumps: int, filename: str, offset: int) -> np.ndarray:
        """Call getPartA from C library."""
        logging.debug(f"Calling getPartA with numClumps={numClumps}, offset={offset}")
        partA = getPartA(numClumps, filename, offset)
        partA = np.array(partA, dtype=int).reshape((12, numClumps), order='F')
        logging.debug(f"getPartA returned array of shape {partA.shape}")
        return partA

    ###################################################################################
    
    def _call_get_part_b(self, numClumps: int, filename: str, offset: int) -> np.ndarray:
        """Call getPartB from C library."""
        logging.debug(f"Calling getPartB with numClumps={numClumps}, offset={offset}")
        partB = getPartB(numClumps, filename, offset)
        partB = np.array([partB], dtype=int)
        logging.debug(f"getPartB returned array of shape {partB.shape}")
        return partB

    ###################################################################################
    
    def _prune_data(self, lineData, lineHeader, ML_Capture):
        """Remove false gate data at beginning of the line."""
        logging.info(f"Pruning data - input shape: lineData={lineData.shape}, lineHeader={lineHeader.shape}, ML_Capture={ML_Capture}")
        
        # Remove false gate data at beginning of the line
        numSamples = lineData.shape[0]
        referenceLine = int(np.ceil(lineData.shape[1]*0.2))-1    
        startPoint = int(np.ceil(numSamples*0.015))-1
        logging.debug(f"Looking for start point from sample {startPoint} in reference line {referenceLine}")
        
        indicesFound = np.where(lineHeader[startPoint:numSamples+1, referenceLine]==3)
        if not len(indicesFound[0]):
            iFirstSample = 1
            logging.debug("No valid start point found, using sample 1")
        else:
            iFirstSample = indicesFound[0][0]+startPoint
            logging.debug(f"Found start point at sample {iFirstSample}")
        
        # Align the pruning
        alignment = np.arange(0,numSamples, np.double(ML_Capture))
        diff = alignment - iFirstSample
        iFirstSample = int(alignment[np.where(diff>=0)[0][0]])
        logging.debug(f"Aligned start point to {iFirstSample}")
        
        # Prune data
        prunedData = lineData[iFirstSample:numSamples+1,:]
        lineHeader = lineHeader[iFirstSample:numSamples+1,:]
        logging.debug(f"Pruned from start: new shape {prunedData.shape}")
                   
        # Remove zero data at end of the line
        # Start from last 1 of the line
        numSamples = prunedData.shape[0]
        startPoint = int(np.floor(numSamples*0.99))-1
        logging.debug(f"Looking for end point from sample {startPoint}")
        
        indicesFound = np.where(lineHeader[startPoint:numSamples+1,referenceLine]==0)
        if not len(indicesFound[0]):
            iLastSample = numSamples
            logging.debug("No valid end point found, using last sample")
        else:
            iLastSample = indicesFound[0][0]+startPoint
            # Align the pruning
            alignment = np.arange(0,numSamples, np.double(ML_Capture))
            diff = alignment - iLastSample
            iLastSample = int(alignment[np.where(diff >= 0)[0][0]])-1
            logging.debug(f"Found and aligned end point to {iLastSample}")
        
        # Prune data
        prunedData = prunedData[:iLastSample+1, :]
        logging.info(f"Pruning complete - final shape: {prunedData.shape}")

        return prunedData

    ###################################################################################
    
    def _sort_rf(self, RFinput, Stride, ML, CRE=1, isVoyager=True):
        """Sort RF data based on multiline parameters."""
        logging.info(f"Sorting RF data - input shape: {RFinput.shape}, Stride={Stride}, ML={ML}, CRE={CRE}, isVoyager={isVoyager}")
        
        # Initialize default parameters
        N = RFinput.shape[0]
        xmitEvents = RFinput.shape[1]
        depth = int(np.floor(N/Stride))
        MLs = np.arange(0,ML)

        # Make into Column Vector
        MLs = MLs[:]

        out1 = np.array([])
        out2 = np.array([])
        out3 = np.array([])
        
        # Preallocate output array, but only for those that will be used
        logging.debug(f"Preallocating arrays for CRE={CRE}")
        if CRE == 4:
            out3 = np.zeros((depth, ML, xmitEvents))
            out2 = np.zeros((depth, ML, xmitEvents))
            out1 = np.zeros((depth, ML, xmitEvents))
            out0 = np.zeros((depth, ML, xmitEvents))
        elif CRE == 3:
            out2 = np.zeros((depth, ML, xmitEvents))
            out1 = np.zeros((depth, ML, xmitEvents))
            out0 = np.zeros((depth, ML, xmitEvents))
        elif CRE == 2:
            out1 = np.zeros((depth, ML, xmitEvents))
            out0 = np.zeros((depth, ML, xmitEvents))
        elif CRE == 1:
            out0 = np.zeros((depth, ML, xmitEvents))
        
        if ((CRE != 1) and (CRE != 2) and (CRE != 4)):
            logging.warning(f"No sort list for CRE={CRE}")
        
        # Determine ML_SortList based on Stride and CRE
        logging.debug(f"Determining ML_SortList for Stride={Stride}, CRE={CRE}")
        if Stride == 128:
            ML_SortList = list(range(128))
        elif Stride == 32:
            if CRE == 4:
                ML_SortList = [4, 4, 5, 5, 6, 6, 7, 7, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]
            else:
                ML_SortList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        elif Stride == 16:
            if CRE == 1:
                ML_SortList = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
            elif CRE == 2:
                ML_SortList = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
            elif CRE == 4:
                ML_SortList = [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]
        elif Stride == 12:
            if CRE ==1:
                ML_SortList = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
            elif CRE == 2:
                ML_SortList = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
            elif CRE == 4:
                ML_SortList = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
        elif Stride == 8:
            if CRE == 1:
                ML_SortList = [0, 2, 4, 6, 1, 3, 5, 7]
            elif CRE == 2:
                ML_SortList = [0, 1, 2, 3, 0, 1, 2, 3]
            elif CRE == 4:
                ML_SortList = [0, 0, 1, 1, 0, 0, 1, 1]
        elif Stride == 4:
            if CRE == 1:
                ML_SortList = [0, 2, 1, 3]
            elif CRE == 2:
                ML_SortList = [0, 1, 0, 1]
            elif CRE == 4:
                ML_SortList = [0, 0, 0, 0]
        elif Stride == 2:
            if CRE == 1:
                ML_SortList = [0, 1]
            elif CRE == 2:
                ML_SortList = [0, 0]
            elif CRE == 4:
                ML_SortList = [0, 0]
        else:
            logging.warning(f"No sort list for Stride={Stride}")
            ML_SortList = []
        
        logging.debug(f"Using ML_SortList: {ML_SortList}")
        
        if ((ML-1)>max(ML_SortList)) or (CRE == 4 and Stride < 16) or (CRE == 2 and Stride < 4):
            logging.warning("Captured ML is insufficient, some ML were not captured")
        
        # Sort 
        logging.debug("Performing RF data sorting...")
        for k in range(ML):
            iML = np.where(ML_SortList == MLs[k])[0]
            out0[:depth, k, :] = RFinput[np.arange(iML[0],(depth*Stride), Stride)]
            if CRE == 2:
                out1[:depth, k, :] = RFinput[np.arange(iML[1], (depth*Stride), Stride), :]
                out2[:depth,k,:] = RFinput[np.arange(iML[1], (depth*Stride), Stride), :]
                out3[:depth,k,:] = RFinput[np.arange(iML[1], (depth*Stride), Stride), :]
            elif CRE == 4:
                out2[:depth, k, :] = RFinput[np.arange(iML[2], (depth*Stride), Stride), :]
                out3[:depth, k, :] = RFinput[np.arange(iML[3], (depth*Stride), Stride), :]

        logging.info(f"RF sorting complete - output shape: {out0.shape}")
        return out0, out1, out2, out3

    ###################################################################################
    
    def _parse_data_f(self, rawrfdata, headerInfo):
        """Parse RF data for Fusion systems."""
        logging.info('Entering parseDataF - parsing Fusion RF data')
        # Definitions
        minNeg = 2**18 # Used to convert integers to 2's complement

        # Find header clumps
        # iHeader pts to the index of the header clump
        # Note that each Header is exactly 1 "Clump" long
        iHeader = np.array(np.where((rawrfdata[0,:] & 1572864)==524288))[0]
        numHeaders = len(iHeader) - 1 # Ignore last header as it is a part of a partial line
        logging.info(f"Found {numHeaders} headers in Fusion data")

        # Get maximum number of samples between consecutive headers
        maxNumSamples = 0
        for m in range(numHeaders):
            tempMax = iHeader[m+1] - iHeader[m] - 1
            if (tempMax > maxNumSamples):
                maxNumSamples = tempMax
        
        numSamples = maxNumSamples*12
        logging.debug(f"Maximum samples between headers: {maxNumSamples}, total samples: {numSamples}")

        # Preallocate arrays
        lineData = np.zeros((numSamples, numHeaders), dtype = np.int32)
        lineHeader = np.zeros((numSamples, numHeaders), dtype = np.uint8)
        logging.debug(f"Preallocated arrays - lineData: {lineData.shape}, lineHeader: {lineHeader.shape}")

        # Extract data
        logging.info("Extracting line data from headers...")
        for m in range(numHeaders):
            if m % 1000 == 0:
                logging.debug(f"Processing header {m}/{numHeaders}")
                
            iStartData = iHeader[m]+2
            iStopData = iHeader[m+1]-1

            if headerInfo.Data_Type[m] == float(0x5a):
                # set stop data to a reasonable value to keep file size form blowing up
                iStopData = iStartData + 10000
                logging.debug(f"Header {m} is push pulse (0x5a), limiting data size")
            
            # Get Data for current line and convert to 2's complement values
            lineData_u32 = rawrfdata[:12,iStartData:iStopData+1]
            lineData_s32 = np.int32(lineData_u32&524287)
            iNeg = np.where(lineData_s32 >= minNeg)
            lineData_s32[iNeg] -= (2*minNeg) # type: ignore
            lineHeader_u8 = (lineData_u32 & 1572864) >> 19

            lineData[:lineData_s32.size,m] = lineData_s32.ravel(order='F')
            lineHeader[:lineHeader_u8.size,m] = lineHeader_u8.ravel(order='F')

        logging.info(f'Exiting parseDataF - lineData shape: {lineData.shape}, lineHeader shape: {lineHeader.shape}')
        return lineData, lineHeader

    ###################################################################################
    
    def _parse_header_v(self, rawrfdata):
        """Parse header for Voyager systems."""
        logging.info("Parsing Voyager header information")
        temp_headerInfo = HeaderInfoStruct()

        iHeader = np.where(np.uint8(rawrfdata[2,0,:])&224)
        numHeaders = len(iHeader)-1 # Ignore last header as it is part of a partial line
        logging.debug(f"Found {numHeaders} headers in Voyager data")

        # Get infor for each header
        for m in range(numHeaders):
            if m % 1000 == 0:
                logging.debug(f"Processing Voyager header {m}/{numHeaders}")
                
            packedHeader = ''
            for k in np.arange(11,0,-1):
                temp = ''
                for i in np.arange(2,0,-1):
                    temp += bin(np.uint8(rawrfdata[i,k,iHeader[m]]))

                # Discard first 3 bits, redundant info
                packedHeader += temp[3:24]

            iBit = 0
            temp_headerInfo.RF_CaptureVersion[m] = int(packedHeader[iBit:iBit+4],2)
            iBit += 4
            temp_headerInfo.Tap_Point[m] = int(packedHeader[iBit:iBit+3],2)
            iBit += 3
            temp_headerInfo.Data_Gate[m] = int(packedHeader[iBit],2)
            iBit += 1
            temp_headerInfo.Multilines_Capture[m] = int(packedHeader[iBit:iBit+4],2)
            iBit += 4
            temp_headerInfo.RF_Sample_Rate[m] = int(packedHeader[iBit],2)
            iBit += 1
            temp_headerInfo.Steer[m] = int(packedHeader[iBit:iBit+6],2)
            iBit += 6
            temp_headerInfo.elevationPlaneOffset[m] = int(packedHeader[iBit:iBit+8],2)
            iBit += 8
            temp_headerInfo.PM_Index[m] = int(packedHeader[iBit:iBit+2],2)
            iBit += 2
            temp_headerInfo.Line_Index[m] = int(packedHeader[iBit:iBit+16],2)
            iBit += 16
            temp_headerInfo.Pulse_Index[m] = int(packedHeader[iBit:iBit+16],2)
            iBit += 16
            temp_headerInfo.Data_Format[m] = int(packedHeader[iBit:iBit+16],2)
            iBit += 16
            temp_headerInfo.Data_Type[m] = int(packedHeader[iBit:iBit+16],2)
            iBit += 16
            temp_headerInfo.Header_Tag[m] = int(packedHeader[iBit:iBit+16],2)
            iBit += 16
            temp_headerInfo.Threed_Pos[m] = int(packedHeader[iBit:iBit+16],2)
            iBit += 16
            temp_headerInfo.Mode_Info[m] = int(packedHeader[iBit:iBit+16],2)
            iBit += 16
            temp_headerInfo.Frame_ID[m] = int(packedHeader[iBit:iBit+32],2)
            iBit += 32
            temp_headerInfo.CSID[m] = int(packedHeader[iBit:iBit+16],2)
            iBit += 16
            temp_headerInfo.Line_Type[m] = int(packedHeader[iBit:iBit+16],2)
            iBit += 16
            temp_headerInfo.Time_Stamp = int(packedHeader[iBit:iBit+32],2)

        logging.info(f"Voyager header parsing complete - processed {numHeaders} headers")
        return temp_headerInfo

    ###################################################################################
    
    def _get_filler_zeros(self, num):
        """Get string of zeros for padding."""
        zeros = "0"
        num -= 1
        while num > 0:
            zeros += "0"
            num -= 1
        return zeros

    ###################################################################################
    
    def _parse_header_f(self, rawrfdata):
        """Parse header for Fusion systems."""
        logging.info('Entering parseHeaderF - parsing Fusion headers')
        # Find header clumps
        # iHeader pts to the index of the header clump
        # Note that each header is exactly 1 "Clump" long
        iHeader = np.array(np.where(rawrfdata[0,:]&1572864 == 524288))[0]
        numHeaders: int = iHeader.size - 1 # Ignore last header as it is a part of a partial line
        logging.info(f"Found {numHeaders} headers in Fusion data")

        HeaderInfo = HeaderInfoStruct()

        # Initialize header arrays
        logging.debug("Initializing header arrays...")
        HeaderInfo.RF_CaptureVersion = np.zeros(numHeaders, dtype=np.uint8)
        HeaderInfo.Tap_Point = np.zeros(numHeaders, np.uint8)
        HeaderInfo.Data_Gate = np.zeros(numHeaders, np.uint8)
        HeaderInfo.Multilines_Capture = np.zeros(numHeaders, np.uint8)
        HeaderInfo.RF_Sample_Rate = np.zeros(numHeaders, np.uint8)
        HeaderInfo.Steer = np.zeros(numHeaders, np.uint8)
        HeaderInfo.elevationPlaneOffset = np.zeros(numHeaders, np.uint8)
        HeaderInfo.PM_Index = np.zeros(numHeaders, np.uint8)
        HeaderInfo.Line_Index = np.zeros(numHeaders, np.uint16)
        HeaderInfo.Pulse_Index = np.zeros(numHeaders, np.uint16)
        HeaderInfo.Data_Format = np.zeros(numHeaders, np.uint16)
        HeaderInfo.Data_Type = np.zeros(numHeaders, np.uint16)
        HeaderInfo.Header_Tag = np.zeros(numHeaders, np.uint16)
        HeaderInfo.Threed_Pos = np.zeros(numHeaders, np.uint16)
        HeaderInfo.Mode_Info = np.zeros(numHeaders, np.uint16)
        HeaderInfo.Frame_ID = np.zeros(numHeaders, np.uint32)
        HeaderInfo.CSID = np.zeros(numHeaders, np.uint16)
        HeaderInfo.Line_Type = np.zeros(numHeaders, np.uint16)
        HeaderInfo.Time_Stamp = np.zeros(numHeaders, np.uint32)

        # Get info for Each Header
        logging.info("Extracting header information...")
        for m in range(numHeaders):
            if m % 1000 == 0:
                logging.debug(f"Processing Fusion header {m}/{numHeaders}")
                
            packedHeader = bin(rawrfdata[12, iHeader[m]])[2:]
            remainingZeros = 4 - len(packedHeader)
            if remainingZeros > 0:
                zeros = self._get_filler_zeros(remainingZeros)
                packedHeader = str(zeros + packedHeader)
            for i in np.arange(11,-1,-1):
                curBin = bin(int(rawrfdata[i,iHeader[m]]))[2:]
                remainingZeros = 21 - len(curBin)
                if remainingZeros > 0:
                    zeros = self._get_filler_zeros(remainingZeros)
                    curBin = str(zeros + curBin)
                packedHeader += curBin

            # Parse packed header bits
            iBit = 2
            HeaderInfo.RF_CaptureVersion[m] = int(packedHeader[iBit:iBit+4], 2)
            iBit += 4
            HeaderInfo.Tap_Point[m] = int(packedHeader[iBit:iBit+3], 2)
            iBit += 3
            HeaderInfo.Data_Gate[m] = int(packedHeader[iBit], 2)
            iBit += 1
            HeaderInfo.Multilines_Capture[m] = int(packedHeader[iBit:iBit+4], 2)
            iBit += 4
            iBit += 15 # Waste 15 bits (unused)
            HeaderInfo.RF_Sample_Rate[m] = int(packedHeader[iBit], 2)
            iBit += 1
            HeaderInfo.Steer[m] = int(packedHeader[iBit:iBit+6], 2)
            iBit += 6
            HeaderInfo.elevationPlaneOffset[m] = int(packedHeader[iBit:iBit+8], 2)
            iBit += 8
            HeaderInfo.PM_Index[m] = int(packedHeader[iBit:iBit+2], 2)
            iBit += 2
            HeaderInfo.Line_Index[m] = int(packedHeader[iBit:iBit+16], 2)
            iBit += 16
            HeaderInfo.Pulse_Index[m] = int(packedHeader[iBit:iBit+16], 2)
            iBit += 16
            HeaderInfo.Data_Format[m] = int(packedHeader[iBit:iBit+16], 2)
            iBit += 16
            HeaderInfo.Data_Type[m] = int(packedHeader[iBit: iBit+16], 2)
            iBit += 16
            HeaderInfo.Header_Tag[m] = int(packedHeader[iBit:iBit+16], 2)
            iBit += 16
            HeaderInfo.Threed_Pos[m] = int(packedHeader[iBit:iBit+16], 2)
            iBit += 16
            HeaderInfo.Mode_Info[m] = int(packedHeader[iBit:iBit+16], 2)
            iBit += 16
            HeaderInfo.Frame_ID[m] = int(packedHeader[iBit:iBit+32], 2)
            iBit += 32
            HeaderInfo.CSID[m] = int(packedHeader[iBit:iBit+16], 2)
            iBit += 16
            HeaderInfo.Line_Type[m] = int(packedHeader[iBit:iBit+16], 2)
            iBit += 16
            HeaderInfo.Time_Stamp[m] = int(str(packedHeader[iBit:iBit+13]+packedHeader[iBit+15:iBit+34]), 2)
            
        logging.info(f'Exiting parseHeaderF - numHeaders: {numHeaders}, Data_Type shape: {HeaderInfo.Data_Type.shape}')
        return HeaderInfo

    ###################################################################################
    
    def _parse_data_v(self, rawrfdata, headerInfo):
        """Parse RF data for Voyager systems."""
        logging.info("Parsing Voyager RF data")
        # Definitions
        minNeg = 16*(2^16) # Used to convert offset integers to 2's complement

        # Find header clumps
        # iHeader pts to the index of the header clump
        # Note that each Header is exactly 1 "Clump" long
        iHeader = np.where(rawrfdata[2,0,:]&224==64)
        numHeaders = len(iHeader)-1 # Ignore last header as it is a part of a partial line
        numSamples = (iHeader[1]-iHeader[0]-1)*12
        logging.debug(f"Voyager data: {numHeaders} headers, {numSamples} samples per line")
        
        # Preallocate arrays
        lineData = np.zeros((numSamples, numHeaders), dtype = np.int32)
        lineHeader = np.zeros((numSamples, numHeaders), dtype = np.uint8)

        # Extract data
        logging.info("Extracting Voyager line data...")
        for m in range(len(numHeaders)):
            if m % 1000 == 0:
                logging.debug(f"Processing Voyager line {m}/{numHeaders}")

            # Get data in between headers
            iStartData = iHeader[m]+1
            iStopData = iHeader[m+1]-1

            # Push pulses (DT 0x5a) are very long, and have no valid RX data
            if headerInfo.Data_Type[m] == float(0x5a):
                # set stop data to a reasonable value to keep file size from blowing up
                iStopData = iStartData+10000
                logging.debug(f"Line {m} is push pulse, limiting data size")
            
            # Get Data for current line and convert to 2's complement values
            lineData_u8 = rawrfdata[:,:,iStartData:iStopData]
            lineData_s32 = np.int32(lineData_u8[0,:,:])+np.int32(lineData_u8[1,:,:])*2^8+np.int32(lineData_u8[2,:,:]&np.uint8(31))*2^16
            iNeg = np.where(lineData_s32>=minNeg)
            lineData_s32[iNeg] = lineData_s32[iNeg]-2*minNeg
            lineHeader_u8 = (lineData_u8[2,:,:]&224)>>6

            lineData[:lineData_s32.size-1,m] = lineData_s32[:lineData_s32.size-1]
            lineHeader[:lineHeader_u8.size-1,m] = lineHeader_u8[:lineHeader_u8.size-1]

        logging.info(f"Voyager data parsing complete - lineData: {lineData.shape}, lineHeader: {lineHeader.shape}")
        return lineData, lineHeader

    ###################################################################################
    
    def _parse_file_header(self, file_obj, endianness):
        """Parse file header information."""
        logging.info("Parsing file header information")
        fileVersion = int.from_bytes(file_obj.read(4), endianness, signed=False)
        numFileHeaderBytes = int.from_bytes(file_obj.read(4), endianness, signed=False)
        logging.info(f"File Version: {fileVersion}, Header Size: {numFileHeaderBytes} bytes")

        # Handle accordingly to fileVersion
        temp_dbParams = dbParams()
        logging.debug(f"Reading file header for version {fileVersion}")
        
        if fileVersion == 2:
            logging.debug("Reading file header version 2")
            temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14*11)], (14, 11), order='F')
            temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.numSubVols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
            temp_dbParams.numPlanes = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
            temp_dbParams.zigZagEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

        elif fileVersion == 3:
            logging.debug("Reading file header version 3")
            temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14*11)],(14,11), order='F')
            temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.numSubVols = int.from_bytes(file_obj.read(4), endianness, signed=False)
            temp_dbParams.numPlanes = int.from_bytes(file_obj.read(4), endianness, signed=False)
            temp_dbParams.zigZagEnabled = int.from_bytes(file_obj.read(4), endianness, signed=False)

            temp_dbParams.multiLineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.linesPerEnsCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.ensPerSeqCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.numCfCols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14)]
            temp_dbParams.numCfEntries = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]
            temp_dbParams.numCfDummies = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(4)]

        elif fileVersion == 4:
            logging.debug("Reading file header version 4")
            temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

            temp_dbParams.azimuthMultilineFactorXbrOutCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.azimuthMultilineFactorXbrInCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

            temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

            temp_dbParams.elevationMultilineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

            temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14*11)],(14,11), order='F')
            temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numSubVols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

            temp_dbParams.Planes = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

            temp_dbParams.zigZagEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

            temp_dbParams.linesPerEnsCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.ensPerSeqCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numCfCols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14)]
            temp_dbParams.numCfEntries = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numCfDummies = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        elif fileVersion == 5:
            logging.debug("Reading file header version 5")
            temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

            temp_dbParams.azimuthMultilineFactorXbrOutCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.azimuthMultilineFactorXbrInCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

            temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

            temp_dbParams.elevationMultilineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.multiLineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

            temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14*11)],(14,11), order='F')
            temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numSubVols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

            temp_dbParams.numPlanes = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

            temp_dbParams.zigZagEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]

            temp_dbParams.linesPerEnsCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.ensPerSeqCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numCfCols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14)]
            temp_dbParams.numCfEntries = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numCfDummies = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        elif fileVersion == 6:
            logging.debug("Reading file header version 6")
            temp_dbParams.tapPoint = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
            temp_dbParams.acqNumActiveScChannels2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.azimuthMultilineFactorXbrOut = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.azimuthMultilineFactorXbrIn = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.azimuthMultilineFactorXbrOutCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.azimuthMultilineFactorXbrInCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numOfSonoCTAngles2dActual = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.elevationMultilineFactor = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.elevationMultilineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.multiLineFactorCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numPiPulses = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.num2DCols = np.reshape([int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14*11)],(14,11), order='F')
            temp_dbParams.fastPiEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numZones2d = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numSubVols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
            temp_dbParams.numPlanes = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
            temp_dbParams.zigZagEnabled = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(1)]
            temp_dbParams.linesPerEnsCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.ensPerSeqCf = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numCfCols = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(14)]
            temp_dbParams.numCfEntries = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]
            temp_dbParams.numCfDummies = [int.from_bytes(file_obj.read(4), endianness, signed=False) for i in range(3)]

        else:
            numFileHeaderBytes = 0
            logging.warning(f"Unknown file version: {fileVersion}")

        logging.info(f"File header parsing complete for version {fileVersion}")
        return temp_dbParams, numFileHeaderBytes

    ###################################################################################
    
    def _parse_rf_data(self, rawrfdata, headerInfo, isVoyager):
        """Parse RF signal data."""
        logging.info("Parsing RF signal data...")
        Tap_Point = headerInfo.Tap_Point[0]
        logging.debug(f"Tap Point: {Tap_Point}, isVoyager: {isVoyager}")
        
        if isVoyager:
            lineData, lineHeader = self._parse_data_v(rawrfdata, headerInfo)
        else: # isFusion
            lineData, lineHeader = self._parse_data_f(rawrfdata, headerInfo)
            Tap_Point = headerInfo.Tap_Point[0]
            if Tap_Point == 0: # Correct for MS 19 bits of 21 real data bits
                logging.debug("Applying bit shift correction for Tap Point 0")
                lineData = lineData << 2
        
        # After parsing lineData, log a sample of the first and last 20 rows for a nonzero column
        nonzero_col = None
        for col in range(lineData.shape[1]):
            if np.any(lineData[:, col] != 0):
                nonzero_col = col
                break
        if nonzero_col is not None:
            logging.info(f"First 20 values of lineData[:, {nonzero_col}]: {lineData[:20, nonzero_col]}")
            logging.info(f"Last 20 values of lineData[:, {nonzero_col}]: {lineData[-20:, nonzero_col]}")
            logging.info(f"Min: {lineData[:, nonzero_col].min()}, Max: {lineData[:, nonzero_col].max()}")
        else:
            logging.warning("No nonzero columns found in lineData!")
        
        logging.info(f"RF data parsing complete - lineData: {lineData.shape}, lineHeader: {lineHeader.shape}")
        return lineData, lineHeader, Tap_Point

    ###################################################################################
    
    def _parse_rf(self, filepath: str, readOffset: int, readSize: int) -> Rfdata:
        """Open and parse RF data file."""
        # Remember to make sure .c files have been compiled before running
        logging.info(f"Opening RF file: {filepath}")
        logging.debug(f"Read parameters - offset: {readOffset}MB, size: {readSize}MB")

        rfdata = Rfdata()
        file_obj = open(filepath, 'rb')

        # Voyager or Fusion?
        VHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 160, 160]
        FHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 11, 11 ]
        fileHeaderSize = len(VHeader)
        
        fileHeader = list(file_obj.read(fileHeaderSize))
        isVoyager = False
        hasFileHeader = False

        # Determine file type
        if fileHeader == VHeader:
            logging.info("Header information found - Parsing Voyager RF capture file")
            isVoyager = True
            hasFileHeader = True
        elif fileHeader == FHeader:
            logging.info("Header information found - Parsing Fusion RF capture file")
            hasFileHeader = True
        else: # Legacy V-ACB file
            logging.info("No header found - Parsing legacy Voyager RF capture file")
            isVoyager = True

        # Load RAW RF data
        start_time = datetime.now()
        logging.info("Starting raw RF data loading...")

        # Read out file header info
        endianness = 'little'
        if hasFileHeader:
            if isVoyager:
                endianness = 'big'      
                logging.debug("Using big-endian for Voyager file")
            else:
                logging.debug("Using little-endian for Fusion file")

            [rfdata.dbParams, numFileHeaderBytes] = self._parse_file_header(file_obj, endianness)
            totalHeaderSize = fileHeaderSize+8+numFileHeaderBytes # 8 bytes from fileVersion and numFileHeaderBytes
            logging.debug(f"Total header size: {totalHeaderSize} bytes")
        else:
            totalHeaderSize = 0
            logging.debug("No file header to parse")
        
        readOffset *= (2**20)
        remainingSize = os.stat(filepath).st_size - totalHeaderSize
        readSize *= (2**20)
        logging.info(f"File size: {os.stat(filepath).st_size} bytes, remaining after header: {remainingSize} bytes")

        if isVoyager:
            logging.debug("Processing Voyager file format")
            # Align read offset and size
            alignment = np.arange(0,remainingSize+1,36)
            offsetDiff = alignment - readOffset
            readDiff = alignment - readSize
            readOffset = alignment[np.where(offsetDiff >= 0)[0][0]].__int__()
            readSize = alignment[np.where(readDiff >= 0)[0][0]].__int__()
            logging.debug(f"Aligned Voyager read - offset: {readOffset}, size: {readSize}")
            
            # Start reading
            rawrfdata = open(filepath,'rb').read(readSize.__int__())
        
        else: # isFusion
            logging.debug("Processing Fusion file format")
            # Align read and offset size
            alignment = np.arange(0,remainingSize+1,32)
            offsetDiff = alignment - readOffset
            readDiff = alignment - readSize 

            matchingIndices = np.where(offsetDiff >= 0)[0]
            if len(matchingIndices) > 0:
                readOffset = alignment[matchingIndices[0]].__int__()
            else:
                readOffset = 0

            matchingIndices = np.where(readDiff >= 0)[0]
            if len(matchingIndices) > 0:
                readSize = alignment[matchingIndices[0]].__int__()
            else:
                readSize = remainingSize
            numClumps = int(np.floor(readSize/32)) # 256 bit clumps
            logging.debug(f"Aligned Fusion read - offset: {readOffset}, size: {readSize}, clumps: {numClumps}")

            offset = totalHeaderSize+readOffset
            partA = self._call_get_part_a(numClumps, filepath, offset)
            partB = self._call_get_part_b(numClumps, filepath, offset)
            rawrfdata = np.concatenate((partA, partB))
            logging.debug(f"Raw RF data shape: {rawrfdata.shape}")

        # Reshape Raw RF Data
        if isVoyager:
            logging.info("Reshaping Voyager raw RF data...")
            numClumps = np.floor(len(rawrfdata)/36) # 1 Clump = 12 Samples (1 Sample = 3 bytes)
            logging.debug(f"Voyager clumps to process: {numClumps}")

            rlimit = 180000000 # Limit ~172 MB for reshape workload, otherwise large memory usage
            if len(rawrfdata)>rlimit:
                logging.warning(f"Large file detected ({len(rawrfdata)} bytes), chunking reshape operation")
                numChunks = np.floor(len(rawrfdata/rlimit))
                numremBytes = np.mod(len(rawrfdata),rlimit)
                numClumpGroup = rlimit/36

                temp = np.zeros((numChunks+1,3,12,numClumpGroup))
                m=1
                n=1
                # Reshape array into clumps 
                for i in range(numChunks):
                    temp[i]=np.reshape(rawrfdata[m:m+rlimit],(3,12,numClumpGroup))
                    m += rlimit
                    n += numClumpGroup
                
                # Handle the remaining bytes
                if numremBytes > 0:
                    temp[numChunks]=np.reshape(rawrfdata[m:numClumps*36+1], (3,12,numClumps-n+1))

                # Combine the reshaped arrays
                rawrfdata = np.concatenate((temp[:]),axis=2)
                logging.debug("Chunked reshape complete")
            else:
                logging.debug("Direct reshape for normal size file")
        
        elapsed_time = datetime.now() - start_time
        logging.info(f"Raw data loading elapsed time: {elapsed_time}")

        # Parse Header
        logging.info("Parsing header info...")
        # Extract header info
        if isVoyager:
            headerInfo = self._parse_header_v(rawrfdata)
        else: # isFusion
            headerInfo = self._parse_header_f(rawrfdata)

        elapsed_time = datetime.now() - start_time
        logging.info(f"Header parsing elapsed time: {elapsed_time}")

        # Parse RF signal data
        lineData, lineHeader, Tap_Point = self._parse_rf_data(rawrfdata, headerInfo, isVoyager)
        logging.info(f"Signal data parsed - lineData shape: {lineData.shape}, lineHeader shape: {lineHeader.shape}")
        # Debug: print stats of lineData and lineHeader
        logging.info(f"lineData stats: min={np.min(lineData)}, max={np.max(lineData)}, mean={np.mean(lineData)}")
        logging.info(f"lineHeader stats: min={np.min(lineHeader)}, max={np.max(lineHeader)}, mean={np.mean(lineHeader)}")

        # Pack data
        rfdata.lineData = lineData
        rfdata.lineHeader = lineHeader
        rfdata.headerInfo = headerInfo

        # Free-up Memory
        del rawrfdata
        logging.debug("Raw RF data memory freed")

        # Sort into Data Types
        # De-interleave rfdata
        logging.info("Organizing based on data type...")

        # Print detailed data type information
        if hasattr(rfdata.headerInfo, 'Data_Type'):
            unique_types = np.unique(rfdata.headerInfo.Data_Type)
            logging.info(f"Found data types: {unique_types}")
            for dtype in unique_types:
                count = np.sum(rfdata.headerInfo.Data_Type == dtype)
                logging.info(f"Data type {dtype}: {count} occurrences")

        # Define data type categories
        DataType_ECHO = np.arange(1,15)
        DataType_EchoMMode = 26
        DataType_COLOR = [17, 21, 22, 23, 24]
        DataType_ColorMMode = [27, 28]
        DataType_ColorTDI = 24
        DataType_CW = 16
        DataType_PW = [18,19]
        DataType_Dummy = [20,25,29,30,31]
        DataType_SWI = [90,91]
        DataType_Misc = [15,88,89]

        # Determine ML_Capture
        logging.debug("Determining ML_Capture parameter...")
        if Tap_Point == 7:
            ML_Capture = 128
            logging.debug("Using ML_Capture=128 for Tap Point 7")
        else:
            ML_Capture = np.double(rfdata.headerInfo.Multilines_Capture[0])
            logging.debug(f"Using ML_Capture={ML_Capture} from header")
        
        if ML_Capture == 0:
            SAMPLE_RATE = np.double(rfdata.headerInfo.RF_Sample_Rate[0])
            if SAMPLE_RATE == 0:
                ML_Capture = 16
                logging.debug("Using ML_Capture=16 (default for 0 sample rate)")
            else: # 20MHz Capture
                ML_Capture = 32
                logging.debug("Using ML_Capture=32 (20MHz capture)")

        Tap_Point = rfdata.headerInfo.Tap_Point[0]
        if Tap_Point == 7: #Hardware is saving the tap point as 7 and now we convert it back to 4
            Tap_Point = 4
            logging.debug("Converted Tap Point from 7 to 4")
        namePoint = ['PostShepard', 'PostAGNOS', 'PostXBR', 'PostQBP', 'PostADC']
        logging.info(f"Tap Point: {namePoint[Tap_Point]}, Capture ML: {ML_Capture}x")

        xmitEvents = len(rfdata.headerInfo.Data_Type)
        logging.debug(f"Total transmit events: {xmitEvents}")

        # Find Echo Data - Modified to handle type 1 as echo data
        logging.info("Processing echo data...")
        echo_index = np.zeros(xmitEvents).astype(np.int32)
        for i in range(len(DataType_ECHO)):
            index = ((rfdata.headerInfo.Data_Type & 255) == DataType_ECHO[i]) # Find least significant byte
            echo_index = np.bitwise_or(np.array(echo_index), np.array(index).astype(np.int32))

        # If no echo data found but we have type 1, treat it as echo data
        if np.sum(echo_index) == 0 and np.any(rfdata.headerInfo.Data_Type == 1):
            logging.info("No standard echo data found, but type 1 data present. Treating as echo data.")
            echo_index = (rfdata.headerInfo.Data_Type == 1).astype(np.int32)

        if np.sum(echo_index) > 0:
            logging.info(f"Found {np.sum(echo_index)} echo data events")
            # Remove false gate data at the beginning of the line
            columnsToDelete =  np.where(echo_index==0)
            pruningLineData = np.delete(rfdata.lineData, columnsToDelete, axis=1)
            pruningLineHeader = np.delete(rfdata.lineHeader, columnsToDelete, axis=1)
            logging.debug(f"After pruning: lineData shape {pruningLineData.shape}")
            
            if Tap_Point == 4:
                echoData = pruningLineData
                logging.debug("Using direct lineData for Tap Point 4")
            else:
                echoData = self._prune_data(pruningLineData, pruningLineHeader, ML_Capture)
                logging.debug(f"Pruned echo data shape: {echoData.shape}")
                
            #pre-XBR Sort
            if Tap_Point == 0 or Tap_Point == 1:
                ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrIn[0]*rfdata.dbParams.elevationMultilineFactor[0]
                logging.info(f"Pre-XBR sort - Echo ML: {ML_Actual}x")
                CRE = 1
                rfdata.echoData = self._sort_rf(echoData, ML_Capture, ML_Actual, CRE, isVoyager)

            elif Tap_Point == 2: # post-XBR Sort
                ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrOut[0]*rfdata.dbParams.elevationMultilineFactor[0]
                logging.info(f"Post-XBR sort - Echo ML: {ML_Actual}x")
                CRE = rfdata.dbParams.acqNumActiveScChannels2d[0]
                logging.info(f"CRE: {CRE}")
                rfdata.echoData = self._sort_rf(echoData, ML_Capture, ML_Actual, CRE, isVoyager)
                
            elif Tap_Point == 4: # post-ADC sort
                ML_Actual = 128
                logging.info(f"Post-ADC sort - Echo ML: {ML_Actual}x")
                CRE = 1
                rfdata.echoData = self._sort_rf(echoData, ML_Actual, ML_Actual, CRE, isVoyager)

            else:
                logging.warning("Do not know how to sort this data set")

        # Find Echo MMode Data
        logging.info("Processing echo MMode data...")
        echoMMode_index = rfdata.headerInfo.Data_Type == DataType_EchoMMode
        if np.sum(echoMMode_index) > 0:
            logging.info(f"Found {np.sum(echoMMode_index)} echo MMode events")
            echoMModeData = self._prune_data(rfdata.lineData[:,echoMMode_index], rfdata.lineHeader[:,echoMMode_index], ML_Capture)
            ML_Actual = 1
            logging.info(f"Echo MMode ML: {ML_Actual}x")
            CRE = 1
            rfdata.echoMModeData = self._sort_rf(echoMModeData, ML_Capture, ML_Actual, CRE, isVoyager)

        # Find color data
        logging.info("Processing color data...")
        color_index = np.zeros(xmitEvents).astype(bool)
        for i in range(len(DataType_COLOR)):
            index = rfdata.headerInfo.Data_Type == DataType_COLOR[i]
            color_index = np.bitwise_or(color_index, index)
        
        if (sum(color_index)>0):
            logging.info(f"Found {np.sum(color_index)} color data events")
            colorData = self._prune_data(rfdata.lineData[:,color_index], rfdata.lineHeader[:,color_index], ML_Capture)
            if (Tap_Point == 0 or Tap_Point == 1):
                ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrInCf*rfdata.dbParams.elevationMultilineFactorCf
            else:
                ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrOutCf*rfdata.dbParams.elevationMultilineFactorCf
            logging.info(f"Color ML: {ML_Actual}x")
            CRE = 1
            rfdata.colorData = self._sort_rf(colorData, ML_Capture, ML_Actual, CRE, isVoyager)

            pkt = rfdata.dbParams.linesPerEnsCf
            nlv = rfdata.dbParams.ensPerSeqCf
            grp = rfdata.dbParams.numCfCols/rfdata.dbParams.ensPerSeqCf
            depth = rfdata.colorData.shape[0]
            logging.debug(f"Color data parameters - pkt: {pkt}, nlv: {nlv}, grp: {grp}, depth: {depth}")

            # Extract and rearrange flow RF data
            frm = np.floor(rfdata.colorData.shape[2]/(nlv*pkt*grp)) # whole frames
            if frm == 0:
                logging.warning("Cannot fully parse color data. RF capture does not contain at least one whole color frame.")
                frm = 1
                grp = np.floor(rfdata.colorData.shape[2]/(nlv*pkt))
            rfdata.colorData = rfdata.colorData[:,:,0:pkt*nlv*grp*frm-1]
            rfdata.colorData = np.reshape(rfdata.colorData, [depth, ML_Actual, nlv, pkt, grp, frm])
            rfdata.colorData = np.transpose(rfdata.colorData, (0,3,1,2,4,5))
            logging.debug(f"Color data reshaped to: {rfdata.colorData.shape}")

        # Find Color MMode Data
        logging.info("Processing color MMode data...")
        colorMMode_index = np.zeros(xmitEvents).astype(bool)
        for i in range(len(DataType_ColorMMode)):
            index = rfdata.headerInfo.Data_Type == DataType_ColorMMode[i]
            colorMMode_index = np.bitwise_or(colorMMode_index, index)
        
        if sum(colorMMode_index) > 0:
            logging.info(f"Found {np.sum(colorMMode_index)} color MMode events")
            colorMModeData = self._prune_data(rfdata.lineData[:,colorMMode_index], rfdata.lineHeader[:,colorMMode_index], ML_Capture)
            ML_Actual = 1
            CRE = 1
            rfdata.colorMModeData = self._sort_rf(colorMModeData, ML_Capture, ML_Actual, CRE, isVoyager)
        
        # Find CW Doppler Data
        logging.info("Processing CW Doppler data...")
        cw_index = np.zeros(xmitEvents).astype(bool)
        index = rfdata.headerInfo.Data_Type == DataType_CW
        cw_index = np.bitwise_or(cw_index, index)

        if (sum(cw_index) > 0):
            logging.info(f"Found {np.sum(cw_index)} CW Doppler events")
            cwData = self._prune_data(rfdata.lineData[:,cw_index], rfdata.lineHeader[:,cw_index], ML_Capture)
            ML_Actual = 1
            CRE = 1
            rfdata.cwData = self._sort_rf(cwData, ML_Capture, ML_Actual, CRE, isVoyager)

        # Find PW Doppler Data
        logging.info("Processing PW Doppler data...")
        pw_index = np.zeros(xmitEvents).astype(bool)
        for i in range(len(DataType_PW)):
            index = rfdata.headerInfo.Data_Type == DataType_PW[i]
            pw_index = np.bitwise_or(pw_index, index)

        if (sum(pw_index) > 0):
            logging.info(f"Found {np.sum(pw_index)} PW Doppler events")
            pwData = self._prune_data(rfdata.lineData[:,pw_index], rfdata.lineHeader[:,pw_index], ML_Capture)
            ML_Actual = 1
            CRE = 1
            rfdata.pwData = self._sort_rf(pwData, ML_Capture, ML_Actual, CRE, isVoyager)

        # Find Dummy Data
        logging.info("Processing dummy data...")
        dummy_index = np.zeros(xmitEvents).astype(bool)
        for i in range(len(DataType_Dummy)):
            index = rfdata.headerInfo.Data_Type == DataType_Dummy[i]
            dummy_index = np.bitwise_or(dummy_index, index)

        if sum(dummy_index)>0:
            logging.info(f"Found {np.sum(dummy_index)} dummy data events")
            dummyData = self._prune_data(rfdata.lineData[:, dummy_index], rfdata.lineHeader[:, dummy_index], ML_Capture)
            ML_Actual = 2
            CRE = 1
            rfdata.dummyData = self._sort_rf(dummyData, ML_Capture, ML_Actual, CRE, isVoyager)

        # Find Shearwave Data
        logging.info("Processing shearwave data...")
        swi_index = np.zeros(xmitEvents).astype(bool)
        for i in range(len(DataType_SWI)):
            index = rfdata.headerInfo.Data_Type == DataType_SWI[i]
            swi_index = np.bitwise_or(swi_index, index)
        
        if sum(swi_index) > 0:
            logging.info(f"Found {np.sum(swi_index)} shearwave events")
            swiData = self._prune_data(rfdata.lineData[:,swi_index], rfdata.lineHeader[:,swi_index], ML_Capture)
            ML_Actual = ML_Capture
            CRE = 1
            rfdata.swiData = self._sort_rf(swiData, ML_Capture, ML_Actual, CRE, isVoyager)

        # Find Misc Data
        logging.info("Processing miscellaneous data...")
        misc_index = np.zeros(xmitEvents).astype(bool)
        for i in range(len(DataType_Misc)):
            index = rfdata.headerInfo.Data_Type == DataType_Misc[i]
            misc_index = np.bitwise_or(misc_index, index)
        
        if sum(misc_index) > 0:
            logging.info(f"Found {np.sum(misc_index)} miscellaneous events")
            miscData = self._prune_data(rfdata.lineData[:,misc_index], rfdata.lineHeader[:,misc_index], ML_Capture)
            ML_Actual = ML_Capture
            CRE = 1
            rfdata.miscData = self._sort_rf(miscData, ML_Capture, ML_Actual, CRE, isVoyager)

        elapsed_time = datetime.now() - start_time
        logging.info(f"Total RF parsing elapsed time: {elapsed_time}")

        # Clean up empty fields in struct
        logging.info("RF parsing complete")

        return rfdata

    ###################################################################################
    
    def _calculate_parameters(self):
        """Calculate and set main parameters as instance variables."""
        logging.info("Calculating parsing parameters...")
        self.txBeamperFrame = np.array(self.rfdata.dbParams.num2DCols).flat[0]
        self.NumSonoCTAngles = self.rfdata.dbParams.numOfSonoCTAngles2dActual[0]
        logging.info(f"Beam parameters - txBeamperFrame: {self.txBeamperFrame}, NumSonoCTAngles: {self.NumSonoCTAngles}")
        self.numFrame = int(np.floor(self.rfdata.lineData.shape[1] / (self.txBeamperFrame * self.NumSonoCTAngles)))
        self.multilinefactor = self.ML_in
        logging.info(f"Calculated numFrame: {self.numFrame}, multilinefactor: {self.multilinefactor}")
        
        # Auto-detect used_os and pt based on nonzero region in lineData
        col = 0
        if np.any(self.rfdata.lineData[:, col] != 0):
            first_nonzero = np.where(self.rfdata.lineData[:, col] != 0)[0][0]
            last_nonzero = np.where(self.rfdata.lineData[:, col] != 0)[0][-1]
            self.used_os = first_nonzero
            self.pt = int(np.floor((last_nonzero - first_nonzero + 1) / self.multilinefactor))
            logging.info(f"Auto-detected: used_os={self.used_os}, pt={self.pt}")
        else:
            # Fallback to original hardcoded values
            self.used_os = 2256 if self.used_os is None else self.used_os
            self.pt = int(np.floor((self.rfdata.lineData.shape[0] - self.used_os) / self.multilinefactor))
            logging.warning(f"Using fallback values: used_os={self.used_os}, pt={self.pt}")

    def _fill_data_arrays(self):
        """Fill RF data arrays for fundamental and harmonic signals."""
        logging.info("Filling RF data arrays for fundamental and harmonic signals...")
        rftemp_all_harm = np.zeros((self.pt, self.ML_out * self.txBeamperFrame))
        rftemp_all_fund = np.zeros((self.pt, self.ML_out * self.txBeamperFrame))
        rf_data_all_harm = np.zeros((self.numFrame, self.NumSonoCTAngles, self.pt, self.ML_out * self.txBeamperFrame))
        rf_data_all_fund = np.zeros((self.numFrame, self.NumSonoCTAngles, self.pt, self.ML_out * self.txBeamperFrame))
        logging.debug(f"Preallocated arrays shapes - fund: {rf_data_all_fund.shape}, harm: {rf_data_all_harm.shape}")

        for k0 in range(self.numFrame):
            if k0 % max(1, self.numFrame // 10) == 0:
                logging.info(f"Processing frame {k0+1}/{self.numFrame}")
            for k1 in range(self.NumSonoCTAngles):
                for k2 in range(self.txBeamperFrame):
                    bi = k0 * self.txBeamperFrame * self.NumSonoCTAngles + k1 * self.txBeamperFrame + k2
                    if bi >= self.rfdata.lineData.shape[1]:
                        logging.warning(f"Skipping bi={bi} as it exceeds lineData columns {self.rfdata.lineData.shape[1]}")
                        continue
                    idx0 = self.used_os + np.arange(self.pt * self.multilinefactor)
                    idx1 = bi
                    if k0 == 0 and k1 == 0 and k2 == 0:
                        logging.debug(f"First extraction - lineData[{idx0[0]}:{idx0[-1]+1}, {idx1}]")
                        logging.debug(f"lineData values sample: {self.rfdata.lineData[idx0, idx1][:10]}")
                    temp = np.transpose(
                        np.reshape(self.rfdata.lineData[idx0, idx1],
                                 (self.multilinefactor, self.pt), order='F')
                    )
                    if k0 == 0 and k1 == 0 and k2 == 0:
                        logging.debug(f"temp shape: {temp.shape}, temp sample: {temp.ravel()[:10]}")
                    # Harmonic: columns 0 and 2 as before
                    if temp.shape[1] > 2:
                        rftemp_all_harm[:, np.arange(self.ML_out) + (k2 * self.ML_out)] = temp[:, [0, 2]]
                    else:
                        logging.warning(f"temp has only {temp.shape[1]} columns, skipping harmonic assignment")
                    
                    # Fundamental: columns 9 and 11, with fallback to last two columns
                    if temp.shape[1] >= 12:
                        rftemp_all_fund[:, np.arange(self.ML_out) + (k2 * self.ML_out)] = temp[:, [9, 11]]
                    elif temp.shape[1] >= 2:
                        # Fallback: use last two columns
                        if k0 == 0 and k1 == 0 and k2 == 0:
                            logging.warning(f"temp has only {temp.shape[1]} columns, using last 2 columns for fundamental")
                        rftemp_all_fund[:, np.arange(self.ML_out) + (k2 * self.ML_out)] = temp[:, [-2, -1]]
                    else:
                        logging.warning(f"temp has only {temp.shape[1]} columns, skipping fundamental assignment")

                rf_data_all_harm[k0][k1] = rftemp_all_harm
                rf_data_all_fund[k0][k1] = rftemp_all_fund

        logging.info("RF data array filling complete")
        return rf_data_all_fund, rf_data_all_harm

    ###################################################################################
    
    def philipsRfParser(self, filepath: str, save_numpy: bool = False) -> np.ndarray:
        """Parse Philips RF data file, save as .mat file, and return shape of data.
        If save_numpy is True, only save the processed data as .npy files in a folder named 'sample_npy' in the sample path.
        If save_numpy is False, only save as .mat file."""
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Starting Philips RF parsing for file: {filepath}")
        logging.info(f"Save format: {'NumPy arrays' if save_numpy else 'MATLAB file'}")
        
        # Parse the RF data
        self.rfdata = self._parse_rf(filepath, 0, 2000)

        # Diagnostics for echoData
        logging.info("Performing echoData diagnostics...")
        logging.info(f"Type of rf.echoData: {type(self.rfdata.echoData) if hasattr(self.rfdata, 'echoData') else 'No echoData'}")
        if hasattr(self.rfdata, 'echoData'):
            if isinstance(self.rfdata.echoData, (list, tuple, np.ndarray)):
                if hasattr(self.rfdata.echoData, 'shape'):
                    logging.info(f"rf.echoData shape: {self.rfdata.echoData.shape}")
                elif isinstance(self.rfdata.echoData, (list, tuple)) and len(self.rfdata.echoData) > 0 and hasattr(self.rfdata.echoData[0], 'shape'):
                    logging.info(f"rf.echoData[0] shape: {self.rfdata.echoData[0].shape}")
                else:
                    logging.warning("rf.echoData is empty or not an array")
            else:
                logging.warning("rf.echoData is not a list/tuple/ndarray")

        # Use the first element if echoData is a tuple/list, else use as is
        echo_data_to_save = None
        if hasattr(self.rfdata, 'echoData'):
            if isinstance(self.rfdata.echoData, (list, tuple)) and len(self.rfdata.echoData) > 0:
                echo_data_to_save = self.rfdata.echoData[0]
                logging.debug("Using first element of echoData tuple/list")
            else:
                echo_data_to_save = self.rfdata.echoData
                logging.debug("Using echoData directly")

        if echo_data_to_save is None or (hasattr(echo_data_to_save, 'size') and echo_data_to_save.size == 0):
            error_msg = f"No echo data found in RF file. Data_Type values: {np.unique(self.rfdata.headerInfo.Data_Type) if hasattr(self.rfdata.headerInfo, 'Data_Type') else 'N/A'}. lineData shape: {self.rfdata.lineData.shape if hasattr(self.rfdata, 'lineData') else 'N/A'}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        # Data preprocessing
        logging.info("Performing data preprocessing...")
        if (self.rfdata.headerInfo.Line_Index[249] == self.rfdata.headerInfo.Line_Index[250]):
            logging.debug("Removing every other column starting from index 0")
            self.rfdata.lineData = self.rfdata.lineData[:,np.arange(2, self.rfdata.lineData.shape[1], 2)]
        else:
            logging.debug("Removing every other column starting from index 1")
            self.rfdata.lineData = self.rfdata.lineData[:,np.arange(1, self.rfdata.lineData.shape[1], 2)]

        # Calculate parameters
        self._calculate_parameters()

        # Fill data arrays
        rf_data_all_fund, rf_data_all_harm = self._fill_data_arrays()

        # Save data
        logging.info("Saving processed data...")
        if not save_numpy:
            destination = str(filepath[:-3] + '.mat')
            logging.info(f"Saving as MATLAB file: {destination}")
            contents = {}
            contents['echoData'] = echo_data_to_save
            contents['lineData'] = self.rfdata.lineData
            contents['lineHeader'] = self.rfdata.lineHeader
            contents['headerInfo'] = self.rfdata.headerInfo
            contents['dbParams'] = self.rfdata.dbParams
            contents['rf_data_all_fund'] = rf_data_all_fund
            contents['rf_data_all_harm'] = rf_data_all_harm
            contents['NumFrame'] = self.numFrame
            contents['NumSonoCTAngles'] = self.NumSonoCTAngles
            contents['pt'] = self.pt
            contents['multilinefactor'] = self.multilinefactor
            if hasattr(self.rfdata, 'echoData') and len(self.rfdata.echoData) > 1:
                contents['echoData1'] = self.rfdata.echoData[1]
                logging.debug("Added echoData1 to output")
            if hasattr(self.rfdata, 'echoData') and len(self.rfdata.echoData) > 2:
                contents['echoData2'] = self.rfdata.echoData[2]
                logging.debug("Added echoData2 to output")
            if hasattr(self.rfdata, 'echoData') and len(self.rfdata.echoData) > 3:
                contents['echoData3'] = self.rfdata.echoData[3]
                logging.debug("Added echoData3 to output")
            if hasattr(self.rfdata, 'echoMModeData'):
                contents['echoMModeData'] = self.rfdata.echoMModeData
                logging.debug("Added echoMModeData to output")
            if hasattr(self.rfdata, 'miscData'):
                contents['miscData'] = self.rfdata.miscData
                logging.debug("Added miscData to output")
            
            if os.path.exists(destination):
                os.remove(destination)
                logging.debug(f"Removed existing file: {destination}")
            savemat(destination, contents)
            logging.info(f"MATLAB file saved successfully: {destination}")
        else:
            # Save as numpy files
            numpy_folder = os.path.join(os.path.dirname(filepath), 'sample_npy')
            if not os.path.exists(numpy_folder):
                os.makedirs(numpy_folder)
                logging.debug(f"Created NumPy output folder: {numpy_folder}")
            logging.info(f"Saving as NumPy files in: {numpy_folder}")
            np.save(os.path.join(numpy_folder, 'echoData.npy'), echo_data_to_save)
            np.save(os.path.join(numpy_folder, 'lineData.npy'), self.rfdata.lineData)
            np.save(os.path.join(numpy_folder, 'lineHeader.npy'), self.rfdata.lineHeader)
            np.save(os.path.join(numpy_folder, 'rf_data_all_fund.npy'), rf_data_all_fund)
            np.save(os.path.join(numpy_folder, 'rf_data_all_harm.npy'), rf_data_all_harm)
            logging.info("NumPy files saved successfully")
        
        result_shape = np.array(rf_data_all_fund).shape
        logging.info(f"Parsing complete. Final data shape: {result_shape}")
        return result_shape


###################################################################################

if __name__ == "__main__":
    # Configure logging for main execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Hardcoded file path - no command line arguments needed
    filepath = r"D:\Omid\0_samples\Philips\David\sample.rf"
    logging.info(f"Starting main execution with file: {filepath}")
    parser = PhilipsRfParser()
    parser.philipsRfParser(filepath, save_numpy=True)
    logging.info("Main execution complete")
    
###################################################################################