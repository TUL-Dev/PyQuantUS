import os
import platform
from pathlib import Path
from datetime import datetime
import warnings
import ctypes as ct
import logging
from typing import Optional, Tuple, List, Any

import numpy as np
from scipy.io import savemat
from philipsRfParser import getPartA, getPartB

###################################################################################
# HeaderInfoStruct Class
###################################################################################
class HeaderInfoStruct:
    """Philips-specific structure containing information from the headers."""
    def __init__(self):
        logging.debug("Initializing HeaderInfoStruct")
        
        self.RF_CaptureVersion: Optional[np.ndarray] = None
        self.Tap_Point: Optional[np.ndarray] = None
        self.Data_Gate: Optional[np.ndarray] = None
        self.Multilines_Capture: Optional[np.ndarray] = None
        self.Steer: Optional[np.ndarray] = None
        self.elevationPlaneOffset: Optional[np.ndarray] = None
        self.PM_Index: Optional[np.ndarray] = None
        self.Pulse_Index: Optional[np.ndarray] = None
        self.Data_Format: Optional[np.ndarray] = None
        self.Data_Type: Optional[np.ndarray] = None
        self.Header_Tag: Optional[np.ndarray] = None
        self.Threed_Pos: Optional[np.ndarray] = None
        self.Mode_Info: Optional[np.ndarray] = None
        self.Frame_ID: Optional[np.ndarray] = None
        self.CSID: Optional[np.ndarray] = None
        self.Line_Index: Optional[np.ndarray] = None
        self.Line_Type: Optional[np.ndarray] = None
        self.Time_Stamp: Optional[np.ndarray] = None
        self.RF_Sample_Rate: Optional[np.ndarray] = None
        
        logging.debug("HeaderInfoStruct initialization complete")

###################################################################################
# DbParams Class
###################################################################################
class DbParams:
    """Philips-specific structure containing signal properties of the scan."""
    def __init__(self):
        logging.debug("Initializing DbParams")
        
        self.acqNumActiveScChannels2d: Optional[List[int]] = None
        self.azimuthMultilineFactorXbrOut: Optional[List[int]] = None
        self.azimuthMultilineFactorXbrIn: Optional[List[int]] = None
        self.numOfSonoCTAngles2dActual: Optional[List[int]] = None
        self.elevationMultilineFactor: Optional[List[int]] = None
        self.numPiPulses: Optional[List[int]] = None
        self.num2DCols: Optional[np.ndarray] = None
        self.fastPiEnabled: Optional[List[int]] = None
        self.numZones2d: Optional[List[int]] = None
        self.numSubVols: Optional[Any] = None
        self.numPlanes: Optional[Any] = None
        self.zigZagEnabled: Optional[Any] = None
        self.azimuthMultilineFactorXbrOutCf: Optional[List[int]] = None
        self.azimuthMultilineFactorXbrInCf: Optional[List[int]] = None
        self.multiLineFactorCf: Optional[List[int]] = None
        self.linesPerEnsCf: Optional[List[int]] = None
        self.ensPerSeqCf: Optional[List[int]] = None
        self.numCfCols: Optional[List[int]] = None
        self.numCfEntries: Optional[List[int]] = None
        self.numCfDummies: Optional[List[int]] = None
        self.elevationMultilineFactorCf: Optional[List[int]] = None
        self.Planes: Optional[List[int]] = None
        self.tapPoint: Optional[List[int]] = None
        
        logging.debug("DbParams initialization complete")

###################################################################################
# Rfdata Class
###################################################################################
class Rfdata:
    """Philips-specific structure containing constructed RF data."""
    def __init__(self):
        logging.debug("Initializing Rfdata")
        
        self.lineData: Optional[np.ndarray] = None
        self.lineHeader: Optional[np.ndarray] = None
        self.headerInfo: Optional[HeaderInfoStruct] = None
        self.echoData: Optional[Any] = None
        self.dbParams: Optional[DbParams] = None
        self.echoMModeData: Optional[Any] = None
        self.miscData: Optional[Any] = None
        
        logging.debug("Rfdata initialization complete")

###################################################################################
# Main Parser Class
###################################################################################
class PhilipsRfParser:
    """Main class for parsing Philips RF data files."""
    def __init__(self, ML_out: int = 2, ML_in: int = 32, used_os: Optional[int] = None):
        """Initialize the parser with default parameters."""
        logging.info(f"Initializing PhilipsRfParser with ML_out={ML_out}, ML_in={ML_in}, used_os={used_os}")
        
        self.ML_out: int = ML_out
        self.ML_in: int = ML_in
        self.used_os: Optional[int] = used_os
        self.rfdata: Optional[Rfdata] = None
        self.txBeamperFrame: Optional[int] = None
        self.NumSonoCTAngles: Optional[int] = None
        self.numFrame: Optional[int] = None
        self.multilinefactor: Optional[int] = None
        self.pt: Optional[int] = None
        
        logging.debug("PhilipsRfParser initialization complete")
        
    ###################################################################################
    # Utility Functions
    ###################################################################################
    @staticmethod
    def _get_filler_zeros(num: int) -> str:
        """Get string of zeros for padding."""
        logging.debug(f"Creating filler zeros, num={num}")
        
        result = '0' * max(0, num - 1)
        
        logging.debug(f"Generated {len(result)} filler zeros")
        return result

    ###################################################################################
    # File Type Detection and File Header
    ###################################################################################
    def _detect_file_type(self, file_obj) -> Tuple[bool, bool, int, List[int]]:
        """Detects file type and returns is_voyager, has_file_header, file_header_size, file_header."""
        logging.info("Starting file type detection")
        
        VHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 160, 160]
        FHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 11, 11]
        file_header_size = len(VHeader)
        
        logging.debug(f"Reading {file_header_size} bytes for header detection")
        file_header = list(file_obj.read(file_header_size))
        logging.debug(f"Read file header: {file_header[:10]}...")
        
        is_voyager = False
        has_file_header = False
        
        if file_header == VHeader:
            logging.info("Header information found - Parsing Voyager RF capture file")
            is_voyager = True
            has_file_header = True
        elif file_header == FHeader:
            logging.info("Header information found - Parsing Fusion RF capture file")
            has_file_header = True
        else:
            logging.info("No header found - Parsing legacy Voyager RF capture file")
            is_voyager = True
            
        logging.debug(f"File type detection complete: is_voyager={is_voyager}, has_file_header={has_file_header}")
        return is_voyager, has_file_header, file_header_size, file_header

    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _parse_file_header_and_offset(self, file_obj, is_voyager: bool, has_file_header: bool, file_header_size: int, filepath: str) -> Tuple[Optional[DbParams], int, str]:
        """Parse file header and calculate total_header_size, endianness, and db_params."""
        logging.info("Parsing file header and calculating offset")
        logging.debug(f"Input parameters: is_voyager={is_voyager}, has_file_header={has_file_header}, file_header_size={file_header_size}")
        
        endianness = 'little'
        db_params = None
        num_file_header_bytes = 0
        
        if has_file_header:
            if is_voyager:
                endianness = 'big'
                logging.debug("Using big-endian for Voyager file")
            else:
                logging.debug("Using little-endian for Fusion file")
                
            logging.info("Parsing file header parameters")
            db_params, num_file_header_bytes = self._parse_file_header(file_obj, endianness)
            total_header_size = file_header_size + 8 + num_file_header_bytes
            
            logging.debug(f"Total header size: {total_header_size} bytes (file_header={file_header_size} + 8 + params={num_file_header_bytes})")
        else:
            total_header_size = 0
            logging.debug("No file header to parse")
            
        logging.info(f"File header parsing complete: endianness={endianness}, total_header_size={total_header_size}")
        return db_params, total_header_size, endianness

    ###################################################################################
    # Raw Data Loading
    ###################################################################################
    def _load_raw_rf_data(self, filepath: str, is_voyager: bool, total_header_size: int, read_offset: int, read_size: int) -> Tuple[Any, Optional[int]]:
        """Load raw RF data from file, handling Voyager and Fusion formats."""
        logging.info(f"Loading raw RF data: is_voyager={is_voyager}, offset={read_offset}MB, size={read_size}MB")
        
        file_size = os.stat(filepath).st_size
        remaining_size = file_size - total_header_size
        logging.debug(f"File size: {file_size} bytes, header size: {total_header_size} bytes, remaining: {remaining_size} bytes")
        
        read_offset *= 2 ** 20
        read_size *= 2 ** 20
        logging.debug(f"Read parameters in bytes: offset={read_offset}, size={read_size}")
        
        if is_voyager:
            logging.info("Loading Voyager format data")
            
            alignment = np.arange(0, remaining_size + 1, 36)
            offset_diff = alignment - read_offset
            read_diff = alignment - read_size
            read_offset = alignment[np.where(offset_diff >= 0)[0][0]].__int__()
            read_size = alignment[np.where(read_diff >= 0)[0][0]].__int__()
            
            logging.debug(f"Aligned Voyager read - offset: {read_offset}, size: {read_size}")
            
            with open(filepath, 'rb') as f:
                f.seek(read_offset)
                rawrfdata = f.read(read_size)
                
            logging.info(f"Loaded {len(rawrfdata)} bytes of Voyager data")
            return rawrfdata, None
        else:
            logging.info("Loading Fusion format data")
            
            alignment = np.arange(0, remaining_size + 1, 32)
            offset_diff = alignment - read_offset
            read_diff = alignment - read_size
            matching_indices = np.where(offset_diff >= 0)[0]
            
            if len(matching_indices) > 0:
                read_offset = alignment[matching_indices[0]].__int__()
            else:
                read_offset = 0
                logging.warning("No matching offset found, using 0")
                
            matching_indices = np.where(read_diff >= 0)[0]
            if len(matching_indices) > 0:
                read_size = alignment[matching_indices[0]].__int__()
            else:
                read_size = remaining_size
                logging.warning(f"No matching size found, using remaining size: {read_size}")
                
            num_clumps = int(np.floor(read_size / 32))
            logging.debug(f"Aligned Fusion read - offset: {read_offset}, size: {read_size}, clumps: {num_clumps}")
            
            offset = total_header_size + read_offset
            logging.info(f"Reading Fusion data: {num_clumps} clumps from offset {offset}")
            
            partA = getPartA(num_clumps, filepath, offset)
            partB = getPartB(num_clumps, filepath, offset)
            logging.debug(f"Retrieved partA: {len(partA)} elements, partB: {len(partB)} elements")
            
            rawrfdata = np.concatenate((np.array(partA, dtype=int).reshape((12, num_clumps), order='F'), np.array([partB], dtype=int)))
            logging.debug(f"Raw RF data shape: {rawrfdata.shape}")
            
            logging.info(f"Loaded Fusion data with shape {rawrfdata.shape}")
            return rawrfdata, num_clumps

    ###################################################################################
    # Data Reshaping
    ###################################################################################
    def _reshape_voyager_raw_data(self, rawrfdata: Any) -> Any:
        """Reshape Voyager raw RF data if needed."""
        logging.info("Reshaping Voyager raw data")
        
        initial_size = len(rawrfdata)
        logging.debug(f"Initial raw data size: {initial_size} bytes")
        
        num_clumps = np.floor(len(rawrfdata) / 36)
        logging.debug(f"Calculated clumps: {num_clumps}")
        
        rlimit = 180_000_000
        if len(rawrfdata) > rlimit:
            logging.warning(f"Large file detected ({len(rawrfdata)} bytes), chunking reshape operation")
            
            num_chunks = int(np.floor(len(rawrfdata) / rlimit))
            num_rem_bytes = np.mod(len(rawrfdata), rlimit)
            num_clump_group = int(rlimit / 36)
            logging.debug(f"Chunking: {num_chunks} chunks, {num_rem_bytes} remaining bytes, {num_clump_group} clumps per chunk")
            
            temp = np.zeros((num_chunks + 1, 3, 12, num_clump_group))
            m = 0
            n = 0
            
            for i in range(num_chunks):
                logging.debug(f"Processing chunk {i+1}/{num_chunks}")
                temp[i] = np.reshape(rawrfdata[m:m + rlimit], (3, 12, num_clump_group))
                m += rlimit
                n += num_clump_group
                
            if num_rem_bytes > 0:
                logging.debug(f"Processing remaining {num_rem_bytes} bytes")
                temp[num_chunks] = np.reshape(rawrfdata[m:int(num_clumps * 36)], (3, 12, int(num_clumps - n)))
                
            rawrfdata = np.concatenate((temp[:]), axis=2)
            logging.debug("Chunked reshape complete")
        else:
            logging.debug("Direct reshape for normal size file")
            rawrfdata = np.reshape(rawrfdata, (3, 12, int(num_clumps)), order='F')
            
        logging.info(f"Voyager reshape complete, final shape: {np.array(rawrfdata).shape}")
        return rawrfdata

    ###################################################################################
    # Header and RF Data Parsing Dispatch
    ###################################################################################
    def _parse_header_dispatch(self, rawrfdata: Any, is_voyager: bool) -> HeaderInfoStruct:
        """Dispatch to the correct header parsing method."""
        logging.info(f"Dispatching header parsing: is_voyager={is_voyager}")
        
        if is_voyager:
            logging.debug("Using Voyager header parser")
            return self._parse_header_v(rawrfdata)
        else:
            logging.debug("Using Fusion header parser")
            return self._parse_header_f(rawrfdata)

    ###################################################################################
    # RF Data Parsing Dispatch
    ###################################################################################
    def _parse_rf_data_dispatch(self, rawrfdata: Any, header_info: HeaderInfoStruct, is_voyager: bool) -> Tuple[np.ndarray, np.ndarray, int]:
        """Dispatch to the correct RF data parsing method."""
        logging.info(f"Dispatching RF data parsing: is_voyager={is_voyager}")
        return self._parse_rf_data(rawrfdata, header_info, is_voyager)

    ###################################################################################
    # Data Type Organization (Placeholder)
    ###################################################################################
    def _organize_data_types(self, rfdata: Rfdata, header_info: HeaderInfoStruct, tap_point: int) -> Rfdata:
        """Organize data types (echo, color, etc.) and assign them to rfdata."""
        logging.info(f"Organizing data types, tap_point={tap_point}")
        logging.debug(f"Available data types in headers: {np.unique(header_info.Data_Type) if hasattr(header_info, 'Data_Type') else 'N/A'}")
        
        rfdata = self._extract_echo_data(rfdata, header_info, tap_point)
        rfdata = self._extract_cw_data(rfdata, header_info, tap_point)
        rfdata = self._extract_pw_data(rfdata, header_info, tap_point)
        rfdata = self._extract_color_data(rfdata, header_info, tap_point)
        rfdata = self._extract_echo_mmode_data(rfdata, header_info, tap_point)
        rfdata = self._extract_color_mmode_data(rfdata, header_info, tap_point)
        rfdata = self._extract_dummy_data(rfdata, header_info, tap_point)
        rfdata = self._extract_swi_data(rfdata, header_info, tap_point)
        rfdata = self._extract_misc_data(rfdata, header_info, tap_point)
        
        logging.info("Data type organization complete")
        return rfdata

    ###################################################################################
    # Echo Data Extraction
    ###################################################################################
    def _extract_echo_data(self, rfdata, header_info, tap_point):
        logging.info(f"Extracting echo data, tap_point={tap_point}")
        
        DataType_ECHO = np.arange(1, 15)
        logging.debug(f"Echo data types: {DataType_ECHO}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            
            # Log the sample rate when it's used to determine ML_Capture
            logging.info(f"RF Sample Rate: {SAMPLE_RATE}")
            
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32 # 20MHz Capture
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
        
        logging.debug(f"ML_Capture={ML_Capture}")
        
        if tap_point == 7:
            tap_point = 4
            logging.debug("Tap point 7 converted to 4")
            
        xmit_events = len(header_info.Data_Type)
        logging.debug(f"Number of transmit events: {xmit_events}")
        
        echo_index = np.zeros(xmit_events).astype(np.int32)
        for dt in DataType_ECHO:
            index = ((header_info.Data_Type & 255) == dt)
            echo_index = np.bitwise_or(echo_index, np.array(index).astype(np.int32))
            
        echo_count = np.sum(echo_index)
        logging.debug(f"Found {echo_count} echo data entries")
        
        if echo_count == 0 and np.any(header_info.Data_Type == 1):
            echo_index = (header_info.Data_Type == 1).astype(np.int32)
            echo_count = np.sum(echo_index)
            logging.debug(f"Fallback: Found {echo_count} entries with Data_Type=1")
            
        if echo_count > 0:
            columns_to_keep = np.where(echo_index == 1)[0]
            logging.debug(f"Processing {len(columns_to_keep)} echo columns")
            
            pruning_line_data = rfdata.lineData[:, columns_to_keep]
            pruning_line_header = rfdata.lineHeader[:, columns_to_keep]
            logging.debug(f"Pruning data shape: {pruning_line_data.shape}")
            
            if tap_point == 4:
                echo_data = pruning_line_data
                logging.debug("Using data directly (tap_point=4)")
            else:
                logging.debug("Pruning data before use")
                echo_data = self._prune_data(pruning_line_data, pruning_line_header, ML_Capture)
                
            if tap_point in [0, 1]:
                ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrIn[0] * rfdata.dbParams.elevationMultilineFactor[0]
                CRE = 1
                logging.debug(f"Tap point {tap_point}: ML_Actual={ML_Actual}, CRE={CRE}")
                rfdata.echoData = self._sort_rf(echo_data, ML_Capture, ML_Actual, CRE, False)
            elif tap_point == 2:
                ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrOut[0] * rfdata.dbParams.elevationMultilineFactor[0]
                CRE = rfdata.dbParams.acqNumActiveScChannels2d[0]
                logging.debug(f"Tap point {tap_point}: ML_Actual={ML_Actual}, CRE={CRE}")
                rfdata.echoData = self._sort_rf(echo_data, ML_Capture, ML_Actual, CRE, False)
            elif tap_point == 4:
                ML_Actual = 128
                CRE = 1
                logging.debug(f"Tap point {tap_point}: ML_Actual={ML_Actual}, CRE={CRE}")
                rfdata.echoData = self._sort_rf(echo_data, ML_Actual, ML_Actual, CRE, False)
                
            logging.info(f"Echo data extracted successfully, shape: {rfdata.echoData[0].shape if hasattr(rfdata.echoData, '__getitem__') else 'N/A'}")
        else:
            logging.warning("No echo data found")
            
        return rfdata

    ###################################################################################
    # CW Data Extraction
    ###################################################################################
    def _extract_cw_data(self, rfdata, header_info, tap_point):
        logging.info(f"Extracting CW data, tap_point={tap_point}")
        
        DataType_CW = 16
        logging.debug(f"CW data type: {DataType_CW}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            
            # Log the sample rate when it's used to determine ML_Capture
            logging.info(f"RF Sample Rate: {SAMPLE_RATE}")
            
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
        
        logging.debug(f"ML_Capture={ML_Capture}")
        
        cw_index = (header_info.Data_Type == DataType_CW)
        cw_count = np.sum(cw_index)
        logging.debug(f"Found {cw_count} CW data entries")
        
        if cw_count > 0:
            logging.debug("Processing CW data")
            cw_data = self._prune_data(rfdata.lineData[:, cw_index], rfdata.lineHeader[:, cw_index], ML_Capture)
            
            ML_Actual = 1
            CRE = 1
            logging.debug(f"CW sorting: ML_Actual={ML_Actual}, CRE={CRE}")
            
            rfdata.cwData = self._sort_rf(cw_data, ML_Capture, ML_Actual, CRE, False)
            logging.info(f"CW data extracted successfully, shape: {rfdata.cwData[0].shape if hasattr(rfdata.cwData, '__getitem__') else 'N/A'}")
        else:
            logging.debug("No CW data found")
            
        return rfdata

    ###################################################################################
    # PW Data Extraction
    ###################################################################################
    def _extract_pw_data(self, rfdata, header_info, tap_point):
        logging.info(f"Extracting PW data, tap_point={tap_point}")
        
        DataType_PW = [18, 19]
        logging.debug(f"PW data types: {DataType_PW}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
            
        logging.debug(f"ML_Capture={ML_Capture}")
        
        xmit_events = len(header_info.Data_Type)
        pw_index = np.zeros(xmit_events).astype(bool)
        
        for dt in DataType_PW:
            index = (header_info.Data_Type == dt)
            pw_index = np.bitwise_or(pw_index, index)
            
        pw_count = np.sum(pw_index)
        logging.debug(f"Found {pw_count} PW data entries")
        
        if pw_count > 0:
            logging.debug("Processing PW data")
            pw_data = self._prune_data(rfdata.lineData[:, pw_index], rfdata.lineHeader[:, pw_index], ML_Capture)
            
            ML_Actual = 1
            CRE = 1
            logging.debug(f"PW sorting: ML_Actual={ML_Actual}, CRE={CRE}")
            
            rfdata.pwData = self._sort_rf(pw_data, ML_Capture, ML_Actual, CRE, False)
            logging.info(f"PW data extracted successfully, shape: {rfdata.pwData[0].shape if hasattr(rfdata.pwData, '__getitem__') else 'N/A'}")
        else:
            logging.debug("No PW data found")
            
        return rfdata

    ###################################################################################
    # Color Data Extraction
    ###################################################################################
    def _extract_color_data(self, rfdata, header_info, tap_point):
        logging.info(f"Extracting color data, tap_point={tap_point}")
        
        DataType_COLOR = [17, 21, 22, 23, 24]
        logging.debug(f"Color data types: {DataType_COLOR}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
            
        logging.debug(f"ML_Capture={ML_Capture}")
        
        xmit_events = len(header_info.Data_Type)
        color_index = np.zeros(xmit_events).astype(bool)
        
        for dt in DataType_COLOR:
            index = (header_info.Data_Type == dt)
            color_index = np.bitwise_or(color_index, index)
            
        color_count = np.sum(color_index)
        logging.debug(f"Found {color_count} color data entries")
        
        if color_count > 0:
            logging.debug("Processing color data")
        if np.sum(color_index) > 0:
            color_data = self._prune_data(rfdata.lineData[:, color_index], rfdata.lineHeader[:, color_index], ML_Capture)
            if tap_point in [0, 1]:
                ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrInCf * rfdata.dbParams.elevationMultilineFactorCf
            else:
                ML_Actual = rfdata.dbParams.azimuthMultilineFactorXbrOutCf * rfdata.dbParams.elevationMultilineFactorCf
            CRE = 1
            rfdata.colorData = self._sort_rf(color_data, ML_Capture, ML_Actual, CRE, False)
        return rfdata

    ###################################################################################
    # Echo M-Mode Data Extraction
    ###################################################################################
    def _extract_echo_mmode_data(self, rfdata, header_info, tap_point):
        logging.info(f"Extracting echo M-Mode data, tap_point={tap_point}")
        
        DataType_EchoMMode = 26
        logging.debug(f"Echo M-Mode data type: {DataType_EchoMMode}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
            
        logging.debug(f"ML_Capture={ML_Capture}")
        
        echo_mmode_index = (header_info.Data_Type == DataType_EchoMMode)
        echo_mmode_count = np.sum(echo_mmode_index)
        logging.debug(f"Found {echo_mmode_count} echo M-Mode data entries")
        
        if echo_mmode_count > 0:
            logging.debug("Processing echo M-Mode data")
            echo_mmode_data = self._prune_data(rfdata.lineData[:, echo_mmode_index], rfdata.lineHeader[:, echo_mmode_index], ML_Capture)
            
            ML_Actual = 1
            CRE = 1
            logging.debug(f"Echo M-Mode sorting: ML_Actual={ML_Actual}, CRE={CRE}")
            
            rfdata.echoMModeData = self._sort_rf(echo_mmode_data, ML_Capture, ML_Actual, CRE, False)
            logging.info(f"Echo M-Mode data extracted successfully, shape: {rfdata.echoMModeData[0].shape if hasattr(rfdata.echoMModeData, '__getitem__') else 'N/A'}")
        else:
            logging.debug("No echo M-Mode data found")
            
        return rfdata

    ###################################################################################
    # Color M-Mode Data Extraction
    ###################################################################################
    def _extract_color_mmode_data(self, rfdata, header_info, tap_point):
        logging.info(f"Extracting color M-Mode data, tap_point={tap_point}")
        
        DataType_ColorMMode = [27, 28]
        logging.debug(f"Color M-Mode data types: {DataType_ColorMMode}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
            
        logging.debug(f"ML_Capture={ML_Capture}")
        
        xmit_events = len(header_info.Data_Type)
        color_mmode_index = np.zeros(xmit_events).astype(bool)
        
        for dt in DataType_ColorMMode:
            index = (header_info.Data_Type == dt)
            color_mmode_index = np.bitwise_or(color_mmode_index, index)
            
        color_mmode_count = np.sum(color_mmode_index)
        logging.debug(f"Found {color_mmode_count} color M-Mode data entries")
        
        if color_mmode_count > 0:
            logging.debug("Processing color M-Mode data")
            color_mmode_data = self._prune_data(rfdata.lineData[:, color_mmode_index], rfdata.lineHeader[:, color_mmode_index], ML_Capture)
            
            ML_Actual = 1
            CRE = 1
            logging.debug(f"Color M-Mode sorting: ML_Actual={ML_Actual}, CRE={CRE}")
            
            rfdata.colorMModeData = self._sort_rf(color_mmode_data, ML_Capture, ML_Actual, CRE, False)
            logging.info(f"Color M-Mode data extracted successfully, shape: {rfdata.colorMModeData[0].shape if hasattr(rfdata.colorMModeData, '__getitem__') else 'N/A'}")
        else:
            logging.debug("No color M-Mode data found")
            
        return rfdata

    ###################################################################################
    # Dummy Data Extraction
    ###################################################################################
    def _extract_dummy_data(self, rfdata, header_info, tap_point):
        logging.info(f"Extracting dummy data, tap_point={tap_point}")
        
        DataType_Dummy = [20, 25, 29, 30, 31]
        logging.debug(f"Dummy data types: {DataType_Dummy}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
            
        logging.debug(f"ML_Capture={ML_Capture}")
        
        xmit_events = len(header_info.Data_Type)
        dummy_index = np.zeros(xmit_events).astype(bool)
        
        for dt in DataType_Dummy:
            index = (header_info.Data_Type == dt)
            dummy_index = np.bitwise_or(dummy_index, index)
            
        dummy_count = np.sum(dummy_index)
        logging.debug(f"Found {dummy_count} dummy data entries")
        
        if dummy_count > 0:
            logging.debug("Processing dummy data")
            dummy_data = self._prune_data(rfdata.lineData[:, dummy_index], rfdata.lineHeader[:, dummy_index], ML_Capture)
            
            ML_Actual = 2
            CRE = 1
            logging.debug(f"Dummy sorting: ML_Actual={ML_Actual}, CRE={CRE}")
            
            rfdata.dummyData = self._sort_rf(dummy_data, ML_Capture, ML_Actual, CRE, False)
            logging.info(f"Dummy data extracted successfully, shape: {rfdata.dummyData[0].shape if hasattr(rfdata.dummyData, '__getitem__') else 'N/A'}")
        else:
            logging.debug("No dummy data found")
            
        return rfdata

    ###################################################################################
    # SWI Data Extraction
    ###################################################################################
    def _extract_swi_data(self, rfdata, header_info, tap_point):
        logging.info(f"Extracting SWI data, tap_point={tap_point}")
        
        DataType_SWI = [90, 91]
        logging.debug(f"SWI data types: {DataType_SWI}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
        xmit_events = len(header_info.Data_Type)
        swi_index = np.zeros(xmit_events).astype(bool)
        for dt in DataType_SWI:
            index = (header_info.Data_Type == dt)
            swi_index = np.bitwise_or(swi_index, index)
        if np.sum(swi_index) > 0:
            swi_data = self._prune_data(rfdata.lineData[:, swi_index], rfdata.lineHeader[:, swi_index], ML_Capture)
            ML_Actual = ML_Capture
            CRE = 1
            rfdata.swiData = self._sort_rf(swi_data, ML_Capture, ML_Actual, CRE, False)
        return rfdata

    ###################################################################################
    # Misc Data Extraction
    ###################################################################################
    def _extract_misc_data(self, rfdata, header_info, tap_point):
        DataType_Misc = [15, 88, 89]
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
        xmit_events = len(header_info.Data_Type)
        misc_index = np.zeros(xmit_events).astype(bool)
        for dt in DataType_Misc:
            index = (header_info.Data_Type == dt)
            misc_index = np.bitwise_or(misc_index, index)
        if np.sum(misc_index) > 0:
            misc_data = self._prune_data(rfdata.lineData[:, misc_index], rfdata.lineHeader[:, misc_index], ML_Capture)
            ML_Actual = ML_Capture
            CRE = 1
            rfdata.miscData = self._sort_rf(misc_data, ML_Capture, ML_Actual, CRE, False)
        return rfdata

    ###################################################################################
    # Data Pruning Utility
    ###################################################################################
    def _prune_data(self, lineData, lineHeader, ML_Capture):
        """Remove false gate data at beginning of the line."""
        logging.info(f"Pruning data - input shape: lineData={lineData.shape}, lineHeader={lineHeader.shape}, ML_Capture={ML_Capture}")
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
        alignment = np.arange(0,numSamples, np.double(ML_Capture))
        diff = alignment - iFirstSample
        iFirstSample = int(alignment[np.where(diff>=0)[0][0]])
        logging.debug(f"Aligned start point to {iFirstSample}")
        prunedData = lineData[iFirstSample:numSamples+1,:]
        lineHeader = lineHeader[iFirstSample:numSamples+1,:]
        logging.debug(f"Pruned from start: new shape {prunedData.shape}")
        numSamples = prunedData.shape[0]
        startPoint = int(np.floor(numSamples*0.99))-1
        logging.debug(f"Looking for end point from sample {startPoint}")
        indicesFound = np.where(lineHeader[startPoint:numSamples+1,referenceLine]==0)
        if not len(indicesFound[0]):
            iLastSample = numSamples
            logging.debug("No valid end point found, using last sample")
        else:
            iLastSample = indicesFound[0][0]+startPoint
            alignment = np.arange(0,numSamples, np.double(ML_Capture))
            diff = alignment - iLastSample
            iLastSample = int(alignment[np.where(diff >= 0)[0][0]])-1
            logging.debug(f"Found and aligned end point to {iLastSample}")
        prunedData = prunedData[:iLastSample+1, :]
        logging.info(f"Pruning complete - final shape: {prunedData.shape}")
        return prunedData

    ###################################################################################
    # RF Sorting Utility
    ###################################################################################
    def _sort_rf(self, RFinput, Stride, ML, CRE=1, isVoyager=True):
        """Sort RF data based on multiline parameters."""
        logging.info(f"Sorting RF data - input shape: {RFinput.shape}, Stride={Stride}, ML={ML}, CRE={CRE}, isVoyager={isVoyager}")
        N = RFinput.shape[0]
        xmitEvents = RFinput.shape[1]
        depth = int(np.floor(N/Stride))
        MLs = np.arange(0,ML)
        MLs = MLs[:]
        out1 = np.array([])
        out2 = np.array([])
        out3 = np.array([])
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
        for k in range(ML):
            iML = np.where(np.array(ML_SortList) == MLs[k])[0]
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
    # Main RF Parsing Orchestrator
    ###################################################################################
    def _parse_rf(self, filepath: str, read_offset: int, read_size: int) -> Rfdata:
        """Open and parse RF data file (refactored into smaller methods)."""
        logging.info(f"Opening RF file: {filepath}")
        logging.debug(f"Read parameters - offset: {read_offset}MB, size: {read_size}MB")
        rfdata = Rfdata()
        with open(filepath, 'rb') as file_obj:
            is_voyager, has_file_header, file_header_size, file_header = self._detect_file_type(file_obj)
            db_params, total_header_size, endianness = self._parse_file_header_and_offset(file_obj, is_voyager, has_file_header, file_header_size, filepath)
        rfdata.dbParams = db_params
        rawrfdata, num_clumps = self._load_raw_rf_data(filepath, is_voyager, total_header_size, read_offset, read_size)
        if is_voyager:
            rawrfdata = self._reshape_voyager_raw_data(rawrfdata)
        header_info = self._parse_header_dispatch(rawrfdata, is_voyager)
        line_data, line_header, tap_point = self._parse_rf_data_dispatch(rawrfdata, header_info, is_voyager)
        rfdata.lineData = line_data
        rfdata.lineHeader = line_header
        rfdata.headerInfo = header_info
        del rawrfdata
        rfdata = self._organize_data_types(rfdata, header_info, tap_point)
        return rfdata

    ###################################################################################
    # Parameter Calculation
    ###################################################################################
    def _calculate_parameters(self) -> None:
        """Calculate and set main parameters as instance variables."""
        logging.info("Calculating parsing parameters...")
        self.txBeamperFrame = int(np.array(self.rfdata.dbParams.num2DCols).flat[0])
        self.NumSonoCTAngles = int(self.rfdata.dbParams.numOfSonoCTAngles2dActual[0])
        logging.info(f"Beam parameters - txBeamperFrame: {self.txBeamperFrame}, NumSonoCTAngles: {self.NumSonoCTAngles}")
        self.numFrame = int(np.floor(self.rfdata.lineData.shape[1] / (self.txBeamperFrame * self.NumSonoCTAngles)))
        self.multilinefactor = self.ML_in
        logging.info(f"Calculated numFrame: {self.numFrame}, multilinefactor: {self.multilinefactor}")
        col = 0
        if np.any(self.rfdata.lineData[:, col] != 0):
            first_nonzero = np.where(self.rfdata.lineData[:, col] != 0)[0][0]
            last_nonzero = np.where(self.rfdata.lineData[:, col] != 0)[0][-1]
            self.used_os = first_nonzero
            self.pt = int(np.floor((last_nonzero - first_nonzero + 1) / self.multilinefactor))
            logging.info(f"Auto-detected: used_os={self.used_os}, pt={self.pt}")
        else:
            self.used_os = 2256 if self.used_os is None else self.used_os
            self.pt = int(np.floor((self.rfdata.lineData.shape[0] - self.used_os) / self.multilinefactor))
            logging.warning(f"Using fallback values: used_os={self.used_os}, pt={self.pt}")

    ###################################################################################
    # Data Array Filling
    ###################################################################################
    def _fill_data_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
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
                    if temp.shape[1] > 2:
                        rftemp_all_harm[:, np.arange(self.ML_out) + (k2 * self.ML_out)] = temp[:, [0, 2]]
                    else:
                        logging.warning(f"temp has only {temp.shape[1]} columns, skipping harmonic assignment")
                    if temp.shape[1] >= 12:
                        rftemp_all_fund[:, np.arange(self.ML_out) + (k2 * self.ML_out)] = temp[:, [9, 11]]
                    elif temp.shape[1] >= 2:
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
    # Voyager Header Parsing
    ###################################################################################
    def _parse_header_v(self, rawrfdata):
        """Parse header for Voyager systems."""
        logging.info("Parsing Voyager header information")
        temp_headerInfo = HeaderInfoStruct()

        iHeader = np.where(np.uint8(rawrfdata[2,0,:])&224)
        numHeaders = len(iHeader)-1 # Ignore last header as it is part of a partial line
        logging.debug(f"Found {numHeaders} headers in Voyager data")

        # Initialize header arrays 
        temp_headerInfo.RF_CaptureVersion = np.zeros(numHeaders, dtype=np.uint8)
        temp_headerInfo.Tap_Point = np.zeros(numHeaders, dtype=np.uint8)
        temp_headerInfo.Data_Gate = np.zeros(numHeaders, dtype=np.uint8)
        temp_headerInfo.Multilines_Capture = np.zeros(numHeaders, dtype=np.uint8)
        temp_headerInfo.RF_Sample_Rate = np.zeros(numHeaders, dtype=np.uint8)
        temp_headerInfo.Steer = np.zeros(numHeaders, dtype=np.uint8)
        temp_headerInfo.elevationPlaneOffset = np.zeros(numHeaders, dtype=np.uint8)
        temp_headerInfo.PM_Index = np.zeros(numHeaders, dtype=np.uint8)
        temp_headerInfo.Line_Index = np.zeros(numHeaders, dtype=np.uint16)
        temp_headerInfo.Pulse_Index = np.zeros(numHeaders, dtype=np.uint16)
        temp_headerInfo.Data_Format = np.zeros(numHeaders, dtype=np.uint16)
        temp_headerInfo.Data_Type = np.zeros(numHeaders, dtype=np.uint16)
        temp_headerInfo.Header_Tag = np.zeros(numHeaders, dtype=np.uint16)
        temp_headerInfo.Threed_Pos = np.zeros(numHeaders, dtype=np.uint16)
        temp_headerInfo.Mode_Info = np.zeros(numHeaders, dtype=np.uint16)
        temp_headerInfo.Frame_ID = np.zeros(numHeaders, dtype=np.uint32)
        temp_headerInfo.CSID = np.zeros(numHeaders, dtype=np.uint16)
        temp_headerInfo.Line_Type = np.zeros(numHeaders, dtype=np.uint16)
        temp_headerInfo.Time_Stamp = np.zeros(numHeaders, dtype=np.uint32)

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
            
            # Log sample rate for first header
            if m == 0:
                logging.info(f"Sample rate from first Voyager header: {temp_headerInfo.RF_Sample_Rate[m]}")
                
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
            temp_headerInfo.Time_Stamp[m] = int(packedHeader[iBit:iBit+32],2)

        logging.info(f"Voyager header parsing complete - processed {numHeaders} headers")
        return temp_headerInfo

    ###################################################################################
    # Fusion Header Parsing
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
            
            # Log sample rate for first header
            if m == 0:
                logging.info(f"Sample rate from first header: {HeaderInfo.RF_Sample_Rate[m]}")
                
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
    # Voyager Data Parsing
    ###################################################################################
    def _parse_data_v(self, rawrfdata, headerInfo):
        """Parse RF data for Voyager systems."""
        logging.info("Parsing Voyager RF data")
        minNeg = 16 * (2**16)  # Corrected exponentiation

        iHeader = np.where(rawrfdata[2,0,:]&224==64)
        numHeaders = len(iHeader)-1
        numSamples = (iHeader[1]-iHeader[0]-1)*12
        logging.debug(f"Voyager data: {numHeaders} headers, {numSamples} samples per line")

        lineData = np.zeros((numSamples, numHeaders), dtype=np.int32)
        lineHeader = np.zeros((numSamples, numHeaders), dtype=np.uint8)

        logging.info("Extracting Voyager line data...")
        for m in range(numHeaders):  # Corrected loop
            if m % 1000 == 0:
                logging.debug(f"Processing Voyager line {m}/{numHeaders}")

            iStartData = iHeader[m]+1
            iStopData = iHeader[m+1]-1

            if headerInfo.Data_Type[m] == float(0x5a):
                iStopData = iStartData+10000
                logging.debug(f"Line {m} is push pulse, limiting data size")

            lineData_u8 = rawrfdata[:,:,iStartData:iStopData]
            lineData_s32 = (
                np.int32(lineData_u8[0,:,:]) +
                np.int32(lineData_u8[1,:,:]) * (2**8) +
                np.int32(lineData_u8[2,:,:] & np.uint8(31)) * (2**16)
            )
            iNeg = np.where(lineData_s32 >= minNeg)
            lineData_s32[iNeg] = lineData_s32[iNeg] - 2*minNeg
            lineHeader_u8 = (lineData_u8[2,:,:] & 224) >> 6

            lineData[:lineData_s32.size, m] = lineData_s32.ravel(order='F')
            lineHeader[:lineHeader_u8.size, m] = lineHeader_u8.ravel(order='F')

        logging.info(f"Voyager data parsing complete - lineData: {lineData.shape}, lineHeader: {lineHeader.shape}")
        return lineData, lineHeader

    ###################################################################################
    # Fusion Data Parsing
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
    # RF Data Parsing
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
    # File Header Parsing
    ###################################################################################
    def _parse_file_header(self, file_obj, endianness):
        """Parse file header information."""
        logging.info("Parsing file header information")
        fileVersion = int.from_bytes(file_obj.read(4), endianness, signed=False)
        numFileHeaderBytes = int.from_bytes(file_obj.read(4), endianness, signed=False)
        logging.info(f"File Version: {fileVersion}, Header Size: {numFileHeaderBytes} bytes")

        # Handle accordingly to fileVersion
        temp_dbParams = DbParams()
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
    # Main Parsing Function
    ###################################################################################
    def philipsRfParser(self, filepath: str, save_numpy: bool = False) -> np.ndarray:
        """Parse Philips RF data file, save as .mat file, and return shape of data.
        If save_numpy is True, only save the processed data as .npy files in a folder named '{sample_name}_extracted' in the sample path.
        If save_numpy is False, only save as .mat file."""
        
        # Set up logging
        if save_numpy:
            # Create numpy folder with sample name + '_extracted'
            sample_name = os.path.splitext(os.path.basename(filepath))[0]
            numpy_folder = os.path.join(os.path.dirname(filepath), f'{sample_name}_extracted')
            if not os.path.exists(numpy_folder):
                os.makedirs(numpy_folder)
            
            # Set up file logging to the numpy folder - all levels
            log_file = os.path.join(numpy_folder, 'parsing_log.txt')
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.DEBUG)  # Capture all levels for file
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            # Set up console logging - only INFO and WARNING
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            # Create custom filter for console to only show INFO and WARNING
            class InfoWarningFilter(logging.Filter):
                def filter(self, record):
                    return record.levelno in (logging.INFO, logging.WARNING)
            
            console_handler.addFilter(InfoWarningFilter())
            
            # Configure logger
            logger = logging.getLogger()
            # Clear any existing handlers
            logger.handlers.clear()
            # Add both handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.DEBUG)  # Logger needs to be at DEBUG to capture all messages
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        logging.info(f"Starting Philips RF parsing for file: {filepath}")
        logging.info(f"Save format: {'NumPy arrays' if save_numpy else 'MATLAB file'}")
        
        self.rfdata = self._parse_rf(filepath, 0, 2000)
        
        # Save header summary if saving as numpy
        if save_numpy:
            self._save_header_summary(numpy_folder)
        
        # Try to find the first available data type to save
        data_priority = [
            ('echoData', 'echoData'),
            ('cwData', 'cwData'),
            ('pwData', 'pwData'),
            ('colorData', 'colorData'),
            ('echoMModeData', 'echoMModeData'),
            ('colorMModeData', 'colorMModeData'),
            ('dummyData', 'dummyData'),
            ('swiData', 'swiData'),
            ('miscData', 'miscData'),
        ]
        
        data_to_save = None
        data_type_label = None
        for attr, label in data_priority:
            if hasattr(self.rfdata, attr) and getattr(self.rfdata, attr) is not None:
                data_to_save = getattr(self.rfdata, attr)
                data_type_label = label
                if isinstance(data_to_save, (list, tuple)) and len(data_to_save) > 0:
                    data_to_save = data_to_save[0]
                break
        
        if data_to_save is None or (hasattr(data_to_save, 'size') and data_to_save.size == 0):
            error_msg = f"No supported data found in RF file. Data_Type values: {np.unique(self.rfdata.headerInfo.Data_Type) if hasattr(self.rfdata.headerInfo, 'Data_Type') else 'N/A'}. lineData shape: {self.rfdata.lineData.shape if hasattr(self.rfdata, 'lineData') else 'N/A'}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        
        logging.info(f"Saving data type: {data_type_label} as 'echoData'")
        
        # Data preprocessing
        if (self.rfdata.headerInfo.Line_Index[249] == self.rfdata.headerInfo.Line_Index[250]):
            self.rfdata.lineData = self.rfdata.lineData[:, np.arange(2, self.rfdata.lineData.shape[1], 2)]
        else:
            self.rfdata.lineData = self.rfdata.lineData[:, np.arange(1, self.rfdata.lineData.shape[1], 2)]
        
        self._calculate_parameters()
        rf_data_all_fund, rf_data_all_harm = self._fill_data_arrays()
        
        if not save_numpy:
            destination = str(filepath[:-3] + '.mat')
            logging.info(f"Saving as MATLAB file: {destination}")
            contents = {
                'echoData': data_to_save,
                'lineData': self.rfdata.lineData,
                'lineHeader': self.rfdata.lineHeader,
                'headerInfo': self.rfdata.headerInfo,
                'dbParams': self.rfdata.dbParams,
                'rf_data_all_fund': rf_data_all_fund,
                'rf_data_all_harm': rf_data_all_harm,
                'NumFrame': self.numFrame,
                'NumSonoCTAngles': self.NumSonoCTAngles,
                'pt': self.pt,
                'multilinefactor': self.multilinefactor,
            }
            
            if hasattr(self.rfdata, 'echoData') and isinstance(self.rfdata.echoData, (list, tuple)) and len(self.rfdata.echoData) > 1:
                contents['echoData1'] = self.rfdata.echoData[1]
            if hasattr(self.rfdata, 'echoData') and isinstance(self.rfdata.echoData, (list, tuple)) and len(self.rfdata.echoData) > 2:
                contents['echoData2'] = self.rfdata.echoData[2]
            if hasattr(self.rfdata, 'echoData') and isinstance(self.rfdata.echoData, (list, tuple)) and len(self.rfdata.echoData) > 3:
                contents['echoData3'] = self.rfdata.echoData[3]
            if hasattr(self.rfdata, 'echoMModeData'):
                contents['echoMModeData'] = self.rfdata.echoMModeData
            if hasattr(self.rfdata, 'miscData'):
                contents['miscData'] = self.rfdata.miscData
            
            if os.path.exists(destination):
                os.remove(destination)
            savemat(destination, contents)
            logging.info(f"MATLAB file saved successfully: {destination}")
        else:
            logging.info(f"Saving as NumPy files in: {numpy_folder}")
            np.save(os.path.join(numpy_folder, 'echoData.npy'), data_to_save)
            np.save(os.path.join(numpy_folder, 'lineData.npy'), self.rfdata.lineData)
            np.save(os.path.join(numpy_folder, 'lineHeader.npy'), self.rfdata.lineHeader)
            np.save(os.path.join(numpy_folder, 'rf_data_all_fund.npy'), rf_data_all_fund)
            np.save(os.path.join(numpy_folder, 'rf_data_all_harm.npy'), rf_data_all_harm)
            logging.info("NumPy files saved successfully")
        
        result_shape = np.array(rf_data_all_fund).shape
        logging.info(f"Parsing complete. Final data shape: {result_shape}")
        
        # Clean up handlers if they were added
        if save_numpy:
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        
        return result_shape

    def _save_header_summary(self, numpy_folder: str):
        """Save a summary of header information to a text file."""
        summary_file = os.path.join(numpy_folder, 'header_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("=== PHILIPS RF FILE HEADER SUMMARY ===\n\n")
            
            # Basic file information
            if hasattr(self.rfdata, 'headerInfo') and self.rfdata.headerInfo:
                h = self.rfdata.headerInfo
                
                # Number of headers/lines
                if hasattr(h, 'Data_Type') and h.Data_Type is not None:
                    f.write(f"Number of headers/lines: {len(h.Data_Type)}\n")
                
                # First header information
                f.write("\n--- First Header Information ---\n")
                if hasattr(h, 'RF_CaptureVersion') and h.RF_CaptureVersion is not None:
                    f.write(f"RF Capture Version: {h.RF_CaptureVersion[0]}\n")
                if hasattr(h, 'Tap_Point') and h.Tap_Point is not None:
                    f.write(f"Tap Point: {h.Tap_Point[0]}\n")
                if hasattr(h, 'RF_Sample_Rate') and h.RF_Sample_Rate is not None:
                    f.write(f"RF Sample Rate: {h.RF_Sample_Rate[0]}\n")
                if hasattr(h, 'Multilines_Capture') and h.Multilines_Capture is not None:
                    f.write(f"Multilines Capture: {h.Multilines_Capture[0]}\n")
                if hasattr(h, 'Data_Gate') and h.Data_Gate is not None:
                    f.write(f"Data Gate: {h.Data_Gate[0]}\n")
                
                # Data types present
                f.write("\n--- Data Types Present ---\n")
                if hasattr(h, 'Data_Type') and h.Data_Type is not None:
                    unique_types = np.unique(h.Data_Type)
                    f.write(f"Unique Data Types: {unique_types}\n")
                    
                    # Count of each data type
                    for dtype in unique_types:
                        count = np.sum(h.Data_Type == dtype)
                        f.write(f"  Type {dtype}: {count} occurrences\n")
                
                # Frame and line information
                f.write("\n--- Frame Information ---\n")
                if hasattr(h, 'Frame_ID') and h.Frame_ID is not None:
                    unique_frames = np.unique(h.Frame_ID)
                    f.write(f"Number of unique frames: {len(unique_frames)}\n")
                    f.write(f"Frame ID range: {unique_frames.min()} to {unique_frames.max()}\n")
                
                if hasattr(h, 'Line_Index') and h.Line_Index is not None:
                    f.write(f"Line index range: {h.Line_Index.min()} to {h.Line_Index.max()}\n")
                
                # Time information
                f.write("\n--- Time Information ---\n")
                if hasattr(h, 'Time_Stamp') and h.Time_Stamp is not None:
                    f.write(f"First timestamp: {h.Time_Stamp[0]}\n")
                    f.write(f"Last timestamp: {h.Time_Stamp[-1]}\n")
            
            # Database parameters
            if hasattr(self.rfdata, 'dbParams') and self.rfdata.dbParams:
                f.write("\n--- Database Parameters ---\n")
                db = self.rfdata.dbParams
                
                if hasattr(db, 'acqNumActiveScChannels2d') and db.acqNumActiveScChannels2d:
                    f.write(f"Active scan channels: {db.acqNumActiveScChannels2d}\n")
                if hasattr(db, 'numOfSonoCTAngles2dActual') and db.numOfSonoCTAngles2dActual:
                    f.write(f"SonoCT angles: {db.numOfSonoCTAngles2dActual}\n")
                if hasattr(db, 'num2DCols') and db.num2DCols is not None:
                    f.write(f"2D columns shape: {db.num2DCols.shape}\n")
                    f.write(f"2D columns first row: {db.num2DCols[0, :] if db.num2DCols.size > 0 else 'N/A'}\n")
            
            # Calculated parameters
            if hasattr(self, 'numFrame'):
                f.write("\n--- Calculated Parameters ---\n")
                f.write(f"Number of frames: {self.numFrame}\n")
                f.write(f"TX beams per frame: {self.txBeamperFrame}\n")
                f.write(f"Number of SonoCT angles: {self.NumSonoCTAngles}\n")
                f.write(f"Multiline factor: {self.multilinefactor}\n")
                f.write(f"Used OS: {self.used_os}\n")
                f.write(f"PT: {self.pt}\n")
            
            # Data shapes
            f.write("\n--- Data Array Shapes ---\n")
            if hasattr(self.rfdata, 'lineData') and self.rfdata.lineData is not None:
                f.write(f"Line data shape: {self.rfdata.lineData.shape}\n")
            if hasattr(self.rfdata, 'lineHeader') and self.rfdata.lineHeader is not None:
                f.write(f"Line header shape: {self.rfdata.lineHeader.shape}\n")
            
            # Available data types
            f.write("\n--- Available Data Arrays ---\n")
            data_attrs = ['echoData', 'cwData', 'pwData', 'colorData', 
                         'echoMModeData', 'colorMModeData', 'dummyData', 
                         'swiData', 'miscData']
            
            for attr in data_attrs:
                if hasattr(self.rfdata, attr):
                    data = getattr(self.rfdata, attr)
                    if data is not None:
                        if isinstance(data, (list, tuple)):
                            f.write(f"{attr}: {len(data)} elements\n")
                            for i, elem in enumerate(data):
                                if hasattr(elem, 'shape'):
                                    f.write(f"  [{i}]: {elem.shape}\n")
                        elif hasattr(data, 'shape'):
                            f.write(f"{attr}: {data.shape}\n")
                        else:
                            f.write(f"{attr}: Available (unknown shape)\n")
        
        logging.info(f"Header summary saved to: {summary_file}")


###################################################################################
# Main Execution
###################################################################################
if __name__ == "__main__":
    # Configure logging for main execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Hardcoded file path - no command line arguments needed
    filepath = r"D:\Omid\0_samples\Philips\David\sample.rf"
    #filepath = r"C:\0_Main\2_Quantitative_ultrasound\2_github\test\3d.rf"
    
    logging.info(f"Starting main execution with file: {filepath}")
    parser = PhilipsRfParser()
    parser.philipsRfParser(filepath, save_numpy=True)
    logging.info("Main execution complete")
    
###################################################################################