import logging
import os
import platform
from typing import Any, List, Tuple

import numpy as np
from scipy.io import savemat
from philipsRfParser import getPartA, getPartB

###################################################################################
# Main Parser Class
###################################################################################
class PhilipsRfParser:
    """Main class for parsing Philips RF data files."""
    
    ###################################################################################
    # HeaderInfoStruct Class
    ###################################################################################
    class HeaderInfoStruct:
        """Philips-specific structure containing information from the headers."""
        def __init__(self):
            logging.debug("Initializing HeaderInfoStruct")
            
            self.RF_CaptureVersion: np.ndarray = None
            self.Tap_Point: np.ndarray = None
            self.Data_Gate: np.ndarray = None
            self.Multilines_Capture: np.ndarray = None
            self.Steer: np.ndarray = None
            self.elevationPlaneOffset: np.ndarray = None
            self.PM_Index: np.ndarray = None
            self.Pulse_Index: np.ndarray = None
            self.Data_Format: np.ndarray = None
            self.Data_Type: np.ndarray = None
            self.Header_Tag: np.ndarray = None
            self.Threed_Pos: np.ndarray = None
            self.Mode_Info: np.ndarray = None
            self.Frame_ID: np.ndarray = None
            self.CSID: np.ndarray = None
            self.Line_Index: np.ndarray = None
            self.Line_Type: np.ndarray = None
            self.Time_Stamp: np.ndarray = None
            self.RF_Sample_Rate: np.ndarray = None
            
            logging.debug("HeaderInfoStruct initialization complete")

    ###################################################################################
    # DbParams Class
    ###################################################################################
    class DbParams:
        """Philips-specific structure containing signal properties of the scan."""
        def __init__(self):
            logging.debug("Initializing DbParams")
            
            self.acqNumActiveScChannels2d: List[int] = None
            self.azimuthMultilineFactorXbrOut: List[int] = None
            self.azimuthMultilineFactorXbrIn: List[int] = None
            self.numOfSonoCTAngles2dActual: List[int] = None
            self.elevationMultilineFactor: List[int] = None
            self.numPiPulses: List[int] = None
            self.num2DCols: np.ndarray = None
            self.fastPiEnabled: List[int] = None
            self.numZones2d: List[int] = None
            self.numSubVols: Any = None
            self.numPlanes: Any = None
            self.zigZagEnabled: Any = None
            self.azimuthMultilineFactorXbrOutCf: List[int] = None
            self.azimuthMultilineFactorXbrInCf: List[int] = None
            self.multiLineFactorCf: List[int] = None
            self.linesPerEnsCf: List[int] = None
            self.ensPerSeqCf: List[int] = None
            self.numCfCols: List[int] = None
            self.numCfEntries: List[int] = None
            self.numCfDummies: List[int] = None
            self.elevationMultilineFactorCf: List[int] = None
            self.Planes: List[int] = None
            self.tapPoint: List[int] = None
            
            logging.debug("DbParams initialization complete")

    ###################################################################################
    # Rfdata Class
    ###################################################################################
    class Rfdata:
        """Philips-specific structure containing constructed RF data."""
        def __init__(self):
            logging.debug("Initializing Rfdata")
            
            self.lineData: np.ndarray = None
            self.lineHeader: np.ndarray = None
            self.headerInfo: 'PhilipsRfParser.HeaderInfoStruct' = None
            self.echoData: Any = None
            self.dbParams: 'PhilipsRfParser.DbParams' = None
            self.echoMModeData: Any = None
            self.miscData: Any = None
            self.cwData: Any = None
            self.pwData: Any = None
            self.colorData: Any = None
            self.colorMModeData: Any = None
            self.dummyData: Any = None
            self.swiData: Any = None
            
            logging.debug("Rfdata initialization complete")
    
    ###################################################################################
    # Initialize the parser with default parameters
    ###################################################################################
    def __init__(self, ML_out: int = 2, ML_in: int = 32, used_os: int = 2256):
        """Initialize the parser with default parameters."""
        logging.info(f"Initializing PhilipsRfParser with ML_out={ML_out}, ML_in={ML_in}, used_os={used_os}")
        
        self.ML_out: int = ML_out
        self.ML_in: int = ML_in
        self.used_os: int = used_os
        self.rfdata: 'PhilipsRfParser.Rfdata' = None
        self.txBeamperFrame: int = None
        self.NumSonoCTAngles: int = None
        self.numFrame: int = None
        self.multilinefactor: int = None
        self.pt: int = None
        
        # Define constant parameters for headers
        self.VHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 160, 160]
        self.FHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 11, 11]
        
        # Define data type constants
        self.DataType_ECHO = np.arange(1, 15)
        self.DataType_CW = 16
        self.DataType_PW = [18, 19]
        self.DataType_COLOR = [17, 21, 22, 23, 24]
        self.DataType_EchoMMode = 26
        self.DataType_ColorMMode = [27, 28]
        self.DataType_Dummy = [20, 25, 29, 30, 31]
        self.DataType_SWI = [90, 91]
        self.DataType_Misc = [15, 88, 89]
        
        # Define ML sort lists
        self.ML_SortList_128 = list(range(128))
        self.ML_SortList_32_CRE4 = [4, 4, 5, 5, 6, 6, 7, 7, 4, 4, 5, 5, 6, 6, 7, 7, 
                                    0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]
        self.ML_SortList_32 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.ML_SortList_16_CRE1 = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
        self.ML_SortList_16_CRE2 = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
        self.ML_SortList_16_CRE4 = [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]
        self.ML_SortList_12_CRE1 = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
        self.ML_SortList_12_CRE2 = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        self.ML_SortList_12_CRE4 = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
        self.ML_SortList_8_CRE1 = [0, 2, 4, 6, 1, 3, 5, 7]
        self.ML_SortList_8_CRE2 = [0, 1, 2, 3, 0, 1, 2, 3]
        self.ML_SortList_8_CRE4 = [0, 0, 1, 1, 0, 0, 1, 1]
        self.ML_SortList_4_CRE1 = [0, 2, 1, 3]
        self.ML_SortList_4_CRE2 = [0, 1, 0, 1]
        self.ML_SortList_4_CRE4 = [0, 0, 0, 0]
        self.ML_SortList_2_CRE1 = [0, 1]
        self.ML_SortList_2_CRE2 = [0, 0]
        self.ML_SortList_2_CRE4 = [0, 0]
        
        logging.debug("PhilipsRfParser initialization complete")
           
    ###################################################################################
    # Main Parsing Function
    ###################################################################################
    def philipsRfParser(self, filepath: str, save_numpy: bool = False) -> np.ndarray:
        """Parse Philips RF data file, save as .mat file, and return shape of data.
        If save_numpy is True, only save the processed data as .npy files in a folder named '{sample_name}_extracted' in the sample path.
        If save_numpy is False, only save as .mat file."""
        
        logging.info(f"Starting parsing of file: {filepath}")
        
        logger = self._setup_logging(filepath, save_numpy)
        
        try:
            logging.info(f"Initiating RF data parsing")
            self.rfdata = self._parse_rf(filepath, 0, 2000)
            
            # Save header summary if saving as numpy
            sample_name = os.path.splitext(os.path.basename(filepath))[0]
            numpy_folder = None
            if save_numpy:
                numpy_folder = os.path.join(os.path.dirname(filepath), f'{sample_name}_extracted')
                logging.info(f"Saving header summary to {numpy_folder}")
                self._save_header_summary(numpy_folder)
            
            data_to_save, data_type_label = self._find_primary_data()
            
            # Process data
            logging.info(f"Processing line data")
            self._preprocess_line_data()
            logging.info(f"Calculating parameters")
            self._calculate_parameters()
            logging.info(f"Filling data arrays")
            rf_data_all_fund, rf_data_all_harm = self._fill_data_arrays()
            
            # Save data in appropriate format
            if save_numpy:
                logging.info(f"Saving data as NumPy arrays")
                result_shape = self._save_numpy_data(numpy_folder, data_to_save, rf_data_all_fund, rf_data_all_harm)
            else:
                logging.info(f"Saving data as MATLAB file")
                result_shape = self._save_matlab_data(filepath, data_to_save, rf_data_all_fund, rf_data_all_harm)
            
            logging.info(f"Parsing complete. Final data shape: {result_shape}")
            return result_shape
            
        finally:
            self._restore_logging(logger, save_numpy)
    
    ###################################################################################
    # Logging Setup
    ###################################################################################
    def _setup_logging(self, filepath: str, save_numpy: bool):
        """Set up logging configuration for the parsing process."""
        # Get the root logger - it may already be configured
        logger = logging.getLogger()
        original_handlers = list(logger.handlers)  # Save original handlers
        original_level = logger.level
        
        # Create formatter for console with function name included
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
        
        if save_numpy:
            # Create numpy folder with sample name + '_extracted'
            sample_name = os.path.splitext(os.path.basename(filepath))[0]
            numpy_folder = os.path.join(os.path.dirname(filepath), f'{sample_name}_extracted')
            if not os.path.exists(numpy_folder):
                os.makedirs(numpy_folder)
            
            # Add file logging to the numpy folder with detailed output
            log_file = os.path.join(numpy_folder, 'parsing_log.txt')
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.DEBUG)  # Capture all levels for file
            
            # Use a detailed formatter that includes source location and function name
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Apply console formatter to existing handlers
            for handler in original_handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setFormatter(console_formatter)
            
            # Ensure logger level is at DEBUG to capture all messages
            logger.setLevel(logging.DEBUG)
            
            # Make console handler show only INFO and above if it existed before
            for handler in original_handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    # Save original level to restore later
                    if not hasattr(handler, '_original_level'):
                        handler._original_level = handler.level
                    handler.setLevel(logging.INFO)
            
            logging.debug(f"Detailed debug logging enabled to file: {log_file}")
        
        logging.info(f"Starting Philips RF parsing for file: {filepath}")
        logging.info(f"Save format: {'NumPy arrays' if save_numpy else 'MATLAB file'}")
        
        return {'handlers': original_handlers, 'level': original_level}
    
    ###################################################################################
    # Logging Restoration
    ###################################################################################
    def _restore_logging(self, logger_state, save_numpy: bool):
        """Restore original logging configuration."""
        if save_numpy:
            # Get the root logger
            logger = logging.getLogger()
            
            # Remove any added handlers
            current_handlers = list(logger.handlers)
            for handler in current_handlers:
                if handler not in logger_state['handlers']:
                    handler.close()
                    logger.removeHandler(handler)
            
            # Restore original handler levels
            for handler in logger_state['handlers']:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    # Restore saved original level if available
                    if hasattr(handler, '_original_level'):
                        handler.setLevel(handler._original_level)
                        delattr(handler, '_original_level')
                    else:
                        handler.setLevel(logger_state['level'])
            
            # Restore original logger level
            logger.setLevel(logger_state['level'])
            logging.debug("Logging configuration restored to original state")
    
    ###################################################################################
    # Primary Data Detection
    ###################################################################################
    def _find_primary_data(self):
        """Find the primary data type to save."""
        logging.info(f"Finding primary data type to save")
        
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
                logging.debug(f"Found data type: {label}")
                break
        
        has_valid_data = data_to_save is not None and (not hasattr(data_to_save, 'size') or data_to_save.size > 0)
        if not has_valid_data:
            error_msg = f"No supported data found in RF file. Data_Type values: {np.unique(self.rfdata.headerInfo.Data_Type) if hasattr(self.rfdata.headerInfo, 'Data_Type') else 'N/A'}. lineData shape: {self.rfdata.lineData.shape if hasattr(self.rfdata, 'lineData') else 'N/A'}"
            logging.error(f"{error_msg}")
            raise RuntimeError(error_msg)
        
        logging.info(f"Saving data type: {data_type_label} as 'echoData'")
        return data_to_save, data_type_label
    
    ###################################################################################
    # Line Data Preprocessing
    ###################################################################################
    def _preprocess_line_data(self):
        """Preprocess the line data."""
        logging.info(f"Preprocessing line data")
        
        # Log original shape
        logging.debug(f"Original lineData shape: {self.rfdata.lineData.shape}")
        
        # Data preprocessing
        if (self.rfdata.headerInfo.Line_Index[249] == self.rfdata.headerInfo.Line_Index[250]):
            logging.debug(f"Line indices 249 and 250 are equal, selecting even columns")
            self.rfdata.lineData = self.rfdata.lineData[:, np.arange(2, self.rfdata.lineData.shape[1], 2)]
        else:
            logging.debug(f"Line indices 249 and 250 differ, selecting odd columns")
            self.rfdata.lineData = self.rfdata.lineData[:, np.arange(1, self.rfdata.lineData.shape[1], 2)]
        
        # Log new shape
        logging.debug(f"Preprocessed lineData shape: {self.rfdata.lineData.shape}")
    
    ###################################################################################
    # NumPy Data Saving
    ####################################################################################
    def _save_numpy_data(self, numpy_folder, data_to_save, rf_data_all_fund, rf_data_all_harm):
        """Save data as NumPy files."""
        logging.info(f"Saving as NumPy files in: {numpy_folder}")
        
        # Save individual arrays
        logging.debug(f"Saving echoData.npy, shape: {data_to_save.shape}")
        np.save(os.path.join(numpy_folder, 'echoData.npy'), data_to_save)
        
        logging.debug(f"Saving lineData.npy, shape: {self.rfdata.lineData.shape}")
        np.save(os.path.join(numpy_folder, 'lineData.npy'), self.rfdata.lineData)
        
        logging.debug(f"Saving lineHeader.npy, shape: {self.rfdata.lineHeader.shape}")
        np.save(os.path.join(numpy_folder, 'lineHeader.npy'), self.rfdata.lineHeader)
        
        logging.debug(f"Saving rf_data_all_fund.npy, shape: {np.array(rf_data_all_fund).shape}")
        np.save(os.path.join(numpy_folder, 'rf_data_all_fund.npy'), rf_data_all_fund)
        
        logging.debug(f"Saving rf_data_all_harm.npy, shape: {np.array(rf_data_all_harm).shape}")
        np.save(os.path.join(numpy_folder, 'rf_data_all_harm.npy'), rf_data_all_harm)
        
        logging.info(f"NumPy files saved successfully")
        return np.array(rf_data_all_fund).shape

    ###################################################################################
    # MATLAB Data Saving
    ####################################################################################
    def _save_matlab_data(self, filepath, data_to_save, rf_data_all_fund, rf_data_all_harm):
        """Save data as MATLAB file."""
        destination = str(filepath[:-3] + '.mat')
        logging.info(f"Saving as MATLAB file: {destination}")
        
        # Prepare contents dictionary
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
        
        # Add optional data if available
        logging.debug(f"Adding optional data fields to MATLAB file")
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
        
        # Remove existing file if necessary
        if os.path.exists(destination):
            logging.debug(f"Removing existing file: {destination}")
            os.remove(destination)
        
        # Save the file
        savemat(destination, contents)
        logging.info(f"MATLAB file saved successfully: {destination}")
        return np.array(rf_data_all_fund).shape

    ###################################################################################
    # Main RF Parsing Orchestrator
    ###################################################################################
    def _parse_rf(self, filepath: str, read_offset: int, read_size: int) -> 'PhilipsRfParser.Rfdata':
        """Open and parse RF data file (refactored into smaller methods)."""
        logging.info(f"Opening RF file: {filepath}")
        logging.debug(f"Read parameters - offset: {read_offset}MB, size: {read_size}MB")
        rfdata = PhilipsRfParser.Rfdata()
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
        logging.debug(f"RF parsing complete")
        return rfdata

    ###################################################################################
    # Save Header Summary
    ###################################################################################
    def _save_header_summary(self, numpy_folder: str):
        """Save a summary of header information to a text file."""
        summary_file = os.path.join(numpy_folder, 'header_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("=== PHILIPS RF FILE HEADER SUMMARY ===\n\n")
            
            # Write each section of the summary
            self._write_header_info_section(f)
            self._write_db_params_section(f)
            self._write_calculated_params_section(f)
            self._write_data_shapes_section(f)
            self._write_available_data_section(f)
        
        logging.info(f"Header summary saved to: {summary_file}")

    ###################################################################################
    # Header Information Section Writing
    ###################################################################################
    def _write_header_info_section(self, file_handle):
        """Write header information section to the summary file."""
        if not hasattr(self.rfdata, 'headerInfo') or not self.rfdata.headerInfo:
            return
            
        h = self.rfdata.headerInfo
        
        # Number of headers/lines
        if hasattr(h, 'Data_Type') and h.Data_Type is not None:
            file_handle.write(f"Number of headers/lines: {len(h.Data_Type)}\n")
        
        # First header information
        file_handle.write("\n--- First Header Information ---\n")
        self._write_first_header_info(file_handle, h)
        
        # Data types present
        self._write_data_types_info(file_handle, h)
        
        # Frame and line information
        self._write_frame_info(file_handle, h)
        
        # Time information
        self._write_time_info(file_handle, h)

    ###################################################################################
    # First Header Information Section Writing
    ###################################################################################
    def _write_first_header_info(self, file_handle, h):
        """Write information from the first header."""
        header_fields = [
            ('RF_CaptureVersion', 'RF Capture Version'),
            ('Tap_Point', 'Tap Point'),
            ('RF_Sample_Rate', 'RF Sample Rate'),
            ('Multilines_Capture', 'Multilines Capture'),
            ('Data_Gate', 'Data Gate')
        ]
        
        for attr, label in header_fields:
            if hasattr(h, attr) and getattr(h, attr) is not None:
                file_handle.write(f"{label}: {getattr(h, attr)[0]}\n")

    ###################################################################################
    # Data Types Information Section Writing
    ###################################################################################
    def _write_data_types_info(self, file_handle, h):
        """Write information about data types."""
        file_handle.write("\n--- Data Types Present ---\n")
        if hasattr(h, 'Data_Type') and h.Data_Type is not None:
            unique_types = np.unique(h.Data_Type)
            file_handle.write(f"Unique Data Types: {unique_types}\n")
            
            # Count of each data type
            for dtype in unique_types:
                count = np.sum(h.Data_Type == dtype)
                file_handle.write(f"  Type {dtype}: {count} occurrences\n")

    ###################################################################################
    # Frame Information Section Writing
    ###################################################################################
    def _write_frame_info(self, file_handle, h):
        """Write information about frames and lines."""
        file_handle.write("\n--- Frame Information ---\n")
        if hasattr(h, 'Frame_ID') and h.Frame_ID is not None:
            unique_frames = np.unique(h.Frame_ID)
            file_handle.write(f"Number of unique frames: {len(unique_frames)}\n")
            file_handle.write(f"Frame ID range: {unique_frames.min()} to {unique_frames.max()}\n")
        
        if hasattr(h, 'Line_Index') and h.Line_Index is not None:
            file_handle.write(f"Line index range: {h.Line_Index.min()} to {h.Line_Index.max()}\n")

    ###################################################################################
    # Time Information Section Writing
    ###################################################################################
    def _write_time_info(self, file_handle, h):
        """Write time-related information."""
        file_handle.write("\n--- Time Information ---\n")
        if hasattr(h, 'Time_Stamp') and h.Time_Stamp is not None:
            file_handle.write(f"First timestamp: {h.Time_Stamp[0]}\n")
            file_handle.write(f"Last timestamp: {h.Time_Stamp[-1]}\n")

    ###################################################################################
    # Database Parameters Section Writing
    ###################################################################################
    def _write_db_params_section(self, file_handle):
        """Write database parameters section to the summary file."""
        if not hasattr(self.rfdata, 'dbParams') or not self.rfdata.dbParams:
            return
            
        file_handle.write("\n--- Database Parameters ---\n")
        db = self.rfdata.dbParams
        
        if hasattr(db, 'acqNumActiveScChannels2d') and db.acqNumActiveScChannels2d:
            file_handle.write(f"Active scan channels: {db.acqNumActiveScChannels2d}\n")
        if hasattr(db, 'numOfSonoCTAngles2dActual') and db.numOfSonoCTAngles2dActual:
            file_handle.write(f"SonoCT angles: {db.numOfSonoCTAngles2dActual}\n")
        if hasattr(db, 'num2DCols') and db.num2DCols is not None:
            file_handle.write(f"2D columns shape: {db.num2DCols.shape}\n")
            file_handle.write(f"2D columns first row: {db.num2DCols[0, :] if db.num2DCols.size > 0 else 'N/A'}\n")

    ###################################################################################
    # Calculated Parameters Section Writing
    ###################################################################################
    def _write_calculated_params_section(self, file_handle):
        """Write calculated parameters section to the summary file."""
        if not hasattr(self, 'numFrame'):
            return
            
        file_handle.write("\n--- Calculated Parameters ---\n")
        file_handle.write(f"Number of frames: {self.numFrame}\n")
        file_handle.write(f"TX beams per frame: {self.txBeamperFrame}\n")
        file_handle.write(f"Number of SonoCT angles: {self.NumSonoCTAngles}\n")
        file_handle.write(f"Multiline factor: {self.multilinefactor}\n")
        file_handle.write(f"Used OS: {self.used_os}\n")
        file_handle.write(f"PT: {self.pt}\n")

    ###################################################################################
    # Data Shapes Section Writing
    ###################################################################################
    def _write_data_shapes_section(self, file_handle):
        """Write data shapes section to the summary file."""
        file_handle.write("\n--- Data Array Shapes ---\n")
        if hasattr(self.rfdata, 'lineData') and self.rfdata.lineData is not None:
            file_handle.write(f"Line data shape: {self.rfdata.lineData.shape}\n")
        if hasattr(self.rfdata, 'lineHeader') and self.rfdata.lineHeader is not None:
            file_handle.write(f"Line header shape: {self.rfdata.lineHeader.shape}\n")

    ###################################################################################
    # Available Data Types Section Writing
    ###################################################################################
    def _write_available_data_section(self, file_handle):
        """Write available data types section to the summary file."""
        file_handle.write("\n--- Available Data Arrays ---\n")
        data_attrs = ['echoData', 'cwData', 'pwData', 'colorData', 
                     'echoMModeData', 'colorMModeData', 'dummyData', 
                     'swiData', 'miscData']
        
        for attr in data_attrs:
            if hasattr(self.rfdata, attr):
                data = getattr(self.rfdata, attr)
                if data is not None:
                    if isinstance(data, (list, tuple)):
                        file_handle.write(f"{attr}: {len(data)} elements\n")
                        for i, elem in enumerate(data):
                            if hasattr(elem, 'shape'):
                                file_handle.write(f"  [{i}]: {elem.shape}\n")
                    elif hasattr(data, 'shape'):
                        file_handle.write(f"{attr}: {data.shape}\n")
                    else:
                        file_handle.write(f"{attr}: Available (unknown shape)\n")

    ###################################################################################
    # Parameter Calculation
    ####################################################################################
    def _calculate_parameters(self) -> None:
        """Calculate and set main parameters as instance variables."""
        logging.info(f"Calculating parsing parameters...")
        
        # Calculate beam parameters
        self.txBeamperFrame = int(np.array(self.rfdata.dbParams.num2DCols).flat[0])
        self.NumSonoCTAngles = int(self.rfdata.dbParams.numOfSonoCTAngles2dActual[0])
        logging.info(f"Beam parameters - txBeamperFrame: {self.txBeamperFrame}, NumSonoCTAngles: {self.NumSonoCTAngles}")
        
        # Calculate frame count
        self.numFrame = int(np.floor(self.rfdata.lineData.shape[1] / (self.txBeamperFrame * self.NumSonoCTAngles)))
        self.multilinefactor = self.ML_in
        logging.info(f"Calculated numFrame: {self.numFrame}, multilinefactor: {self.multilinefactor}")
        
        # Determine OS and PT parameters
        col = 0
        if np.any(self.rfdata.lineData[:, col] != 0):
            # Auto-detect based on data
            first_nonzero = np.where(self.rfdata.lineData[:, col] != 0)[0][0]
            last_nonzero = np.where(self.rfdata.lineData[:, col] != 0)[0][-1]
            self.used_os = first_nonzero  # Override default with detected value
            self.pt = int(np.floor((last_nonzero - first_nonzero + 1) / self.multilinefactor))
            logging.info(f"Auto-detected: used_os={self.used_os}, pt={self.pt}")
        else:
            # Use the value provided in the constructor (default is 2256)
            logging.info(f"Using provided used_os={self.used_os}")
            self.pt = int(np.floor((self.rfdata.lineData.shape[0] - self.used_os) / self.multilinefactor))
            logging.debug(f"Calculated pt={self.pt} based on shape {self.rfdata.lineData.shape[0]} and multilinefactor {self.multilinefactor}")
      
    ###################################################################################
    # Data Array Filling
    ####################################################################################
    def _fill_data_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fill RF data arrays for fundamental and harmonic signals."""
        logging.info(f"Filling RF data arrays for fundamental and harmonic signals...")
        
        # Preallocate arrays
        rftemp_all_harm = np.zeros((self.pt, self.ML_out * self.txBeamperFrame))
        rftemp_all_fund = np.zeros((self.pt, self.ML_out * self.txBeamperFrame))
        rf_data_all_harm = np.zeros((self.numFrame, self.NumSonoCTAngles, self.pt, self.ML_out * self.txBeamperFrame))
        rf_data_all_fund = np.zeros((self.numFrame, self.NumSonoCTAngles, self.pt, self.ML_out * self.txBeamperFrame))
        logging.debug(f"Preallocated arrays shapes - fund: {rf_data_all_fund.shape}, harm: {rf_data_all_harm.shape}")
        
        # Process each frame
        for k0 in range(self.numFrame):
            if k0 % max(1, self.numFrame // 10) == 0:
                logging.info(f"Processing frame {k0+1}/{self.numFrame}")
            
            # Process angles within frame
            for k1 in range(self.NumSonoCTAngles):
                # Process beams within angle
                for k2 in range(self.txBeamperFrame):
                    bi = k0 * self.txBeamperFrame * self.NumSonoCTAngles + k1 * self.txBeamperFrame + k2
                    
                    # Skip if beam index exceeds available data
                    if bi >= self.rfdata.lineData.shape[1]:
                        logging.warning(f"Skipping bi={bi} as it exceeds lineData columns {self.rfdata.lineData.shape[1]}")
                        continue
                    
                    # Extract data for this beam
                    idx0 = self.used_os + np.arange(self.pt * self.multilinefactor)
                    idx1 = bi
                    
                    # Log first extraction for debugging
                    if k0 == 0 and k1 == 0 and k2 == 0:
                        logging.debug(f"First extraction - lineData[{idx0[0]}:{idx0[-1]+1}, {idx1}]")
                        logging.debug(f"lineData values sample: {self.rfdata.lineData[idx0, idx1][:10]}")
                    
                    # Reshape data for multiline
                    temp = np.transpose(
                        np.reshape(self.rfdata.lineData[idx0, idx1],
                                 (self.multilinefactor, self.pt), order='F')
                    )
                    
                    # Log first reshape for debugging
                    if k0 == 0 and k1 == 0 and k2 == 0:
                        logging.debug(f"temp shape: {temp.shape}, temp sample: {temp.ravel()[:10]}")
                    
                    # Harmonic extraction
                    if temp.shape[1] > 2:
                        rftemp_all_harm[:, np.arange(self.ML_out) + (k2 * self.ML_out)] = temp[:, [0, 2]]
                    else:
                        logging.warning(f"temp has only {temp.shape[1]} columns, skipping harmonic assignment")
                    
                    # Fundamental extraction
                    if temp.shape[1] >= 12:
                        rftemp_all_fund[:, np.arange(self.ML_out) + (k2 * self.ML_out)] = temp[:, [9, 11]]
                    elif temp.shape[1] >= 2:
                        if k0 == 0 and k1 == 0 and k2 == 0:
                            logging.warning(f"temp has only {temp.shape[1]} columns, using last 2 columns for fundamental")
                        rftemp_all_fund[:, np.arange(self.ML_out) + (k2 * self.ML_out)] = temp[:, [-2, -1]]
                    else:
                        logging.warning(f"temp has only {temp.shape[1]} columns, skipping fundamental assignment")
                
                # Store arrays for this angle
                rf_data_all_harm[k0][k1] = rftemp_all_harm
                rf_data_all_fund[k0][k1] = rftemp_all_fund
        
        logging.info(f"RF data array filling complete")
        return rf_data_all_fund, rf_data_all_harm

    ###################################################################################
    # File Type Detection and File Header
    ####################################################################################
    def _detect_file_type(self, file_obj) -> Tuple[bool, bool, int, List[int]]:
        """Detects file type and returns is_voyager, has_file_header, file_header_size, file_header."""
        logging.info(f"Starting file type detection")
        
        file_header_size = len(self.VHeader)
        
        logging.debug(f"Reading {file_header_size} bytes for header detection")
        file_header = list(file_obj.read(file_header_size))
        logging.debug(f"Read file header: {file_header[:10]}...")
        
        is_voyager = False
        has_file_header = False
        
        if file_header == self.VHeader:
            logging.info(f"Header information found - Parsing Voyager RF capture file")
            is_voyager = True
            has_file_header = True
        elif file_header == self.FHeader:
            logging.info(f"Header information found - Parsing Fusion RF capture file")
            has_file_header = True
        else:
            logging.info(f"No header found - Parsing legacy Voyager RF capture file")
            is_voyager = True
            
        logging.debug(f"File type detection complete: is_voyager={is_voyager}, has_file_header={has_file_header}")
        return is_voyager, has_file_header, file_header_size, file_header

    ###################################################################################
    # File Header Parsing
    ####################################################################################
    def _parse_file_header_and_offset(self, file_obj, is_voyager: bool, has_file_header: bool, file_header_size: int, filepath: str) -> Tuple['PhilipsRfParser.DbParams', int, str]:
        """Parse file header and calculate total_header_size, endianness, and db_params."""
        logging.info(f"Parsing file header and calculating offset")
        logging.debug(f"Input parameters: is_voyager={is_voyager}, has_file_header={has_file_header}, file_header_size={file_header_size}")
        
        endianness = 'little'
        db_params = PhilipsRfParser.DbParams()
        num_file_header_bytes = 0
        
        if has_file_header:
            if is_voyager:
                endianness = 'big'
                logging.debug(f"Using big-endian for Voyager file")
            else:
                logging.debug(f"Using little-endian for Fusion file")
                
            logging.info(f"Parsing file header parameters")
            db_params, num_file_header_bytes = self._parse_file_header(file_obj, endianness)
            total_header_size = file_header_size + 8 + num_file_header_bytes
            
            logging.debug(f"Total header size: {total_header_size} bytes (file_header={file_header_size} + 8 + params={num_file_header_bytes})")
        else:
            total_header_size = 0
            logging.debug(f"No file header to parse")
            
        logging.info(f"File header parsing complete: endianness={endianness}, total_header_size={total_header_size}")
        return db_params, total_header_size, endianness

    ###################################################################################
    # File Header Parsing
    ####################################################################################
    def _parse_file_header(self, file_obj, endianness: str) -> Tuple['PhilipsRfParser.DbParams', int]:
        """Parse file header information."""
        logging.info(f"Parsing file header information")
        
        fileVersion = int.from_bytes(file_obj.read(4), endianness, signed=False)
        numFileHeaderBytes = int.from_bytes(file_obj.read(4), endianness, signed=False)
        logging.info(f"File Version: {fileVersion}, Header Size: {numFileHeaderBytes} bytes")

        # Handle accordingly to fileVersion
        temp_dbParams = PhilipsRfParser.DbParams()
        logging.debug(f"Reading file header for version {fileVersion}")
        
        if fileVersion == 2:
            self._parse_file_header_v2(file_obj, endianness, temp_dbParams)
        elif fileVersion == 3:
            self._parse_file_header_v3(file_obj, endianness, temp_dbParams)
        elif fileVersion == 4:
            self._parse_file_header_v4(file_obj, endianness, temp_dbParams)
        elif fileVersion == 5:
            self._parse_file_header_v5(file_obj, endianness, temp_dbParams)
        elif fileVersion == 6:
            self._parse_file_header_v6(file_obj, endianness, temp_dbParams)
        else:
            numFileHeaderBytes = 0
            logging.warning(f"Unknown file version: {fileVersion}")

        logging.info(f"File header parsing complete for version {fileVersion}")
        return temp_dbParams, numFileHeaderBytes
    
    ###################################################################################
    # Helper Methods
    ###################################################################################
    def _read_int_array(self, file_obj, endianness: str, count: int) -> List[int]:
        """Helper method to read an array of integers."""
        logging.debug(f"Reading {count} integers with endianness '{endianness}'")
        
        result = [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(count)]
        
        logging.debug(f"Read {len(result)} integers, first few: {result[:min(5, len(result))]}")
        return result
    
    ###################################################################################
    # Helper Methods
    ###################################################################################
    def _read_2d_cols(self, file_obj, endianness: str) -> np.ndarray:
        """Helper method to read and reshape 2D columns data."""
        logging.debug(f"Reading 2D columns data with endianness '{endianness}'")
        
        flat_data = self._read_int_array(file_obj, endianness, 14*11)
        result = np.reshape(flat_data, (14, 11), order='F')
        
        logging.debug(f"Reshaped data to shape: {result.shape}")
        return result
    
    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _parse_file_header_v2(self, file_obj, endianness: str, temp_dbParams: 'PhilipsRfParser.DbParams') -> None:
        """Parse file header version 2."""
        logging.debug("Reading file header version 2")
        
        # Basic parameters (arrays of 4)
        temp_dbParams.acqNumActiveScChannels2d = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.azimuthMultilineFactorXbrOut = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.azimuthMultilineFactorXbrIn = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numOfSonoCTAngles2dActual = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.elevationMultilineFactor = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numPiPulses = self._read_int_array(file_obj, endianness, 4)
        
        # 2D columns
        temp_dbParams.num2DCols = self._read_2d_cols(file_obj, endianness)
        
        # Additional parameters
        temp_dbParams.fastPiEnabled = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numZones2d = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numSubVols = self._read_int_array(file_obj, endianness, 1)
        temp_dbParams.numPlanes = self._read_int_array(file_obj, endianness, 1)
        temp_dbParams.zigZagEnabled = self._read_int_array(file_obj, endianness, 1)
    
    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _parse_file_header_v3(self, file_obj, endianness: str, temp_dbParams: 'PhilipsRfParser.DbParams') -> None:
        """Parse file header version 3."""
        logging.debug("Reading file header version 3")
        
        # Basic parameters (arrays of 4)
        temp_dbParams.acqNumActiveScChannels2d = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.azimuthMultilineFactorXbrOut = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.azimuthMultilineFactorXbrIn = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numOfSonoCTAngles2dActual = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.elevationMultilineFactor = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numPiPulses = self._read_int_array(file_obj, endianness, 4)
        
        # 2D columns
        temp_dbParams.num2DCols = self._read_2d_cols(file_obj, endianness)
        
        # Additional parameters
        temp_dbParams.fastPiEnabled = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numZones2d = self._read_int_array(file_obj, endianness, 4)
        
        # Single values
        temp_dbParams.numSubVols = int.from_bytes(file_obj.read(4), endianness, signed=False)
        temp_dbParams.numPlanes = int.from_bytes(file_obj.read(4), endianness, signed=False)
        temp_dbParams.zigZagEnabled = int.from_bytes(file_obj.read(4), endianness, signed=False)
        
        # Color flow parameters
        temp_dbParams.multiLineFactorCf = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.linesPerEnsCf = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.ensPerSeqCf = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numCfCols = self._read_int_array(file_obj, endianness, 14)
        temp_dbParams.numCfEntries = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numCfDummies = self._read_int_array(file_obj, endianness, 4)
    
    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _parse_file_header_v4(self, file_obj, endianness: str, temp_dbParams: 'PhilipsRfParser.DbParams') -> None:
        """Parse file header version 4."""
        logging.debug("Reading file header version 4")
        
        # Basic parameters (arrays of 3)
        temp_dbParams.acqNumActiveScChannels2d = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.azimuthMultilineFactorXbrOut = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.azimuthMultilineFactorXbrIn = self._read_int_array(file_obj, endianness, 3)

        # Color flow azimuth parameters
        temp_dbParams.azimuthMultilineFactorXbrOutCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.azimuthMultilineFactorXbrInCf = self._read_int_array(file_obj, endianness, 3)

        # Additional basic parameters
        temp_dbParams.numOfSonoCTAngles2dActual = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.elevationMultilineFactor = self._read_int_array(file_obj, endianness, 3)

        # Color flow elevation parameters
        temp_dbParams.elevationMultilineFactorCf = self._read_int_array(file_obj, endianness, 3)

        # More basic parameters
        temp_dbParams.numPiPulses = self._read_int_array(file_obj, endianness, 3)
        
        # 2D columns
        temp_dbParams.num2DCols = self._read_2d_cols(file_obj, endianness)
        
        # Additional parameters
        temp_dbParams.fastPiEnabled = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numZones2d = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numSubVols = self._read_int_array(file_obj, endianness, 1)

        # Planes instead of numPlanes
        temp_dbParams.Planes = self._read_int_array(file_obj, endianness, 1)

        # More parameters
        temp_dbParams.zigZagEnabled = self._read_int_array(file_obj, endianness, 1)

        # Color flow parameters
        temp_dbParams.linesPerEnsCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.ensPerSeqCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfCols = self._read_int_array(file_obj, endianness, 14)
        temp_dbParams.numCfEntries = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfDummies = self._read_int_array(file_obj, endianness, 3)
    
    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _parse_file_header_v5(self, file_obj, endianness: str, temp_dbParams: 'PhilipsRfParser.DbParams') -> None:
        """Parse file header version 5."""
        logging.debug("Reading file header version 5")
        
        # Basic parameters (arrays of 3)
        temp_dbParams.acqNumActiveScChannels2d = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.azimuthMultilineFactorXbrOut = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.azimuthMultilineFactorXbrIn = self._read_int_array(file_obj, endianness, 3)

        # Color flow azimuth parameters
        temp_dbParams.azimuthMultilineFactorXbrOutCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.azimuthMultilineFactorXbrInCf = self._read_int_array(file_obj, endianness, 3)

        # Additional basic parameters
        temp_dbParams.numOfSonoCTAngles2dActual = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.elevationMultilineFactor = self._read_int_array(file_obj, endianness, 3)

        # Color flow elevation and multiline parameters
        temp_dbParams.elevationMultilineFactorCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.multiLineFactorCf = self._read_int_array(file_obj, endianness, 3)

        # More basic parameters
        temp_dbParams.numPiPulses = self._read_int_array(file_obj, endianness, 3)
        
        # 2D columns
        temp_dbParams.num2DCols = self._read_2d_cols(file_obj, endianness)
        
        # Additional parameters
        temp_dbParams.fastPiEnabled = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numZones2d = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numSubVols = self._read_int_array(file_obj, endianness, 1)

        temp_dbParams.numPlanes = self._read_int_array(file_obj, endianness, 1)

        temp_dbParams.zigZagEnabled = self._read_int_array(file_obj, endianness, 1)

        # Color flow parameters
        temp_dbParams.linesPerEnsCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.ensPerSeqCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfCols = self._read_int_array(file_obj, endianness, 14)
        temp_dbParams.numCfEntries = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfDummies = self._read_int_array(file_obj, endianness, 3)
    
    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _parse_file_header_v6(self, file_obj, endianness: str, temp_dbParams: 'PhilipsRfParser.DbParams') -> None:
        """Parse file header version 6."""
        logging.debug("Reading file header version 6")
        
        # Tap point parameter
        temp_dbParams.tapPoint = self._read_int_array(file_obj, endianness, 1)
        
        # Basic parameters (arrays of 3)
        temp_dbParams.acqNumActiveScChannels2d = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.azimuthMultilineFactorXbrOut = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.azimuthMultilineFactorXbrIn = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.azimuthMultilineFactorXbrOutCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.azimuthMultilineFactorXbrInCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numOfSonoCTAngles2dActual = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.elevationMultilineFactor = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.elevationMultilineFactorCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.multiLineFactorCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numPiPulses = self._read_int_array(file_obj, endianness, 3)
        
        # 2D columns
        temp_dbParams.num2DCols = self._read_2d_cols(file_obj, endianness)
        
        # Additional parameters
        temp_dbParams.fastPiEnabled = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numZones2d = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numSubVols = self._read_int_array(file_obj, endianness, 1)
        temp_dbParams.numPlanes = self._read_int_array(file_obj, endianness, 1)
        temp_dbParams.zigZagEnabled = self._read_int_array(file_obj, endianness, 1)
        
        # Color flow parameters
        temp_dbParams.linesPerEnsCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.ensPerSeqCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfCols = self._read_int_array(file_obj, endianness, 14)
        temp_dbParams.numCfEntries = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfDummies = self._read_int_array(file_obj, endianness, 3)

    ###################################################################################
    # Raw Data Loading
    ###################################################################################
    def _load_raw_rf_data(self, filepath: str, is_voyager: bool, total_header_size: int, read_offset: int, read_size: int) -> Tuple[Any, int]:
        """Load raw RF data from file, handling Voyager and Fusion formats."""
        logging.info(f"Loading raw RF data: is_voyager={is_voyager}, offset={read_offset}MB, size={read_size}MB")
        
        # Calculate file sizes and convert units
        file_size, remaining_size, read_offset_bytes, read_size_bytes = self._calculate_read_parameters(
            filepath, total_header_size, read_offset, read_size
        )
        
        # Load data based on format
        if is_voyager:
            return self._load_voyager_data(filepath, remaining_size, read_offset_bytes, read_size_bytes)
        else:
            return self._load_fusion_data(filepath, total_header_size, remaining_size, read_offset_bytes, read_size_bytes)

    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _calculate_read_parameters(self, filepath: str, total_header_size: int, read_offset: int, read_size: int) -> Tuple[int, int, int, int]:
        """Calculate file sizes and convert MB to bytes for read parameters."""
        file_size = os.stat(filepath).st_size
        remaining_size = file_size - total_header_size
        logging.debug(f"File size: {file_size} bytes, header size: {total_header_size} bytes, remaining: {remaining_size} bytes")
        
        # Convert from MB to bytes
        read_offset_bytes = read_offset * (2 ** 20)
        read_size_bytes = read_size * (2 ** 20)
        logging.debug(f"Read parameters in bytes: offset={read_offset_bytes}, size={read_size_bytes}")
        
        return file_size, remaining_size, read_offset_bytes, read_size_bytes

    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _load_voyager_data(self, filepath: str, remaining_size: int, read_offset: int, read_size: int) -> Tuple[Any, int]:
        """Load data in Voyager format."""
        logging.info("Loading Voyager format data")
        
        # Align read parameters to Voyager data format
        read_offset, read_size = self._align_voyager_parameters(remaining_size, read_offset, read_size)
        
        # Read the raw data
        with open(filepath, 'rb') as f:
            f.seek(read_offset)
            rawrfdata = f.read(read_size)
        
        logging.info(f"Loaded {len(rawrfdata)} bytes of Voyager data")
        return rawrfdata, 0

    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _align_voyager_parameters(self, remaining_size: int, read_offset: int, read_size: int) -> Tuple[int, int]:
        """Align read parameters to Voyager format boundaries (36 bytes)."""
        alignment = np.arange(0, remaining_size + 1, 36)
        offset_diff = alignment - read_offset
        read_diff = alignment - read_size
        
        aligned_offset = alignment[np.where(offset_diff >= 0)[0][0]].__int__()
        aligned_size = alignment[np.where(read_diff >= 0)[0][0]].__int__()
        
        logging.debug(f"Aligned Voyager read - offset: {aligned_offset}, size: {aligned_size}")
        return aligned_offset, aligned_size

    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _load_fusion_data(self, filepath: str, total_header_size: int, remaining_size: int, read_offset: int, read_size: int) -> Tuple[Any, int]:
        """Load data in Fusion format."""
        logging.info("Loading Fusion format data")
        
        # Align read parameters to Fusion data format
        read_offset, read_size = self._align_fusion_parameters(remaining_size, read_offset, read_size)
        
        # Calculate number of clumps and final offset
        num_clumps = int(np.floor(read_size / 32))
        offset = total_header_size + read_offset
        logging.info(f"Reading Fusion data: {num_clumps} clumps from offset {offset}")
        
        # Read and process the data
        rawrfdata = self._read_and_process_fusion_data(filepath, offset, num_clumps)
        
        logging.info(f"Loaded Fusion data with shape {rawrfdata.shape}")
        return rawrfdata, num_clumps

    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _align_fusion_parameters(self, remaining_size: int, read_offset: int, read_size: int) -> Tuple[int, int]:
        """Align read parameters to Fusion format boundaries (32 bytes)."""
        alignment = np.arange(0, remaining_size + 1, 32)
        offset_diff = alignment - read_offset
        read_diff = alignment - read_size
        
        # Find matching offset
        matching_indices = np.where(offset_diff >= 0)[0]
        if len(matching_indices) > 0:
            aligned_offset = alignment[matching_indices[0]].__int__()
        else:
            aligned_offset = 0
            logging.warning("No matching offset found, using 0")
        
        # Find matching size
        matching_indices = np.where(read_diff >= 0)[0]
        if len(matching_indices) > 0:
            aligned_size = alignment[matching_indices[0]].__int__()
        else:
            aligned_size = remaining_size
            logging.warning(f"No matching size found, using remaining size: {aligned_size}")
        
        logging.debug(f"Aligned Fusion read - offset: {aligned_offset}, size: {aligned_size}")
        return aligned_offset, aligned_size

    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _read_and_process_fusion_data(self, filepath: str, offset: int, num_clumps: int) -> np.ndarray:
        """Read and process Fusion format data using external functions."""
        # External functions from philipsRfParser module
        partA = getPartA(num_clumps, filepath, offset)
        partB = getPartB(num_clumps, filepath, offset)
        logging.debug(f"Retrieved partA: {len(partA)} elements, partB: {len(partB)} elements")
        
        # Process and reshape the data
        rawrfdata = np.concatenate((
            np.array(partA, dtype=int).reshape((12, num_clumps), order='F'), 
            np.array([partB], dtype=int)
        ))
        logging.debug(f"Raw RF data shape: {rawrfdata.shape}")
        
        return rawrfdata

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
    def _parse_header_dispatch(self, rawrfdata: Any, is_voyager: bool) -> 'PhilipsRfParser.HeaderInfoStruct':
        """Dispatch to the correct header parsing method."""
        logging.info(f"Dispatching header parsing: is_voyager={is_voyager}")
        
        if is_voyager:
            logging.debug(f"Using Voyager header parser")
            return self._parse_header_v(rawrfdata)
        else:
            logging.debug(f"Using Fusion header parser")
            return self._parse_header_f(rawrfdata)

    ###################################################################################
    # Voyager Header Parsing
    ###################################################################################
    def _parse_header_v(self, rawrfdata):
        """Parse header for Voyager systems."""
        logging.info("Parsing Voyager header information")
        
        # Find headers and initialize structure
        temp_headerInfo = PhilipsRfParser.HeaderInfoStruct()
        iHeader, numHeaders = self._find_voyager_headers(rawrfdata)
        
        # Initialize header arrays
        temp_headerInfo = self._initialize_header_arrays(temp_headerInfo, numHeaders)
        
        # Process each header
        self._process_voyager_headers(rawrfdata, iHeader, numHeaders, temp_headerInfo)
        
        logging.info(f"Voyager header parsing complete - processed {numHeaders} headers")
        return temp_headerInfo
    
    ###################################################################################
    # Voyager Header Parsing
    ###################################################################################
    def _find_voyager_headers(self, rawrfdata):
        """Find headers in Voyager data."""
        iHeader = np.where(np.uint8(rawrfdata[2,0,:])&224)
        numHeaders = len(iHeader)-1  # Ignore last header as it is part of a partial line
        logging.debug(f"Found {numHeaders} headers in Voyager data")
        return iHeader, numHeaders

    ###################################################################################
    # Voyager Header Parsing
    ###################################################################################
    def _initialize_header_arrays(self, header_info, numHeaders):
        """Initialize header arrays with appropriate sizes and types."""
        # Initialize 8-bit fields
        header_info.RF_CaptureVersion = np.zeros(numHeaders, dtype=np.uint8)
        header_info.Tap_Point = np.zeros(numHeaders, dtype=np.uint8)
        header_info.Data_Gate = np.zeros(numHeaders, dtype=np.uint8)
        header_info.Multilines_Capture = np.zeros(numHeaders, dtype=np.uint8)
        header_info.RF_Sample_Rate = np.zeros(numHeaders, dtype=np.uint8)
        header_info.Steer = np.zeros(numHeaders, dtype=np.uint8)
        header_info.elevationPlaneOffset = np.zeros(numHeaders, dtype=np.uint8)
        header_info.PM_Index = np.zeros(numHeaders, dtype=np.uint8)
        
        # Initialize 16-bit fields
        header_info.Line_Index = np.zeros(numHeaders, dtype=np.uint16)
        header_info.Pulse_Index = np.zeros(numHeaders, dtype=np.uint16)
        header_info.Data_Format = np.zeros(numHeaders, dtype=np.uint16)
        header_info.Data_Type = np.zeros(numHeaders, dtype=np.uint16)
        header_info.Header_Tag = np.zeros(numHeaders, dtype=np.uint16)
        header_info.Threed_Pos = np.zeros(numHeaders, dtype=np.uint16)
        header_info.Mode_Info = np.zeros(numHeaders, dtype=np.uint16)
        header_info.CSID = np.zeros(numHeaders, dtype=np.uint16)
        header_info.Line_Type = np.zeros(numHeaders, dtype=np.uint16)
        
        # Initialize 32-bit fields
        header_info.Frame_ID = np.zeros(numHeaders, dtype=np.uint32)
        header_info.Time_Stamp = np.zeros(numHeaders, dtype=np.uint32)
        
        return header_info

    ###################################################################################
    # Voyager Header Parsing
    ###################################################################################
    def _process_voyager_headers(self, rawrfdata, iHeader, numHeaders, temp_headerInfo):
        """Process each header in Voyager data."""
        for m in range(numHeaders):
            if m % 1000 == 0:
                logging.debug(f"Processing Voyager header {m}/{numHeaders}")
            
            # Build packed header string from raw data
            packedHeader = self._build_voyager_packed_header(rawrfdata, iHeader, m)
            
            # Parse the header values
            self._parse_voyager_header_values(packedHeader, m, temp_headerInfo)

    ###################################################################################
    # Voyager Header Parsing
    ###################################################################################
    def _build_voyager_packed_header(self, rawrfdata, iHeader, m):
        """Build packed header string from raw data."""
        packedHeader = ''
        for k in np.arange(11, 0, -1):
            temp = ''
            for i in np.arange(2, 0, -1):
                temp += bin(np.uint8(rawrfdata[i, k, iHeader[m]]))
            # Discard first 3 bits, redundant info
            packedHeader += temp[3:24]
        return packedHeader

    ###################################################################################
    # Voyager Header Parsing
    ###################################################################################
    def _parse_voyager_header_values(self, packedHeader, m, header_info):
        """Parse values from packed header string."""
        iBit = 0
        
        # Parse 8-bit fields
        header_info.RF_CaptureVersion[m] = int(packedHeader[iBit:iBit+4], 2)
        iBit += 4
        header_info.Tap_Point[m] = int(packedHeader[iBit:iBit+3], 2)
        iBit += 3
        header_info.Data_Gate[m] = int(packedHeader[iBit], 2)
        iBit += 1
        header_info.Multilines_Capture[m] = int(packedHeader[iBit:iBit+4], 2)
        iBit += 4
        header_info.RF_Sample_Rate[m] = int(packedHeader[iBit], 2)
        iBit += 1
        
        # Log sample rate for first header
        if m == 0:
            logging.info(f"Sample rate from first Voyager header: {header_info.RF_Sample_Rate[m]}")
        
        header_info.Steer[m] = int(packedHeader[iBit:iBit+6], 2)
        iBit += 6
        header_info.elevationPlaneOffset[m] = int(packedHeader[iBit:iBit+8], 2)
        iBit += 8
        header_info.PM_Index[m] = int(packedHeader[iBit:iBit+2], 2)
        iBit += 2
        
        # Parse 16-bit fields
        header_info.Line_Index[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        header_info.Pulse_Index[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        header_info.Data_Format[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        header_info.Data_Type[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        header_info.Header_Tag[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        header_info.Threed_Pos[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        header_info.Mode_Info[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        
        # Parse 32-bit fields
        header_info.Frame_ID[m] = int(packedHeader[iBit:iBit+32], 2)
        iBit += 32
        header_info.CSID[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        header_info.Line_Type[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        header_info.Time_Stamp[m] = int(packedHeader[iBit:iBit+32], 2)

    ###################################################################################
    # Fusion Header Parsing
    ###################################################################################
    def _parse_header_f(self, rawrfdata):
        """Parse header for Fusion systems."""
        logging.info('Entering parseHeaderF - parsing Fusion headers')
        
        # Find headers and initialize structure
        iHeader, numHeaders = self._find_fusion_headers(rawrfdata)
        HeaderInfo = PhilipsRfParser.HeaderInfoStruct()
        
        # Initialize arrays
        HeaderInfo = self._initialize_header_arrays(HeaderInfo, numHeaders)
        
        # Process each header
        self._process_fusion_headers(rawrfdata, iHeader, numHeaders, HeaderInfo)
        
        logging.info(f'Exiting parseHeaderF - numHeaders: {numHeaders}, Data_Type shape: {HeaderInfo.Data_Type.shape}')
        return HeaderInfo

    ###################################################################################
    # Fusion Header Parsing
    ###################################################################################
    def _find_fusion_headers(self, rawrfdata):
        """Find headers in Fusion data."""
        # Find header clumps
        # iHeader pts to the index of the header clump
        # Note that each header is exactly 1 "Clump" long
        iHeader = np.array(np.where(rawrfdata[0,:]&1572864 == 524288))[0]
        numHeaders = iHeader.size - 1  # Ignore last header as it is a part of a partial line
        logging.info(f"Found {numHeaders} headers in Fusion data")
        return iHeader, numHeaders

    ###################################################################################
    # Fusion Header Parsing
    ###################################################################################
    def _process_fusion_headers(self, rawrfdata, iHeader, numHeaders, HeaderInfo):
        """Process each header in Fusion data."""
        logging.info("Extracting header information...")
        for m in range(numHeaders):
            if m % 1000 == 0:
                logging.debug(f"Processing Fusion header {m}/{numHeaders}")
            
            # Build packed header string from raw data
            packedHeader = self._build_fusion_packed_header(rawrfdata, iHeader, m)
            
            # Parse the header values
            self._parse_fusion_header_values(packedHeader, m, HeaderInfo)

    ###################################################################################
    # Fusion Header Parsing
    ###################################################################################
    def _build_fusion_packed_header(self, rawrfdata, iHeader, m):
        """Build packed header string from raw data for Fusion systems."""
        # Get the data from the 13th element (index 12)
        packedHeader = bin(rawrfdata[12, iHeader[m]])[2:]
        
        # Add leading zeros if needed
        remainingZeros = 4 - len(packedHeader)
        if remainingZeros > 0:
            zeros = self._get_filler_zeros(remainingZeros)
            packedHeader = str(zeros + packedHeader)
        
        # Add data from remaining elements in reverse order
        for i in np.arange(11, -1, -1):
            curBin = bin(int(rawrfdata[i, iHeader[m]]))[2:]
            remainingZeros = 21 - len(curBin)
            if remainingZeros > 0:
                zeros = self._get_filler_zeros(remainingZeros)
                curBin = str(zeros + curBin)
            packedHeader += curBin
        
        return packedHeader

    ###################################################################################
    # Fusion Header Parsing
    ###################################################################################
    def _parse_fusion_header_values(self, packedHeader, m, HeaderInfo):
        """Parse values from packed header string for Fusion systems."""
        iBit = 2  # Start at bit 2
        
        # Parse 8-bit fields
        HeaderInfo.RF_CaptureVersion[m] = int(packedHeader[iBit:iBit+4], 2)
        iBit += 4
        HeaderInfo.Tap_Point[m] = int(packedHeader[iBit:iBit+3], 2)
        iBit += 3
        HeaderInfo.Data_Gate[m] = int(packedHeader[iBit], 2)
        iBit += 1
        HeaderInfo.Multilines_Capture[m] = int(packedHeader[iBit:iBit+4], 2)
        iBit += 4
        iBit += 15  # Skip 15 unused bits
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
        
        # Parse 16-bit fields
        HeaderInfo.Line_Index[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Pulse_Index[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Data_Format[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Data_Type[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Header_Tag[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Threed_Pos[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Mode_Info[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        
        # Parse 32-bit fields
        HeaderInfo.Frame_ID[m] = int(packedHeader[iBit:iBit+32], 2)
        iBit += 32
        HeaderInfo.CSID[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        HeaderInfo.Line_Type[m] = int(packedHeader[iBit:iBit+16], 2)
        iBit += 16
        
        # Special handling for Time_Stamp - concatenate specific bit ranges
        HeaderInfo.Time_Stamp[m] = int(str(packedHeader[iBit:iBit+13]+packedHeader[iBit+15:iBit+34]), 2)

    ###################################################################################
    # RF Data Parsing Dispatch
    ###################################################################################
    def _parse_rf_data_dispatch(self, rawrfdata: Any, header_info: 'PhilipsRfParser.HeaderInfoStruct', is_voyager: bool) -> Tuple[np.ndarray, np.ndarray, int]:
        """Dispatch to the correct RF data parsing method."""
        logging.info(f"Dispatching RF data parsing: is_voyager={is_voyager}")
        return self._parse_rf_data(rawrfdata, header_info, is_voyager)

    ###################################################################################
    # RF Data Parsing
    ###################################################################################
    def _parse_rf_data(self, rawrfdata, headerInfo: 'PhilipsRfParser.HeaderInfoStruct', isVoyager: bool) -> Tuple[np.ndarray, np.ndarray, int]:
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
    # Voyager Data Parsing
    ###################################################################################
    def _parse_data_v(self, rawrfdata, headerInfo: 'PhilipsRfParser.HeaderInfoStruct') -> Tuple[np.ndarray, np.ndarray]:
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
    def _parse_data_f(self, rawrfdata, headerInfo: 'PhilipsRfParser.HeaderInfoStruct') -> Tuple[np.ndarray, np.ndarray]:
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
    # Data Type Organization
    ####################################################################################
    def _organize_data_types(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Organize data types (echo, color, etc.) and assign them to rfdata."""
        logging.info(f"Organizing data types, tap_point={tap_point}")
        logging.debug(f"Available data types in headers: {np.unique(header_info.Data_Type) if hasattr(header_info, 'Data_Type') else 'N/A'}")
        
        # Extract different data types
        logging.debug(f"Extracting echo data")
        rfdata = self._extract_echo_data(rfdata, header_info, tap_point)
        
        logging.debug(f"Extracting CW data")
        rfdata = self._extract_cw_data(rfdata, header_info, tap_point)
        
        logging.debug(f"Extracting PW data")
        rfdata = self._extract_pw_data(rfdata, header_info, tap_point)
        
        logging.debug(f"Extracting color data")
        rfdata = self._extract_color_data(rfdata, header_info, tap_point)
        
        logging.debug(f"Extracting echo M-mode data")
        rfdata = self._extract_echo_mmode_data(rfdata, header_info, tap_point)
        
        logging.debug(f"Extracting color M-mode data")
        rfdata = self._extract_color_mmode_data(rfdata, header_info, tap_point)
        
        logging.debug(f"Extracting dummy data")
        rfdata = self._extract_dummy_data(rfdata, header_info, tap_point)
        
        logging.debug(f"Extracting SWI data")
        rfdata = self._extract_swi_data(rfdata, header_info, tap_point)
        
        logging.debug(f"Extracting miscellaneous data")
        rfdata = self._extract_misc_data(rfdata, header_info, tap_point)
        
        logging.info(f"Data type organization complete")
        return rfdata

    ###################################################################################
    # Echo Data Extraction
    ###################################################################################
    def _extract_echo_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract echo data from RF data."""
        logging.info(f"Extracting echo data, tap_point={tap_point}")
        
        # Get multiline capture settings and adjust tap point if needed
        ML_Capture, tap_point = self._get_echo_capture_settings(header_info, tap_point)
        
        # Find echo data indices in the dataset
        echo_index, echo_count = self._find_echo_data_indices(header_info)
        
        # Process echo data if any was found
        if echo_count > 0:
            rfdata = self._process_echo_data(rfdata, header_info, echo_index, ML_Capture, tap_point)
            logging.info(f"Echo data extracted successfully, shape: {rfdata.echoData[0].shape if hasattr(rfdata.echoData, '__getitem__') else 'N/A'}")
        else:
            logging.warning("No echo data found")
        
        return rfdata

    ###################################################################################
    # Echo Data Extraction
    ###################################################################################
    def _get_echo_capture_settings(self, header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> Tuple[float, int]:
        """Get multiline capture settings and adjust tap point if needed."""
        logging.debug(f"Echo data types: {self.DataType_ECHO}")
        
        # Determine ML_Capture based on tap point and multilines capture setting
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        
        # If ML_Capture is 0, determine it from sample rate
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            logging.info(f"RF Sample Rate: {SAMPLE_RATE}")
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32  # 20MHz Capture
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
        
        logging.debug(f"ML_Capture={ML_Capture}")
        
        # Adjust tap point if needed
        if tap_point == 7:
            tap_point = 4
            logging.debug("Tap point 7 converted to 4")
        
        return ML_Capture, tap_point

    ###################################################################################
    # Echo Data Extraction
    ###################################################################################
    def _find_echo_data_indices(self, header_info: 'PhilipsRfParser.HeaderInfoStruct') -> Tuple[np.ndarray, int]:
        """Find indices of echo data in the dataset."""
        xmit_events = len(header_info.Data_Type)
        logging.debug(f"Number of transmit events: {xmit_events}")
        
        # Build index of echo data types
        echo_index = np.zeros(xmit_events).astype(np.int32)
        for dt in self.DataType_ECHO:
            index = ((header_info.Data_Type & 255) == dt)
            echo_index = np.bitwise_or(echo_index, np.array(index).astype(np.int32))
        
        echo_count = np.sum(echo_index)
        logging.debug(f"Found {echo_count} echo data entries")
        
        # Fallback to Data_Type=1 if no echo data found
        if echo_count == 0 and np.any(header_info.Data_Type == 1):
            echo_index = (header_info.Data_Type == 1).astype(np.int32)
            echo_count = np.sum(echo_index)
            logging.debug(f"Fallback: Found {echo_count} entries with Data_Type=1")
        
        return echo_index, echo_count

    ###################################################################################
    # Echo Data Extraction
    ###################################################################################
    def _process_echo_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', 
                          echo_index: np.ndarray, ML_Capture: float, tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Process echo data and assign it to rfdata."""
        # Get columns to keep based on echo index
        columns_to_keep = np.where(echo_index == 1)[0]
        logging.debug(f"Processing {len(columns_to_keep)} echo columns")
        
        # Extract line data for echo columns
        pruning_line_data = rfdata.lineData[:, columns_to_keep]
        pruning_line_header = rfdata.lineHeader[:, columns_to_keep]
        logging.debug(f"Pruning data shape: {pruning_line_data.shape}")
        
        # Get echo data, either directly or pruned
        echo_data = self._get_echo_data_for_tap_point(rfdata, pruning_line_data, pruning_line_header, ML_Capture, tap_point)
        
        # Apply RF sorting based on tap point
        rfdata = self._sort_echo_data_by_tap_point(rfdata, echo_data, ML_Capture, tap_point)
        
        return rfdata

    ###################################################################################
    # Echo Data Extraction
    ###################################################################################
    def _get_echo_data_for_tap_point(self, rfdata: 'PhilipsRfParser.Rfdata', pruning_line_data: np.ndarray, 
                                    pruning_line_header: np.ndarray, ML_Capture: float, tap_point: int) -> np.ndarray:
        """Get echo data for the specified tap point."""
        if tap_point == 4:
            echo_data = pruning_line_data
            logging.debug("Using data directly (tap_point=4)")
        else:
            logging.debug("Pruning data before use")
            echo_data = self._prune_data(pruning_line_data, pruning_line_header, ML_Capture)
        
        return echo_data

    ###################################################################################
    # Echo Data Extraction
    ###################################################################################
    def _sort_echo_data_by_tap_point(self, rfdata: 'PhilipsRfParser.Rfdata', echo_data: np.ndarray, 
                                    ML_Capture: float, tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Sort echo data based on tap point and assign to rfdata."""
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
        
        return rfdata

    ###################################################################################
    # CW Data Extraction
    ###################################################################################
    def _extract_cw_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract CW data from RF data."""
        logging.info(f"Extracting CW data, tap_point={tap_point}")
        logging.debug(f"CW data type: {self.DataType_CW}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            
            # Log the sample rate when it's used to determine ML_Capture
            logging.info(f"RF Sample Rate: {SAMPLE_RATE}")
            
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
        
        logging.debug(f"ML_Capture={ML_Capture}")
        
        cw_index = (header_info.Data_Type == self.DataType_CW)
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
    def _extract_pw_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract PW data from RF data."""
        logging.info(f"Extracting PW data, tap_point={tap_point}")
        logging.debug(f"PW data types: {self.DataType_PW}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
            
        logging.debug(f"ML_Capture={ML_Capture}")
        
        xmit_events = len(header_info.Data_Type)
        pw_index = np.zeros(xmit_events).astype(bool)
        
        for dt in self.DataType_PW:
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
    def _extract_color_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract color data from RF data."""
        logging.info(f"Extracting color data, tap_point={tap_point}")
        logging.debug(f"Color data types: {self.DataType_COLOR}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
            
        logging.debug(f"ML_Capture={ML_Capture}")
        
        xmit_events = len(header_info.Data_Type)
        color_index = np.zeros(xmit_events).astype(bool)
        
        for dt in self.DataType_COLOR:
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
    def _extract_echo_mmode_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract echo M-Mode data from RF data."""
        logging.info(f"Extracting echo M-Mode data, tap_point={tap_point}")
        logging.debug(f"Echo M-Mode data type: {self.DataType_EchoMMode}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
            
        logging.debug(f"ML_Capture={ML_Capture}")
        
        echo_mmode_index = (header_info.Data_Type == self.DataType_EchoMMode)
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
    def _extract_color_mmode_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract color M-Mode data from RF data."""
        logging.info(f"Extracting color M-Mode data, tap_point={tap_point}")
        logging.debug(f"Color M-Mode data types: {self.DataType_ColorMMode}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
            
        logging.debug(f"ML_Capture={ML_Capture}")
        
        xmit_events = len(header_info.Data_Type)
        color_mmode_index = np.zeros(xmit_events).astype(bool)
        
        for dt in self.DataType_ColorMMode:
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
    def _extract_dummy_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract dummy data from RF data."""
        logging.info(f"Extracting dummy data, tap_point={tap_point}")
        logging.debug(f"Dummy data types: {self.DataType_Dummy}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
            logging.debug(f"ML_Capture was 0, set to {ML_Capture} based on sample rate {SAMPLE_RATE}")
            
        logging.debug(f"ML_Capture={ML_Capture}")
        
        xmit_events = len(header_info.Data_Type)
        dummy_index = np.zeros(xmit_events).astype(bool)
        
        for dt in self.DataType_Dummy:
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
    def _extract_swi_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract SWI data from RF data."""
        logging.info(f"Extracting SWI data, tap_point={tap_point}")
        logging.debug(f"SWI data types: {self.DataType_SWI}")
        
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
        xmit_events = len(header_info.Data_Type)
        swi_index = np.zeros(xmit_events).astype(bool)
        for dt in self.DataType_SWI:
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
    def _extract_misc_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract miscellaneous data from RF data."""
        logging.debug(f"Misc data types: {self.DataType_Misc}")
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if ML_Capture == 0:
            SAMPLE_RATE = float(header_info.RF_Sample_Rate[0])
            ML_Capture = 16 if SAMPLE_RATE == 0 else 32
        xmit_events = len(header_info.Data_Type)
        misc_index = np.zeros(xmit_events).astype(bool)
        for dt in self.DataType_Misc:
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
        
        # Initialize dimensions and output arrays
        N, xmitEvents, depth, MLs = self._initialize_rf_sort_dimensions(RFinput, Stride, ML)
        
        # Initialize output arrays based on CRE
        out0, out1, out2, out3 = self._initialize_rf_sort_outputs(depth, ML, xmitEvents, CRE)
        
        # Get the ML sort list for specified Stride and CRE
        ML_SortList = self._get_ml_sort_list(Stride, CRE)
        
        # Check for potential issues
        self._check_ml_sort_validity(ML, ML_SortList, CRE, Stride)
        
        # Fill output arrays based on the sort list
        out0, out1, out2, out3 = self._fill_rf_sort_outputs(
            RFinput, out0, out1, out2, out3, 
            MLs, ML_SortList, depth, Stride, ML, CRE
        )
        
        logging.info(f"RF sorting complete - output shape: {out0.shape}")
        return out0, out1, out2, out3

    ###################################################################################
    # RF Sorting Utility
    ###################################################################################
    def _initialize_rf_sort_dimensions(self, RFinput, Stride, ML):
        """Initialize dimensions for RF sorting."""
        logging.debug(f"Initializing dimensions - input shape: {RFinput.shape}, Stride: {Stride}, ML: {ML}")
        
        # Calculate dimensions
        N = RFinput.shape[0]
        xmitEvents = RFinput.shape[1]
        depth = int(np.floor(N/Stride))
        
        # Create array of multiline indices
        MLs = np.arange(0, ML)
        
        logging.debug(f"Calculated dimensions - N: {N}, xmitEvents: {xmitEvents}, depth: {depth}, MLs range: 0-{ML-1}")
        return N, xmitEvents, depth, MLs

    ###################################################################################
    # RF Sorting Utility
    ###################################################################################
    def _initialize_rf_sort_outputs(self, depth, ML, xmitEvents, CRE):
        """Initialize output arrays based on CRE value."""
        logging.debug(f"Initializing output arrays - depth: {depth}, ML: {ML}, xmitEvents: {xmitEvents}, CRE: {CRE}")
        
        # Initialize arrays with empty values
        out0 = out1 = out2 = out3 = np.array([])
        
        # Initialize array shape for logging
        array_shape = (depth, ML, xmitEvents)
        array_size_mb = (depth * ML * xmitEvents * 4) / (1024 * 1024)  # Assuming 4 bytes per element
        
        # Create arrays based on CRE
        if CRE == 4:
            logging.debug(f"Creating 4 output arrays of shape {array_shape} (~{array_size_mb:.2f}MB each)")
            out3 = np.zeros(array_shape)
            out2 = np.zeros(array_shape)
            out1 = np.zeros(array_shape)
            out0 = np.zeros(array_shape)
        elif CRE == 3:
            logging.debug(f"Creating 3 output arrays of shape {array_shape} (~{array_size_mb:.2f}MB each)")
            out2 = np.zeros(array_shape)
            out1 = np.zeros(array_shape)
            out0 = np.zeros(array_shape)
        elif CRE == 2:
            logging.debug(f"Creating 2 output arrays of shape {array_shape} (~{array_size_mb:.2f}MB each)")
            out1 = np.zeros(array_shape)
            out0 = np.zeros(array_shape)
        elif CRE == 1:
            logging.debug(f"Creating 1 output array of shape {array_shape} (~{array_size_mb:.2f}MB)")
            out0 = np.zeros(array_shape)
        else:
            logging.warning(f"Unsupported CRE value: {CRE}, using CRE=1")
            logging.debug(f"Creating 1 output array of shape {array_shape} (~{array_size_mb:.2f}MB)")
            out0 = np.zeros(array_shape)
        
        logging.debug(f"Output arrays initialized successfully")
        return out0, out1, out2, out3

    ###################################################################################
    # RF Sorting Utility
    ###################################################################################
    def _get_ml_sort_list(self, Stride, CRE):
        """Get the appropriate ML sort list based on Stride and CRE."""
        logging.debug(f"Getting ML sort list for Stride={Stride}, CRE={CRE}")
        
        # Initialize empty list
        ML_SortList = []
        
        # Select sort list based on Stride and CRE values
        if Stride == 128:
            ML_SortList = self.ML_SortList_128
        elif Stride == 32:
            if CRE == 4:
                ML_SortList = self.ML_SortList_32_CRE4
            else:
                ML_SortList = self.ML_SortList_32
        elif Stride == 16:
            if CRE == 1:
                ML_SortList = self.ML_SortList_16_CRE1
            elif CRE == 2:
                ML_SortList = self.ML_SortList_16_CRE2
            elif CRE == 4:
                ML_SortList = self.ML_SortList_16_CRE4
        elif Stride == 12:
            if CRE == 1:
                ML_SortList = self.ML_SortList_12_CRE1
            elif CRE == 2:
                ML_SortList = self.ML_SortList_12_CRE2
            elif CRE == 4:
                ML_SortList = self.ML_SortList_12_CRE4
        elif Stride == 8:
            if CRE == 1:
                ML_SortList = self.ML_SortList_8_CRE1
            elif CRE == 2:
                ML_SortList = self.ML_SortList_8_CRE2
            elif CRE == 4:
                ML_SortList = self.ML_SortList_8_CRE4
        elif Stride == 4:
            if CRE == 1:
                ML_SortList = self.ML_SortList_4_CRE1
            elif CRE == 2:
                ML_SortList = self.ML_SortList_4_CRE2
            elif CRE == 4:
                ML_SortList = self.ML_SortList_4_CRE4
        elif Stride == 2:
            if CRE == 1:
                ML_SortList = self.ML_SortList_2_CRE1
            elif CRE == 2:
                ML_SortList = self.ML_SortList_2_CRE2
            elif CRE == 4:
                ML_SortList = self.ML_SortList_2_CRE4
        else:
            logging.warning(f"No sort list for Stride={Stride}")
        
        logging.debug(f"Using ML_SortList with {len(ML_SortList)} elements")
        return ML_SortList

    ###################################################################################
    # RF Sorting Utility
    ###################################################################################
    def _check_ml_sort_validity(self, ML, ML_SortList, CRE, Stride):
        """Check if the ML sort list is valid for the requested parameters."""
        logging.debug(f"Checking ML sort list validity - ML: {ML}, CRE: {CRE}, Stride: {Stride}")
        
        # Check if sort list is empty
        if not ML_SortList:
            logging.warning(f"Empty ML_SortList for Stride={Stride}, CRE={CRE}")
            return
        
        # Log sort list properties
        logging.debug(f"ML_SortList - length: {len(ML_SortList)}, min: {min(ML_SortList)}, max: {max(ML_SortList)}")
        
        # Check if ML value exceeds what's available in the sort list
        if ((ML-1) > max(ML_SortList)):
            logging.warning(f"ML ({ML}) exceeds max value in ML_SortList ({max(ML_SortList)})")
        
        # Check for special configuration issues
        if (CRE == 4 and Stride < 16):
            logging.warning(f"Insufficient ML capture for CRE=4 with Stride={Stride} (should be >= 16)")
            
        if (CRE == 2 and Stride < 4):
            logging.warning(f"Insufficient ML capture for CRE=2 with Stride={Stride} (should be >= 4)")
        
        logging.debug(f"ML sort list validity check complete")

    ###################################################################################
    # RF Sorting Utility
    ###################################################################################
    def _fill_rf_sort_outputs(self, RFinput, out0, out1, out2, out3, MLs, ML_SortList, depth, Stride, ML, CRE):
        """Fill output arrays based on the sort list."""
        logging.info(f"Filling output arrays - ML: {ML}, CRE: {CRE}, depth: {depth}")
        
        # Skip if sort list is empty
        if not ML_SortList:
            logging.warning(f"Empty ML_SortList, unable to fill output arrays")
            return out0, out1, out2, out3
        
        # Log the first few items in the sort list
        preview_length = min(10, len(ML_SortList))
        logging.debug(f"Using ML_SortList (first {preview_length}): {ML_SortList[:preview_length]}")
        
        # Store matches for logging
        matches_found = 0
        ml_not_found = []
        
        # Process each multiline index
        for k in range(ML):
            logging.debug(f"Processing ML index {k} of {ML}")
            
            # Get indices in sort list that match current multiline index
            iML = np.where(np.array(ML_SortList) == MLs[k])[0]
            
            # Skip if no matching indices found
            if len(iML) == 0:
                logging.warning(f"No matching indices for ML={MLs[k]} in sort list")
                ml_not_found.append(MLs[k])
                continue
            
            matches_found += 1
            logging.debug(f"Found {len(iML)} matches for ML={MLs[k]} at indices {iML}")
            
            # Fill primary output array
            self._fill_output_array(out0, RFinput, depth, k, iML[0], Stride)
            
            # Fill additional output arrays based on CRE
            if CRE >= 2 and len(iML) > 1:
                logging.debug(f"Filling CRE={CRE} outputs for ML={MLs[k]}")
                self._fill_output_array(out1, RFinput, depth, k, iML[1], Stride)
                
                # These are duplicated for backward compatibility
                if out2.size > 0:
                    self._fill_output_array(out2, RFinput, depth, k, iML[1], Stride)
                if out3.size > 0:
                    self._fill_output_array(out3, RFinput, depth, k, iML[1], Stride)
            
            # Fill tertiary and quaternary output arrays for CRE=4
            if CRE == 4 and len(iML) > 3:
                logging.debug(f"Filling CRE=4 tertiary and quaternary outputs for ML={MLs[k]}")
                self._fill_output_array(out2, RFinput, depth, k, iML[2], Stride)
                self._fill_output_array(out3, RFinput, depth, k, iML[3], Stride)
        
        # Log summary statistics
        logging.info(f"Output array filling complete - {matches_found}/{ML} multilines processed")
        if ml_not_found:
            logging.warning(f"Missing multilines: {ml_not_found}")
        
        return out0, out1, out2, out3

    ###################################################################################
    # RF Sorting Utility
    ###################################################################################
    def _fill_output_array(self, output_array, input_array, depth, k, iML_index, Stride):
        """Fill a specific output array with data from the input array."""
                
        # Check if output array is valid
        if output_array.size == 0:
            logging.debug(f"Skipping fill operation - output_array is empty")
            return
        
        # Create indices for strided access
        indices = np.arange(iML_index, (depth*Stride), Stride)
        
        # Log diagnostics about the fill operation
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Filling output array[:, {k}, :] from input_array[{indices[0]}:{indices[-1]}:{Stride}]")
            logging.debug(f"Number of indices: {len(indices)}, expected depth: {depth}")
        
        # Perform the fill operation
        output_array[:depth, k, :] = input_array[indices]

    ###################################################################################
    # Utility Functions
    ###################################################################################
    @staticmethod
    def _get_filler_zeros(num: int) -> str:
        """Get string of zeros for padding."""
        logging.debug(f"Creating filler zeros, num={num}")
        
        # Ensure we don't create negative length strings
        count = max(0, num - 1)
        result = '0' * count
        
        logging.debug(f"Generated {len(result)} filler zeros")
        return result

   
###################################################################################
# Main Execution
###################################################################################
if __name__ == "__main__":
    # === Logging Configuration for Complete Debug Output ===
    # Configure the root logger to capture absolutely everything
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Capture all levels
    
    # Create a detailed formatter that includes function names
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s'
    )
    
    # Console handler with full debug output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Show everything, including DEBUG
    console_handler.setFormatter(detailed_formatter)
    
    # Clear any existing handlers and add our handler
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Log system information for diagnostic purposes
    logging.debug("==== Debug Logging Activated ====")
    logging.debug(f"Python version: {platform.python_version()}")
    logging.debug(f"Platform: {platform.platform()}")
    logging.debug(f"Current directory: {os.getcwd()}")
    
    # Hardcoded file path - no command line arguments needed
    filepath = r"D:\Omid\0_samples\Philips\David\sample.rf"
    #filepath = r"C:\0_Main\2_Quantitative_ultrasound\2_github\test\3d.rf"
    
    logging.info(f"Starting main execution with file: {filepath}")
    parser = PhilipsRfParser()
    parser.philipsRfParser(filepath, save_numpy=True)
    logging.info("Main execution complete")
      
    ###################################################################################