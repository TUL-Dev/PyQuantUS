import logging
import os
import platform
from logging.handlers import RotatingFileHandler
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
            
            self.acqNumActiveScChannels2d: np.ndarray = None
            self.azimuthMultilineFactorXbrOut: np.ndarray = None
            self.azimuthMultilineFactorXbrIn: np.ndarray = None
            self.numOfSonoCTAngles2dActual: np.ndarray = None
            self.elevationMultilineFactor: np.ndarray = None
            self.numPiPulses: np.ndarray = None
            self.num2DCols: np.ndarray = None
            self.fastPiEnabled: np.ndarray = None
            self.numZones2d: np.ndarray = None
            self.numSubVols: np.ndarray = None
            self.numPlanes: np.ndarray = None
            self.zigZagEnabled: np.ndarray = None
            self.azimuthMultilineFactorXbrOutCf: np.ndarray = None
            self.azimuthMultilineFactorXbrInCf: np.ndarray = None
            self.multiLineFactorCf: np.ndarray = None
            self.linesPerEnsCf: np.ndarray = None
            self.ensPerSeqCf: np.ndarray = None
            self.numCfCols: np.ndarray = None
            self.numCfEntries: np.ndarray = None
            self.numCfDummies: np.ndarray = None
            self.elevationMultilineFactorCf: np.ndarray = None
            self.Planes: np.ndarray = None
            self.tapPoint: np.ndarray = None
            
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
    def __init__(self, multiline_output: int = 2, multiline_input: int = 32, offset_samples: int = 2256):
        """Initialize the parser with default parameters.
        
        Args:
            multiline_output: Output multiline factor (default: 2)
            multiline_input: Input multiline factor (default: 32)
            offset_samples: Number of samples to offset in the data (default: 2256)
        """
        logging.info(f"Initializing PhilipsRfParser with multiline_output={multiline_output}, multiline_input={multiline_input}, offset_samples={offset_samples}")
        
        self.use_c = True
        
        self.multiline_output: int = multiline_output  # Formerly ML_out
        self.multiline_input: int = multiline_input    # Formerly ML_in
        self.offset_samples: int = offset_samples      # Formerly used_os
        self.rfdata: 'PhilipsRfParser.Rfdata' = None
        self.tx_beams_per_frame: int = None            # Formerly txBeamperFrame
        self.num_sonoct_angles: int = None             # Formerly NumSonoCTAngles
        self.num_frames: int = None                    # Formerly numFrame
        self.multiline_factor: int = None              # Formerly multilinefactor
        self.points_per_line: int = None               # Formerly pt
        self.read_offset_MB: int = 0
        self.read_size_MB: int = 2000
        self.is_voyager = None
        self.is_fusion = None
        self.has_file_header = None
        self.file_header_size = None
        
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
            self.rfdata = self._parse_rf(filepath, read_offset_MB=self.read_offset_MB, read_size_MB=self.read_size_MB)
            
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
            
            logging.debug(f"Detailed debug logging enabled to file: {log_file}")
        
        logging.info(f"Starting Philips RF parsing for file: {filepath}")
        logging.info(f"Save format: {'NumPy arrays' if save_numpy else 'MATLAB file'}")
        
        return {'handlers': original_handlers, 'level': original_level}
    
    ###################################################################################
    # Main RF Parsing Orchestrator
    ###################################################################################
    def _parse_rf(self, filepath: str, read_offset_MB: int, read_size_MB: int) -> 'PhilipsRfParser.Rfdata':
        """Open and parse RF data file (refactored into smaller methods)."""
        
        # Log the start of the RF file opening process
        logging.info(f"Opening RF file: {filepath}")
        logging.debug(f"Read parameters - offset: {read_offset_MB}MB, size: {read_size_MB}MB")
        
        # Initialize an Rfdata object to store parsed data
        rfdata = PhilipsRfParser.Rfdata()
        
        # Open the RF data file in binary read mode
        with open(filepath, 'rb') as file_obj:
            # Detect the file type and set the appropriate flags
            self._detect_file_type(file_obj)
            # Parse the file header and calculate the total header size
            db_params, total_header_size_bytes = self._parse_file_header_and_offset(file_obj)
        
        # Assign the parsed database parameters to the rfdata object
        rfdata.dbParams = db_params
        
        # Load the raw RF data from the file
        rawrfdata, _ = self._load_raw_rf_data(filepath, total_header_size_bytes, read_offset_MB, read_size_MB)
        
        # If the data is from a Voyager system, reshape it accordingly
        if self.is_voyager:
            rawrfdata = self._reshape_voyager_raw_data(rawrfdata)
        
        # Parse the header information from the raw RF data
        header_info = self._parse_header_dispatch(rawrfdata)
        
        # Parse the RF data and extract line data, line header, and tap point
        line_data, line_header, tap_point = self._parse_rf_data_dispatch(rawrfdata, header_info)
        
        # Assign the parsed line data, line header, and header info to the rfdata object
        rfdata.lineData = line_data
        rfdata.lineHeader = line_header
        rfdata.headerInfo = header_info
        
        # Delete the raw RF data to free up memory
        del rawrfdata
        
        # Organize the data types within the rfdata object
        rfdata = self._organize_data_types(rfdata, header_info, tap_point)
        
        # Log the completion of the RF parsing process
        logging.debug(f"RF parsing complete")
        logging.debug(f"RF data shape: {rfdata.lineData.shape}")
        
        # Return the populated rfdata object
        return rfdata

    ###################################################################################
    # Detect the file type
    ####################################################################################
    def _detect_file_type(self, file_obj) -> None:
        """
        Detects the type of the RF data file by reading its header and comparing it to known header patterns.
        Sets the is_voyager, is_fusion, and has_file_header flags based on detection.
        
        This method works by reading a signature of bytes from the beginning of the file and 
        comparing it to known patterns (VHeader and FHeader). These signature arrays each contain 
        20 elements, where each element represents a single byte value (0-255).
        
        The file_header_size is set to the length of the signature array (20). Since each element
        in the signature array represents exactly one byte to read from the file, the array length 
        directly corresponds to the number of bytes that need to be read for comparison.
        
        After reading file_header_size bytes, the method compares them to:
        - VHeader: signature for Voyager RF capture files
        - FHeader: signature for Fusion RF capture files
        
        If no match is found, it's treated as a legacy Voyager file with no header.
        
        Args:
            file_obj: The file object to read from
        """
        self.file_header_size = len(self.VHeader)
        logging.debug(f"File header size: {self.file_header_size}")
        
        # Read the header bytes from the file
        file_header_bytes = list(file_obj.read(self.file_header_size))
        logging.debug(f"Read file header: {file_header_bytes}")
        
        # Initialize file type flags
        self.is_voyager = False
        self.is_fusion = False
        self.has_file_header = False
        
        # Determine file type based on header pattern
        if file_header_bytes == self.VHeader:
            logging.info(f"Header information found - Parsing Voyager RF capture file")
            self.is_voyager = True
            self.has_file_header = True
        elif file_header_bytes == self.FHeader:
            logging.info(f"Header information found - Parsing Fusion RF capture file")
            self.is_fusion = True
            self.has_file_header = True
        else:
            logging.info(f"No header found - Parsing legacy Voyager RF capture file")
            self.is_voyager = True
            
        logging.debug(
            f"File type detection complete: "
            f"is_voyager={self.is_voyager}, "
            f"is_fusion={self.is_fusion}, "
            f"has_file_header={self.has_file_header}, "
            f"file_header_size={self.file_header_size}"
        )
        
    ###################################################################################
    # Parse the file header and calculate total_header_size, endianness, and db_params.
    ####################################################################################
    def _parse_file_header_and_offset(self, file_obj) -> Tuple['PhilipsRfParser.DbParams', int, str]:
        """Parse file header and calculate total_header_size, endianness, and db_params.
        
        Endianness refers to the order of bytes in binary data. In this context, it determines 
        how multi-byte data is read from the file.
        'Little-endian' means the least significant byte is stored first, while
        'big-endian' means the most significant byte is stored first. 
        The endianness is set based on the file type: 'big-endian' for Voyager files and 
        'little-endian' for Fusion files.
        
        The total header size in bytes is calculated as:
            - file_header_size: The length of signature array (VHeader or FHeader), which is 20.
              Since each element of the array represents one byte value (0-255), the array length
              matches exactly the number of bytes read from the file.
            - 8 bytes: 4 bytes for file version + 4 bytes for header size information
            - num_file_header_bytes: The size of the parameter data as indicated in the header
        
        Args:
            file_obj: The file object to read from

        Returns:
            Tuple containing:
                - Database parameters
                - Total header size in bytes
                - Endianness ('big' or 'little')
        """
        
        # Log the start of the parsing process
        logging.info(f"Parsing file header and calculating offset")
        
        # Initialize default values for endianness, database parameters, and number of file header bytes
        endianness = 'little'
        db_params = PhilipsRfParser.DbParams()
        num_file_header_bytes = 0
        
        # Check if the file has a header to process
        if self.has_file_header:
            # Determine endianness based on the file type
            if self.is_voyager:
                endianness = 'big'
                logging.debug(f"Using big-endian for Voyager file")
            else:
                logging.debug(f"Using little-endian for Fusion file")
                
            logging.info(f"Parsing file header parameters")
            
            # Parse the file header and get the database parameters and number of header bytes
            db_params, num_file_header_bytes = self._parse_file_header(file_obj, endianness)
            logging.debug(f"Number of file header bytes: {num_file_header_bytes}")
            
            # Calculate the total header size in bytes:
            # - file_header_bytes: signature bytes (20)
            # - 8 bytes: 4 for version + 4 for header size info
            # - num_file_header_bytes: actual parameters
            total_header_size_bytes = self.file_header_size + 8 + num_file_header_bytes
            
            logging.debug(f"Total header size: {total_header_size_bytes} bytes (file_header={self.file_header_size} + 8 + params={num_file_header_bytes})")
        else:
            # If no file header is present, set total header size to 0
            total_header_size_bytes = 0
            logging.debug(f"No file header to parse")
            
        # Log the completion of file header parsing
        logging.info(f"File header parsing complete: endianness={endianness}, total_header_size={total_header_size_bytes}")
        # Return the database parameters and total header size
        return db_params, total_header_size_bytes

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
        temp_dbParams.numSubVols = self._read_int_array(file_obj, endianness, 1)[0]
        temp_dbParams.numPlanes = self._read_int_array(file_obj, endianness, 1)[0]
        temp_dbParams.zigZagEnabled = self._read_int_array(file_obj, endianness, 1)[0]
        
        # Log all parameters at the end
        logging.debug(f"Set acqNumActiveScChannels2d: {temp_dbParams.acqNumActiveScChannels2d}")
        logging.debug(f"Set azimuthMultilineFactorXbrOut: {temp_dbParams.azimuthMultilineFactorXbrOut}")
        logging.debug(f"Set azimuthMultilineFactorXbrIn: {temp_dbParams.azimuthMultilineFactorXbrIn}")
        logging.debug(f"Set numOfSonoCTAngles2dActual: {temp_dbParams.numOfSonoCTAngles2dActual}")
        logging.debug(f"Set elevationMultilineFactor: {temp_dbParams.elevationMultilineFactor}")
        logging.debug(f"Set numPiPulses: {temp_dbParams.numPiPulses}")
        logging.debug(f"Set num2DCols: {temp_dbParams.num2DCols}")
        logging.debug(f"Set fastPiEnabled: {temp_dbParams.fastPiEnabled}")
        logging.debug(f"Set numZones2d: {temp_dbParams.numZones2d}")
        logging.debug(f"Set numSubVols: {temp_dbParams.numSubVols}")
        logging.debug(f"Set numPlanes: {temp_dbParams.numPlanes}")
        logging.debug(f"Set zigZagEnabled: {temp_dbParams.zigZagEnabled}")
        logging.debug("File header version 2 parsing complete")
    
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
        temp_dbParams.numSubVols = self._read_int_array(file_obj, endianness, 1)[0]
        temp_dbParams.numPlanes = self._read_int_array(file_obj, endianness, 1)[0]
        temp_dbParams.zigZagEnabled = self._read_int_array(file_obj, endianness, 1)[0]
        
        # Color flow parameters
        temp_dbParams.multiLineFactorCf = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.linesPerEnsCf = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.ensPerSeqCf = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numCfCols = self._read_int_array(file_obj, endianness, 14)
        temp_dbParams.numCfEntries = self._read_int_array(file_obj, endianness, 4)
        temp_dbParams.numCfDummies = self._read_int_array(file_obj, endianness, 4)
        
        # Log all parameters at the end
        logging.debug(f"Set acqNumActiveScChannels2d: {temp_dbParams.acqNumActiveScChannels2d}")
        logging.debug(f"Set azimuthMultilineFactorXbrOut: {temp_dbParams.azimuthMultilineFactorXbrOut}")
        logging.debug(f"Set azimuthMultilineFactorXbrIn: {temp_dbParams.azimuthMultilineFactorXbrIn}")
        logging.debug(f"Set numOfSonoCTAngles2dActual: {temp_dbParams.numOfSonoCTAngles2dActual}")
        logging.debug(f"Set elevationMultilineFactor: {temp_dbParams.elevationMultilineFactor}")
        logging.debug(f"Set numPiPulses: {temp_dbParams.numPiPulses}")
        logging.debug(f"Set num2DCols: {temp_dbParams.num2DCols}")
        logging.debug(f"Set fastPiEnabled: {temp_dbParams.fastPiEnabled}")
        logging.debug(f"Set numZones2d: {temp_dbParams.numZones2d}")
        logging.debug(f"Set numSubVols: {temp_dbParams.numSubVols}")
        logging.debug(f"Set numPlanes: {temp_dbParams.numPlanes}")
        logging.debug(f"Set zigZagEnabled: {temp_dbParams.zigZagEnabled}")
        logging.debug(f"Set multiLineFactorCf: {temp_dbParams.multiLineFactorCf}")
        logging.debug(f"Set linesPerEnsCf: {temp_dbParams.linesPerEnsCf}")
        logging.debug(f"Set ensPerSeqCf: {temp_dbParams.ensPerSeqCf}")
        logging.debug(f"Set numCfCols: {temp_dbParams.numCfCols}")
        logging.debug(f"Set numCfEntries: {temp_dbParams.numCfEntries}")
        logging.debug(f"Set numCfDummies: {temp_dbParams.numCfDummies}")
        logging.debug("File header version 3 parsing complete")
    
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
        temp_dbParams.numSubVols = self._read_int_array(file_obj, endianness, 1)[0]

        # Planes instead of numPlanes
        temp_dbParams.Planes = self._read_int_array(file_obj, endianness, 1)[0]

        # More parameters
        temp_dbParams.zigZagEnabled = self._read_int_array(file_obj, endianness, 1)[0]

        # Color flow parameters
        temp_dbParams.linesPerEnsCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.ensPerSeqCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfCols = self._read_int_array(file_obj, endianness, 14)
        temp_dbParams.numCfEntries = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfDummies = self._read_int_array(file_obj, endianness, 3)
        
        # Log all parameters at the end
        logging.debug(f"Set acqNumActiveScChannels2d: {temp_dbParams.acqNumActiveScChannels2d}")
        logging.debug(f"Set azimuthMultilineFactorXbrOut: {temp_dbParams.azimuthMultilineFactorXbrOut}")
        logging.debug(f"Set azimuthMultilineFactorXbrIn: {temp_dbParams.azimuthMultilineFactorXbrIn}")
        logging.debug(f"Set azimuthMultilineFactorXbrOutCf: {temp_dbParams.azimuthMultilineFactorXbrOutCf}")
        logging.debug(f"Set azimuthMultilineFactorXbrInCf: {temp_dbParams.azimuthMultilineFactorXbrInCf}")
        logging.debug(f"Set numOfSonoCTAngles2dActual: {temp_dbParams.numOfSonoCTAngles2dActual}")
        logging.debug(f"Set elevationMultilineFactor: {temp_dbParams.elevationMultilineFactor}")
        logging.debug(f"Set elevationMultilineFactorCf: {temp_dbParams.elevationMultilineFactorCf}")
        logging.debug(f"Set numPiPulses: {temp_dbParams.numPiPulses}")
        logging.debug(f"Set num2DCols: {temp_dbParams.num2DCols}")
        logging.debug(f"Set fastPiEnabled: {temp_dbParams.fastPiEnabled}")
        logging.debug(f"Set numZones2d: {temp_dbParams.numZones2d}")
        logging.debug(f"Set numSubVols: {temp_dbParams.numSubVols}")
        logging.debug(f"Set Planes: {temp_dbParams.Planes}")
        logging.debug(f"Set zigZagEnabled: {temp_dbParams.zigZagEnabled}")
        logging.debug(f"Set linesPerEnsCf: {temp_dbParams.linesPerEnsCf}")
        logging.debug(f"Set ensPerSeqCf: {temp_dbParams.ensPerSeqCf}")
        logging.debug(f"Set numCfCols: {temp_dbParams.numCfCols}")
        logging.debug(f"Set numCfEntries: {temp_dbParams.numCfEntries}")
        logging.debug(f"Set numCfDummies: {temp_dbParams.numCfDummies}")
        logging.debug("File header version 4 parsing complete")
    
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
        temp_dbParams.numSubVols = self._read_int_array(file_obj, endianness, 1)[0]

        temp_dbParams.numPlanes = self._read_int_array(file_obj, endianness, 1)[0]

        temp_dbParams.zigZagEnabled = self._read_int_array(file_obj, endianness, 1)[0]

        # Color flow parameters
        temp_dbParams.linesPerEnsCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.ensPerSeqCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfCols = self._read_int_array(file_obj, endianness, 14)
        temp_dbParams.numCfEntries = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfDummies = self._read_int_array(file_obj, endianness, 3)
        
        # Log all parameters at the end
        logging.debug(f"Set acqNumActiveScChannels2d: {temp_dbParams.acqNumActiveScChannels2d}")
        logging.debug(f"Set azimuthMultilineFactorXbrOut: {temp_dbParams.azimuthMultilineFactorXbrOut}")
        logging.debug(f"Set azimuthMultilineFactorXbrIn: {temp_dbParams.azimuthMultilineFactorXbrIn}")
        logging.debug(f"Set azimuthMultilineFactorXbrOutCf: {temp_dbParams.azimuthMultilineFactorXbrOutCf}")
        logging.debug(f"Set azimuthMultilineFactorXbrInCf: {temp_dbParams.azimuthMultilineFactorXbrInCf}")
        logging.debug(f"Set numOfSonoCTAngles2dActual: {temp_dbParams.numOfSonoCTAngles2dActual}")
        logging.debug(f"Set elevationMultilineFactor: {temp_dbParams.elevationMultilineFactor}")
        logging.debug(f"Set elevationMultilineFactorCf: {temp_dbParams.elevationMultilineFactorCf}")
        logging.debug(f"Set multiLineFactorCf: {temp_dbParams.multiLineFactorCf}")
        logging.debug(f"Set numPiPulses: {temp_dbParams.numPiPulses}")
        logging.debug(f"Set num2DCols: {temp_dbParams.num2DCols}")
        logging.debug(f"Set fastPiEnabled: {temp_dbParams.fastPiEnabled}")
        logging.debug(f"Set numZones2d: {temp_dbParams.numZones2d}")
        logging.debug(f"Set numSubVols: {temp_dbParams.numSubVols}")
        logging.debug(f"Set numPlanes: {temp_dbParams.numPlanes}")
        logging.debug(f"Set zigZagEnabled: {temp_dbParams.zigZagEnabled}")
        logging.debug(f"Set linesPerEnsCf: {temp_dbParams.linesPerEnsCf}")
        logging.debug(f"Set ensPerSeqCf: {temp_dbParams.ensPerSeqCf}")
        logging.debug(f"Set numCfCols: {temp_dbParams.numCfCols}")
        logging.debug(f"Set numCfEntries: {temp_dbParams.numCfEntries}")
        logging.debug(f"Set numCfDummies: {temp_dbParams.numCfDummies}")
        logging.debug("File header version 5 parsing complete")
    
    ###################################################################################
    # File Header Parsing
    ###################################################################################
    def _parse_file_header_v6(self, file_obj, endianness: str, temp_dbParams: 'PhilipsRfParser.DbParams') -> None:
        """Parse file header version 6."""
        logging.debug("Reading file header version 6")
        
        # Tap point parameter
        temp_dbParams.tapPoint = self._read_int_array(file_obj, endianness, 1)[0]
        
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
        temp_dbParams.numSubVols = self._read_int_array(file_obj, endianness, 1)[0]
        temp_dbParams.numPlanes = self._read_int_array(file_obj, endianness, 1)[0]
        temp_dbParams.zigZagEnabled = self._read_int_array(file_obj, endianness, 1)[0]
        
        # Color flow parameters
        temp_dbParams.linesPerEnsCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.ensPerSeqCf = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfCols = self._read_int_array(file_obj, endianness, 14)
        temp_dbParams.numCfEntries = self._read_int_array(file_obj, endianness, 3)
        temp_dbParams.numCfDummies = self._read_int_array(file_obj, endianness, 3)
        
        # Log all parameters at the end
        logging.debug(f"Set tapPoint: {temp_dbParams.tapPoint}")
        logging.debug(f"Set acqNumActiveScChannels2d: {temp_dbParams.acqNumActiveScChannels2d}")
        logging.debug(f"Set azimuthMultilineFactorXbrOut: {temp_dbParams.azimuthMultilineFactorXbrOut}")
        logging.debug(f"Set azimuthMultilineFactorXbrIn: {temp_dbParams.azimuthMultilineFactorXbrIn}")
        logging.debug(f"Set azimuthMultilineFactorXbrOutCf: {temp_dbParams.azimuthMultilineFactorXbrOutCf}")
        logging.debug(f"Set azimuthMultilineFactorXbrInCf: {temp_dbParams.azimuthMultilineFactorXbrInCf}")
        logging.debug(f"Set numOfSonoCTAngles2dActual: {temp_dbParams.numOfSonoCTAngles2dActual}")
        logging.debug(f"Set elevationMultilineFactor: {temp_dbParams.elevationMultilineFactor}")
        logging.debug(f"Set elevationMultilineFactorCf: {temp_dbParams.elevationMultilineFactorCf}")
        logging.debug(f"Set multiLineFactorCf: {temp_dbParams.multiLineFactorCf}")
        logging.debug(f"Set numPiPulses: {temp_dbParams.numPiPulses}")
        logging.debug(f"Set num2DCols: {temp_dbParams.num2DCols}")
        logging.debug(f"Set fastPiEnabled: {temp_dbParams.fastPiEnabled}")
        logging.debug(f"Set numZones2d: {temp_dbParams.numZones2d}")
        logging.debug(f"Set numSubVols: {temp_dbParams.numSubVols}")
        logging.debug(f"Set numPlanes: {temp_dbParams.numPlanes}")
        logging.debug(f"Set zigZagEnabled: {temp_dbParams.zigZagEnabled}")
        logging.debug(f"Set linesPerEnsCf: {temp_dbParams.linesPerEnsCf}")
        logging.debug(f"Set ensPerSeqCf: {temp_dbParams.ensPerSeqCf}")
        logging.debug(f"Set numCfCols: {temp_dbParams.numCfCols}")
        logging.debug(f"Set numCfEntries: {temp_dbParams.numCfEntries}")
        logging.debug(f"Set numCfDummies: {temp_dbParams.numCfDummies}")
        logging.debug("File header version 6 parsing complete")
    
    ###################################################################################
    # Helper Methods
    ###################################################################################
    def _read_int_array(self, file_obj, endianness: str, count: int) -> np.ndarray:
        """Helper method to read an array of integers as a numpy array for speed."""
        logging.debug(f"Reading {count} integers with endianness '{endianness}'")
        
        # Read the specified number of integers from the file object
        result = np.frombuffer(file_obj.read(4*count), dtype=np.dtype('<u4') if endianness=='little' else np.dtype('>u4'))
        logging.debug(f"Read {result.size} integers, first few: {result[:min(5, len(result))]}")
        
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
    # Raw Data Loading
    ###################################################################################
    def _load_raw_rf_data(self, filepath: str, total_header_size_bytes: int, read_offset_MB: int, read_size_MB: int) -> Tuple[Any, int]:
        """Load raw RF data from file, handling Voyager and Fusion formats."""
        logging.info(f"Loading raw RF data: is_voyager={self.is_voyager}, is_fusion={self.is_fusion}, offset={read_offset_MB}MB, size={read_size_MB}MB")
        
        # Calculate the parameters needed for reading the file
        _, remaining_size_bytes, read_offset_bytes, read_size_bytes = self._calculate_file_sizes(filepath,
                                                                                                total_header_size_bytes,
                                                                                                read_offset_MB,
                                                                                                read_size_MB
                                                                                                )
        
        # Load data based on the file type
        if self.is_voyager:
            return self._load_voyager_data(filepath, remaining_size_bytes, read_offset_bytes, read_size_bytes)
        elif self.is_fusion:
            return self._load_fusion_data(filepath, total_header_size_bytes, remaining_size_bytes, read_offset_bytes, read_size_bytes)
        else:
            # Raise an error if the file type is unknown
            raise RuntimeError("Unknown file type: neither Voyager nor Fusion detected.")
    
    ###################################################################################
    # File Size Calculation
    ###################################################################################
    def _calculate_file_sizes(self, filepath: str, total_header_size: int, read_offset_MB: int, read_size_MB: int) -> Tuple[int, int, int, int]:
        """Calculate file sizes and convert MB to bytes for read parameters."""
        
        # Get the total file size in bytes
        file_size_bytes = os.stat(filepath).st_size
        
        # Calculate the remaining size after the header
        remaining_size_bytes = file_size_bytes - total_header_size
        logging.debug(f"File size: {file_size_bytes} bytes, header size: {total_header_size} bytes, remaining: {remaining_size_bytes} bytes")
        
        # Convert read offset and size from MB to bytes
        read_offset_bytes = read_offset_MB * (2 ** 20)
        read_size_bytes = read_size_MB * (2 ** 20)
        logging.debug(f"Read parameters in bytes: offset={read_offset_bytes}, size={read_size_bytes}")
        
        return file_size_bytes, remaining_size_bytes, read_offset_bytes, read_size_bytes
    
    ###################################################################################
    # Load Voyager data
    ###################################################################################
    def _load_voyager_data(self, filepath: str, remaining_size_bytes: int, read_offset_bytes: int, read_size_bytes: int) -> Tuple[Any, int]:
        """Load data in Voyager format."""
        logging.info("Loading Voyager format data")
        
        # Align read parameters to Voyager data format
        read_offset_bytes, read_size_bytes = self._align_voyager_parameters(remaining_size_bytes, read_offset_bytes, read_size_bytes)
        
        # Read the raw data
        with open(filepath, 'rb') as f:
            f.seek(read_offset_bytes)
            rawrfdata = f.read(read_size_bytes)
        
        logging.info(f"Loaded {len(rawrfdata)} bytes of Voyager data")
        return rawrfdata, 0
    
    ###################################################################################
    # Alignment for Voyager
    ###################################################################################
    def _align_voyager_parameters(self, remaining_size_bytes: int, read_offset_bytes: int, read_size_bytes: int) -> Tuple[int, int]:
        """Align read parameters to Voyager format boundaries (36 bytes)."""
        alignment_bytes = np.arange(0, remaining_size_bytes + 1, 36)
        offset_diff_bytes = alignment_bytes - read_offset_bytes
        read_diff_bytes = alignment_bytes - read_size_bytes
        
        aligned_offset_bytes = alignment_bytes[np.where(offset_diff_bytes >= 0)[0][0]].__int__()
        aligned_size_bytes = alignment_bytes[np.where(read_diff_bytes >= 0)[0][0]].__int__()
        
        logging.debug(f"Aligned Voyager read - offset: {aligned_offset_bytes}, size: {aligned_size_bytes}")
        return aligned_offset_bytes, aligned_size_bytes

    ###################################################################################
    # Load Fusion data
    ###################################################################################
    def _load_fusion_data(self, filepath: str, total_header_size_bytes: int, remaining_size_bytes: int, read_offset_bytes: int, read_size_bytes: int) -> Tuple[Any, int]:
        """Load data in Fusion format."""
        logging.info("Loading Fusion format data")
        
        # Align read parameters to Fusion data format
        read_offset_bytes, read_size_bytes = self._align_fusion_parameters(remaining_size_bytes, read_offset_bytes, read_size_bytes)
        
        # Calculate number of clumps and final offset
        num_clumps = int(np.floor(read_size_bytes / 32))
        offset_bytes = total_header_size_bytes + read_offset_bytes
        logging.info(f"Reading Fusion data: {num_clumps} clumps from offset {offset_bytes}")
        
        # Read and process the data
        rawrfdata = self._read_and_process_fusion_data(filepath, offset_bytes, num_clumps, use_c=self.use_c)
        
        logging.info(f"Loaded Fusion data with shape {rawrfdata.shape}")
        return rawrfdata, num_clumps
    
    ###################################################################################
    # Alignment for Fusion
    ###################################################################################
    def _align_fusion_parameters(self, remaining_size_bytes: int, read_offset_bytes: int, read_size_bytes: int) -> Tuple[int, int]:
        """Align read parameters to Fusion format boundaries (32 bytes)."""
        
        # Calculate alignment bytes based on 32-byte boundaries
        alignment_bytes = np.arange(0, remaining_size_bytes + 1, 32)
        offset_diff_bytes = alignment_bytes - read_offset_bytes
        read_diff_bytes = alignment_bytes - read_size_bytes
        
        # Find matching offset
        matching_indices = np.where(offset_diff_bytes >= 0)[0]
        if len(matching_indices) > 0:
            aligned_offset_bytes = alignment_bytes[matching_indices[0]].__int__()
        else:
            aligned_offset_bytes = 0
            logging.warning("No matching offset found, using 0")
        
        # Find matching size
        matching_indices = np.where(read_diff_bytes >= 0)[0]
        if len(matching_indices) > 0:
            aligned_size_bytes = alignment_bytes[matching_indices[0]].__int__()
        else:
            aligned_size_bytes = remaining_size_bytes
            logging.warning(f"No matching size found, using remaining size: {aligned_size_bytes}")
        
        logging.debug(f"Aligned Fusion read - offset: {aligned_offset_bytes}, size: {aligned_size_bytes}")
        return aligned_offset_bytes, aligned_size_bytes
    
    ###################################################################################
    # Read and process Fusion format data
    ###################################################################################
    def _read_and_process_fusion_data(self, filepath: str, offset_bytes: int, num_clumps: int, use_c: bool) -> np.ndarray:
        """Read and process Fusion format data.
        
        Args:
            filepath: Path to the RF data file
            offset_bytes: Offset in bytes where to start reading
            num_clumps: Number of data clumps to read
            use_c: If True, use C implementation instead of Python
        """
        logging.debug(f"Starting to process {num_clumps} clumps...")
        logging.debug(f"Using C implementation: {use_c}")
        logging.debug(f"Offset bytes: {offset_bytes}")
        logging.debug(f"Filepath: {filepath}")
        
        if use_c:
            # Use C implementation from philipsRfParser module
            part_a = getPartA(num_clumps, filepath, offset_bytes)
            part_b = getPartB(num_clumps, filepath, offset_bytes)
        else:
            # Use Python implementation
            part_a = self._get_part_a_py(num_clumps, filepath, offset_bytes)
            part_b = self._get_part_b_py(num_clumps, filepath, offset_bytes)
            
        logging.debug(f"Retrieved partA: {len(part_a)} elements, partB: {len(part_b)} elements")
        
        # Process and reshape the data
        rawrfdata = np.concatenate((
            np.array(part_a, dtype=int).reshape((12, num_clumps), order='F'),
            np.array([part_b], dtype=int)
        ))
        logging.debug(f"Raw RF data shape: {rawrfdata.shape}")
        
        return rawrfdata

    ###################################################################################
    # Python implementations of C functions - getPartA
    ###################################################################################
    def _get_part_a_py(self, num_clumps: int, filepath: str, offset_bytes: int) -> list:
        """Python implementation of getPartA from C code.
        Follows exact same implementation as philips_rf_parser.c get_partA function.
        """
        logging.debug(f"[_get_part_a_py] Starting to process {num_clumps} clumps...")
        
        part_a = [0] * (12 * num_clumps)  # Pre-allocate array like C
        bytes_read = bytearray(256)  # Match C allocation
        
        with open(filepath, 'rb') as fd:
            fd.seek(offset_bytes)
            
            i = 0  # Byte position in current chunk
            x = 0  # Position in output array
            j = 0  # Chunk counter
            bits_left = 0
            bit_offset = 4  # Initial bit offset
            last_percentage = -1
            
            while j < num_clumps:
                if not j or i == 31:  # Start of new chunk
                    assert bit_offset == 4
                    bit_offset = 8
                    chunk = fd.read(32)
                    if not chunk:
                        break
                    bytes_read[:len(chunk)] = chunk
                    j += 1
                    i = 0
                    
                    # Log progress percentage
                    current_percentage = (j * 100) // num_clumps
                    if current_percentage > last_percentage:
                        logging.debug(f"[_get_part_a_py] Progress: {current_percentage}%")
                        last_percentage = current_percentage
                else:
                    # Exactly match C bit manipulation
                    mask = (~0) << (8 - bit_offset)
                    first = (bytes_read[i] & mask) >> (8 - bit_offset)
                    first |= (bytes_read[i + 1] << bit_offset)
                    
                    second = bytes_read[i + 1] >> (8 - bit_offset)
                    second |= (bytes_read[i + 2] << bit_offset)
                    
                    third = bytes_read[i + 2] >> (8 - bit_offset)
                    
                    bits_left = 5 - bit_offset
                    if bits_left > 0:
                        i += 1
                        mask = ~((~0) << bits_left)
                        temp = mask & bytes_read[i + 2]
                        third |= temp << bit_offset
                        bit_offset = 8 - bits_left
                    elif bits_left < 0:
                        mask = ~((~0) << 5)
                        third &= mask
                        bit_offset = -bits_left
                    else:
                        i += 1
                        bit_offset = 8
                    
                    # Create a bytes object to match C's memory layout
                    value_bytes = bytes([first & 0xFF, second & 0xFF, third & 0xFF, 0])
                    part_a[x] = int.from_bytes(value_bytes, byteorder='little')
                    x += 1
                    i += 2
        
        logging.debug("[_get_part_a_py] Processing completed (100%)")
        return part_a

    ###################################################################################
    # Python implementations of C functions - getPartB
    ###################################################################################
    def _get_part_b_py(self, num_clumps: int, filepath: str, offset_bytes: int) -> list:
        """Python implementation of getPartB from C code.
        Follows exact same implementation as philips_rf_parser.c get_partB function.
        """
        logging.debug(f"[_get_part_b_py] Starting to process {num_clumps} clumps...")
        
        part_b = [0] * num_clumps  # Pre-allocate array like C
        bytes_read = bytearray(256)  # Match C allocation
        mask = ~((~0) << 4)  # 4-bit mask
        
        with open(filepath, 'rb') as fd:
            fd.seek(offset_bytes)
            
            x = 0
            j = 0
            last_percentage = -1
            
            while j < num_clumps:
                chunk = fd.read(32)
                if not chunk:
                    break
                bytes_read[:len(chunk)] = chunk
                
                # Match C implementation exactly
                cur_num = bytes_read[0]
                cur_num &= mask
                part_b[x] = int(cur_num)  # Cast to int like C does
                x += 1
                j += 1
                
                # Log progress percentage every 5%
                current_percentage = (j * 100) // num_clumps
                if current_percentage > last_percentage and current_percentage % 5 == 0:
                    logging.debug(f"[_get_part_b_py] Progress: {current_percentage}%")
                    last_percentage = current_percentage
        
        logging.debug("[_get_part_b_py] Processing completed (100%)")
        return part_b

    ###################################################################################
    # Reshape Voyager raw data
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
    # Parse header dispatch
    ###################################################################################
    def _parse_header_dispatch(self, rawrfdata: Any) -> 'PhilipsRfParser.HeaderInfoStruct':
        """Dispatch to the correct header parsing method."""
        logging.info(f"Dispatching header parsing: is_voyager={self.is_voyager}, is_fusion={self.is_fusion}")
        if self.is_voyager:
            logging.debug(f"Using Voyager header parser")
            return self._parse_header_voyager(rawrfdata)
        elif self.is_fusion:
            logging.debug(f"Using Fusion header parser")
            return self._parse_header_fusion(rawrfdata)
        else:
            raise RuntimeError("Unknown file type: neither Voyager nor Fusion detected.")

    ###################################################################################
    # Voyager Header Parsing
    ###################################################################################
    def _parse_header_voyager(self, rawrfdata):
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
    # Find headers in Voyager data
    ###################################################################################
    def _find_voyager_headers(self, rawrfdata):
        """Find headers in Voyager data."""
        iHeader = np.where(np.uint8(rawrfdata[2,0,:])&224)
        numHeaders = len(iHeader)-1  # Ignore last header as it is part of a partial line
        logging.debug(f"Found {numHeaders} headers in Voyager data")
        return iHeader, numHeaders

    ###################################################################################
    # Initialize header arrays
    ###################################################################################
    def _initialize_header_arrays(self, header_info, numHeaders):
        """Initialize header arrays with appropriate sizes and types."""
        logging.debug(f"Initializing header arrays for {numHeaders} headers")
        
        # Initialize 8-bit fields
        header_info.RF_CaptureVersion = np.zeros(numHeaders, dtype=np.uint8)
        header_info.Tap_Point = np.zeros(numHeaders, dtype=np.uint8)
        header_info.Data_Gate = np.zeros(numHeaders, dtype=np.uint8)
        header_info.Multilines_Capture = np.zeros(numHeaders, dtype=np.uint8)
        header_info.RF_Sample_Rate = np.zeros(numHeaders, dtype=np.uint8)
        header_info.Steer = np.zeros(numHeaders, dtype=np.uint8)
        header_info.elevationPlaneOffset = np.zeros(numHeaders, dtype=np.uint8)
        header_info.PM_Index = np.zeros(numHeaders, dtype=np.uint8)
        logging.debug("Initialized 8-bit fields")
        
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
        logging.debug("Initialized 16-bit fields")
        
        # Initialize 32-bit fields
        header_info.Frame_ID = np.zeros(numHeaders, dtype=np.uint32)
        header_info.Time_Stamp = np.zeros(numHeaders, dtype=np.uint32)
        logging.debug("Initialized 32-bit fields")
        
        logging.info("Header arrays initialization complete")
        return header_info

    ###################################################################################
    # Process each header in Voyager data
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
    # Build packed header string from raw data
    ###################################################################################
    def _build_voyager_packed_header(self, rawrfdata, iHeader, m):
        """Build packed header string from raw data."""
        logging.debug(f"Building packed header for header index {m}")
        
        packedHeader = ''
        for k in np.arange(11, 0, -1):
            temp = ''
            for i in np.arange(2, 0, -1):
                value = np.uint8(rawrfdata[i, k, iHeader[m]])
                temp += bin(value)
                
            # Discard first 3 bits, redundant info
            packedHeader += temp[3:24]
            logging.debug(f"Intermediate packed header: {packedHeader}")
        logging.debug(f"Final packed header: {packedHeader}")
        
        return packedHeader

    ###################################################################################
    # Parse values from packed header string
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
    def _parse_header_fusion(self, rawrfdata):
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
    # Find headers in Fusion data
    ###################################################################################
    def _find_fusion_headers(self, rawrfdata):
        """Find and extract header locations in Fusion format RF data.

        This method identifies the positions of headers in Fusion format data by performing
        a bitwise operation on the first row of the raw RF data. In Fusion format, headers
        are identified by a specific bit pattern where (value & 1572864 == 524288).
        Each header occupies exactly one "Clump" in the data.

        Args:
            rawrfdata (numpy.ndarray): Raw RF data array. Expected to be a 2D array where:
                - First dimension represents the rows (13 rows per clump)
                - Second dimension represents the columns (number of clumps)

        Returns:
            tuple: A tuple containing:
                - iHeader (numpy.ndarray): Array of indices where headers are found
                - numHeaders (int): Number of valid headers (excluding the last partial header)

        Note:
            - The method uses the bit pattern 1572864 (0x180000) as a mask and looks for
              values that equal 524288 (0x80000) after masking
        """
        logging.info('Entering _find_fusion_headers')
        logging.info(f'Raw RF data shape: {rawrfdata.shape}')

        # Get first row of data and perform bitwise operation
        first_row = rawrfdata[0, :]
        logging.info(f'First row shape: {first_row.shape}')
        
        # Apply bit mask to find headers
        mask = 1572864  # 0x180000
        target = 524288  # 0x80000
        masked_values = first_row & mask
        logging.info(f'Number of values after masking: {len(masked_values)}')
        
        # Find indices where the masked value equals the target
        iHeader = np.where(masked_values == target)[0]
        logging.info(f'Found {len(iHeader)} potential header locations at indices: {iHeader[:min(10, len(iHeader))]}... (showing first 10 of {len(iHeader)})')

        # Calculate number of headers (excluding last partial header)
        if len(iHeader) > 1:
            # Check spacing between headers
            header_spacing = np.diff(iHeader)
            logging.info(f'Spacing between headers: {header_spacing[:5]}...{header_spacing[-5:]} (showing first 5 and last 5 of {len(header_spacing)})')
            
            # Get number of complete headers
            numHeaders = len(iHeader) - 1
            logging.info(f'Number of complete headers (excluding last partial): {numHeaders}')
        else:
            numHeaders = 0
            logging.info('No complete headers found')

        logging.info('Exiting _find_fusion_headers')
        return iHeader, numHeaders

    ###################################################################################
    # Process each header in Fusion data
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
    # Build packed header string from raw data for Fusion systems
    ###################################################################################
    def _build_fusion_packed_header(self, rawrfdata, iHeader, m):
        """Build a binary string representation of a Fusion format header from raw data."""
        # Get the data from the 13th element (index 12)
        packedHeader = bin(rawrfdata[12, iHeader[m]])[2:]
        
        # Add leading zeros if needed
        remainingZeros = 4 - len(packedHeader)
        if remainingZeros > 0:
            zeros = self._get_filler_zeros(remainingZeros)
            packedHeader = str(zeros + packedHeader)
        
        # Add data from remaining elements in forward order to match little-endian
        for i in range(12):
            curBin = bin(int(rawrfdata[i, iHeader[m]]))[2:]
            remainingZeros = 21 - len(curBin)
            if remainingZeros > 0:
                zeros = self._get_filler_zeros(remainingZeros)
                curBin = str(zeros + curBin)
            packedHeader += curBin
        
        return packedHeader
    
    ###################################################################################
    # Get filler zeros
    ###################################################################################
    @staticmethod
    def _get_filler_zeros(num: int) -> str:
        """Get string of zeros for padding."""
        #logging.debug(f"Creating filler zeros, num={num}")
        
        # Ensure we don't create negative length strings
        count = max(0, num - 1)
        result = '0' * count
        
        #logging.debug(f"Generated {len(result)} filler zeros")
        return result

    ###################################################################################
    # Parse values from packed header string for Fusion systems
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
        time_stamp_bits = packedHeader[iBit:iBit+13] + packedHeader[iBit+15:iBit+34]
        HeaderInfo.Time_Stamp[m] = int(time_stamp_bits, 2)
        
        # Log all header fields for first header and every 1000th header
        if m == 0 or m % 1000 == 0:
            logging.info(f"\nComplete Header {m} Data:")
            logging.info("8-bit fields:")
            logging.info(f"  RF_CaptureVersion: {HeaderInfo.RF_CaptureVersion[m]}")
            logging.info(f"  Tap_Point: {HeaderInfo.Tap_Point[m]}")
            logging.info(f"  Data_Gate: {HeaderInfo.Data_Gate[m]}")
            logging.info(f"  Multilines_Capture: {HeaderInfo.Multilines_Capture[m]}")
            logging.info(f"  RF_Sample_Rate: {HeaderInfo.RF_Sample_Rate[m]}")
            logging.info(f"  Steer: {HeaderInfo.Steer[m]}")
            logging.info(f"  elevationPlaneOffset: {HeaderInfo.elevationPlaneOffset[m]}")
            logging.info(f"  PM_Index: {HeaderInfo.PM_Index[m]}")
            
            logging.info("\n16-bit fields:")
            logging.info(f"  Line_Index: {HeaderInfo.Line_Index[m]}")
            logging.info(f"  Pulse_Index: {HeaderInfo.Pulse_Index[m]}")
            logging.info(f"  Data_Format: {HeaderInfo.Data_Format[m]}")
            logging.info(f"  Data_Type: {HeaderInfo.Data_Type[m]}")
            logging.info(f"  Header_Tag: {HeaderInfo.Header_Tag[m]}")
            logging.info(f"  Threed_Pos: {HeaderInfo.Threed_Pos[m]}")
            logging.info(f"  Mode_Info: {HeaderInfo.Mode_Info[m]}")
            
            logging.info("\n32-bit and special fields:")
            logging.info(f"  Frame_ID: {HeaderInfo.Frame_ID[m]}")
            logging.info(f"  CSID: {HeaderInfo.CSID[m]}")
            logging.info(f"  Line_Type: {HeaderInfo.Line_Type[m]}")
            logging.info(f"  Time_Stamp: {HeaderInfo.Time_Stamp[m]}")
            
            # Add hex representation for relevant fields
            logging.info("\nHex representations:")
            logging.info(f"  Data_Type: 0x{HeaderInfo.Data_Type[m]:04x}")
            logging.info(f"  Frame_ID: 0x{HeaderInfo.Frame_ID[m]:08x}")
            logging.info(f"  Time_Stamp: 0x{HeaderInfo.Time_Stamp[m]:08x}")
            logging.info("-" * 50)

    ###################################################################################
    # Dispatch to the correct RF data parsing method
    ###################################################################################
    def _parse_rf_data_dispatch(self, rawrfdata: Any, header_info: 'PhilipsRfParser.HeaderInfoStruct') -> Tuple[np.ndarray, np.ndarray, int]:
        """Dispatch to the correct RF data parsing method."""
        logging.info(f"Dispatching RF data parsing: is_voyager={self.is_voyager}, is_fusion={self.is_fusion}")
        return self._parse_rf_data(rawrfdata, header_info)

    ###################################################################################
    # Parse RF signal data
    ###################################################################################
    def _parse_rf_data(self, rawrfdata, headerInfo: 'PhilipsRfParser.HeaderInfoStruct') -> Tuple[np.ndarray, np.ndarray, int]:
        """Parse RF signal data."""
        logging.info("Parsing RF signal data...")
        Tap_Point = headerInfo.Tap_Point[0]
        logging.debug(f"Tap Point: {Tap_Point}, isVoyager: {self.is_voyager}, isFusion: {self.is_fusion}")
        
        # Determine the parsing method based on the file type
        if self.is_voyager:
            # Parse data for Voyager systems
            lineData, lineHeader = self._parse_data_voyager(rawrfdata, headerInfo)
        elif self.is_fusion:
            # Parse data for Fusion systems
            lineData, lineHeader = self._parse_data_fusion(rawrfdata, headerInfo)
            Tap_Point = headerInfo.Tap_Point[0]
            if Tap_Point == 0: # Correct for MS 19 bits of 21 real data bits
                logging.debug("Applying bit shift correction for Tap Point 0")
                lineData = lineData << 2
        else:
            # Raise an error if the file type is unknown
            raise RuntimeError("Unknown file type: neither Voyager nor Fusion detected.")
        
        # After parsing lineData, log a sample of the first and last 20 rows for a nonzero column
        nonzero_col = None
        for col in range(lineData.shape[1]):
            if np.any(lineData[:, col] != 0):
                nonzero_col = col
                break
        
        # Log the first and last 20 values of a nonzero column, if found
        if nonzero_col is not None:
            logging.info(f"First 20 values of lineData[:, {nonzero_col}]: {lineData[:20, nonzero_col]}")
            logging.info(f"Last 20 values of lineData[:, {nonzero_col}]: {lineData[-20:, nonzero_col]}")
            logging.info(f"Min: {lineData[:, nonzero_col].min()}, Max: {lineData[:, nonzero_col].max()}")
        else:
            logging.warning("No nonzero columns found in lineData!")
        
        logging.info(f"RF data parsing complete - lineData: {lineData.shape}, lineHeader: {lineHeader.shape}")
        return lineData, lineHeader, Tap_Point

    ###################################################################################
    # Parse Voyager data
    ###################################################################################
    def _parse_data_voyager(self, rawrfdata, headerInfo: 'PhilipsRfParser.HeaderInfoStruct') -> Tuple[np.ndarray, np.ndarray]:
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
    # Parse Fusion data
    ###################################################################################
    def _parse_data_fusion(self, rawrfdata, headerInfo: 'PhilipsRfParser.HeaderInfoStruct') -> Tuple[np.ndarray, np.ndarray]:
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
    # Get multiline capture settings and adjust tap point if needed
    ###################################################################################
    def _get_echo_capture_settings(self, header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> Tuple[float, int]:
        """Get multiline capture settings and adjust tap point if needed."""
        logging.debug(f"Echo data types: {self.DataType_ECHO}")
        
        # Determine ML_Capture based on tap point and multilines capture setting
        ML_Capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        logging.debug(f"ML_Capture: {ML_Capture}")
        
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
    # Find echo data indices
    ###################################################################################
    def _find_echo_data_indices(self, header_info: 'PhilipsRfParser.HeaderInfoStruct') -> Tuple[np.ndarray, int]:
        """Find indices of echo data in the dataset."""
        xmit_events = len(header_info.Data_Type)
        
        # log header_info.Data_Type
        logging.debug(f"header_info.Data_Type first 10 elements: {header_info.Data_Type[:10]}")
        logging.debug(f"header_info.Data_Type last 10 elements: {header_info.Data_Type[-10:]}")
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
    # Process echo data
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
    # Get echo data for the specified tap point
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
    # Sort echo data based on tap point and assign to rfdata
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
    # Continuous Wave Data Extraction
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
    # Prune data
    ###################################################################################
    def _prune_data(self, line_data, line_header, multiline_capture):
        """Remove false gate data at beginning of the line."""
        logging.info(f"Pruning data - input shape: line_data={line_data.shape}, line_header={line_header.shape}, multiline_capture={multiline_capture}")
        
        # Determine the number of samples and reference line for pruning
        num_samples = line_data.shape[0]
        reference_line = int(np.ceil(line_data.shape[1]*0.2))-1    
        
        # Calculate the starting point for pruning
        start_point = int(np.ceil(num_samples*0.015))-1
        logging.debug(f"Looking for start point from sample {start_point} in reference line {reference_line}")
        
        # Find indices where the line header matches the condition for start point
        indices_found = np.where(line_header[start_point:num_samples+1, reference_line]==3)
        if not len(indices_found[0]):
            first_sample = 1
            logging.debug("No valid start point found, using sample 1")
        else:
            first_sample = indices_found[0][0]+start_point
            logging.debug(f"Found start point at sample {first_sample}")
        
        # Align the start point to the nearest valid position based on multiline capture
        alignment = np.arange(0,num_samples, np.double(multiline_capture))
        diff = alignment - first_sample
        first_sample = int(alignment[np.where(diff>=0)[0][0]])
        logging.debug(f"Aligned start point to {first_sample}")
        
        # Prune the data from the calculated start point
        pruned_data = line_data[first_sample:num_samples+1,:]
        line_header = line_header[first_sample:num_samples+1,:]
        logging.debug(f"Pruned from start: new shape {pruned_data.shape}")
        
        # Recalculate the number of samples after initial pruning
        num_samples = pruned_data.shape[0]
        
        # Calculate the end point for pruning
        start_point = int(np.floor(num_samples*0.99))-1
        logging.debug(f"Looking for end point from sample {start_point}")
        
        # Find indices where the line header matches the condition for end point
        indices_found = np.where(line_header[start_point:num_samples+1,reference_line]==0)
        if not len(indices_found[0]):
            last_sample = num_samples
            logging.debug("No valid end point found, using last sample")
        else:
            last_sample = indices_found[0][0]+start_point
            alignment = np.arange(0,num_samples, np.double(multiline_capture))
            diff = alignment - last_sample
            last_sample = int(alignment[np.where(diff >= 0)[0][0]])-1
            logging.debug(f"Found and aligned end point to {last_sample}")
        
        # Prune the data to the calculated end point
        pruned_data = pruned_data[:last_sample+1, :]
        logging.info(f"Pruning complete - final shape: {pruned_data.shape}")
        return pruned_data

    ###################################################################################
    # Sort RF data based on multiline parameters
    ###################################################################################
    def _sort_rf(self, RFinput, Stride, ML, CRE=1, isVoyager=True):
        """Sort RF data based on multiline parameters.
        
        Args:
            RFinput: Input RF data array
            Stride: Stride value for the data
            ML: Multiline factor
            CRE: Cross-Resolution-Enhancement factor (default: 1)
            isVoyager: Whether this is Voyager format data (default: True)
        """
        logging.info(f"Sorting RF data - input shape: {RFinput.shape}, Stride={Stride}, ML={ML}, CRE={CRE}, isVoyager={isVoyager}")
        
        # Initialize dimensions and output arrays
        N, xmit_events, depth, multilines = self._initialize_rf_sort_dimensions(RFinput, Stride, ML)
        
        # Initialize output arrays based on CRE
        out0, out1, out2, out3 = self._initialize_rf_sort_outputs(depth, ML, xmit_events, CRE)
        
        # Get the ML sort list for specified Stride and CRE
        multiline_sort_list = self._get_ml_sort_list(Stride, CRE)
        
        # Check for potential issues
        self._check_ml_sort_validity(ML, multiline_sort_list, CRE, Stride)
        
        # Fill output arrays based on the sort list
        out0, out1, out2, out3 = self._fill_rf_sort_outputs(
            RFinput, out0, out1, out2, out3, 
            multilines, multiline_sort_list, depth, Stride, ML, CRE
        )
        
        logging.info(f"RF sorting complete - output shape: {out0.shape}")
        return out0, out1, out2, out3

    ###################################################################################
    # Initialize dimensions for RF sorting
    ###################################################################################
    def _initialize_rf_sort_dimensions(self, RFinput, Stride, ML):
        """Initialize dimensions for RF sorting."""
        logging.debug(f"Initializing dimensions - input shape: {RFinput.shape}, Stride: {Stride}, ML: {ML}")
        
        # Calculate dimensions
        N = RFinput.shape[0]
        xmit_events = RFinput.shape[1]
        depth = int(np.floor(N/Stride))
        
        # Create array of multiline indices
        multilines = np.arange(0, ML)
        
        logging.debug(f"Calculated dimensions - N: {N}, xmit_events: {xmit_events}, depth: {depth}, multilines range: 0-{ML-1}")
        return N, xmit_events, depth, multilines

    ###################################################################################
    # Initialize output arrays based on cross-resolution-enhancement value
    ###################################################################################
    def _initialize_rf_sort_outputs(self, depth, ML, xmit_events, CRE):
        """Initialize output arrays based on CRE value.
        
        Args:
            depth: Depth of the output arrays
            ML: Multiline factor
            xmit_events: Number of transmit events
            CRE: Cross-Resolution-Enhancement factor
            
        Returns:
            Tuple of initialized output arrays (out0, out1, out2, out3) based on CRE value
        """
        logging.debug(f"Initializing output arrays - depth: {depth}, ML: {ML}, xmit_events: {xmit_events}, CRE: {CRE}")
        
        # Initialize arrays with empty values
        out0 = out1 = out2 = out3 = np.array([])
        
        # Initialize array shape for logging
        array_shape = (depth, ML, xmit_events)
        array_size_mb = (depth * ML * xmit_events * 4) / (1024 * 1024)  # Assuming 4 bytes per element
        
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
    # Get the appropriate ML sort list based on Stride and CRE
    ###################################################################################
    def _get_ml_sort_list(self, Stride, CRE):
        """Get the appropriate ML sort list based on Stride and CRE."""
        logging.debug(f"Getting ML sort list for Stride={Stride}, CRE={CRE}")
        
        # Initialize empty list
        multiline_sort_list = []
        
        # Select sort list based on Stride and CRE values
        if Stride == 128:
            multiline_sort_list = self.ML_SortList_128
        elif Stride == 32:
            if CRE == 4:
                multiline_sort_list = self.ML_SortList_32_CRE4
            else:
                multiline_sort_list = self.ML_SortList_32
        elif Stride == 16:
            if CRE == 1:
                multiline_sort_list = self.ML_SortList_16_CRE1
            elif CRE == 2:
                multiline_sort_list = self.ML_SortList_16_CRE2
            elif CRE == 4:
                multiline_sort_list = self.ML_SortList_16_CRE4
        elif Stride == 12:
            if CRE == 1:
                multiline_sort_list = self.ML_SortList_12_CRE1
            elif CRE == 2:
                multiline_sort_list = self.ML_SortList_12_CRE2
            elif CRE == 4:
                multiline_sort_list = self.ML_SortList_12_CRE4
        elif Stride == 8:
            if CRE == 1:
                multiline_sort_list = self.ML_SortList_8_CRE1
            elif CRE == 2:
                multiline_sort_list = self.ML_SortList_8_CRE2
            elif CRE == 4:
                multiline_sort_list = self.ML_SortList_8_CRE4
        elif Stride == 4:
            if CRE == 1:
                multiline_sort_list = self.ML_SortList_4_CRE1
            elif CRE == 2:
                multiline_sort_list = self.ML_SortList_4_CRE2
            elif CRE == 4:
                multiline_sort_list = self.ML_SortList_4_CRE4
        elif Stride == 2:
            if CRE == 1:
                multiline_sort_list = self.ML_SortList_2_CRE1
            elif CRE == 2:
                multiline_sort_list = self.ML_SortList_2_CRE2
            elif CRE == 4:
                multiline_sort_list = self.ML_SortList_2_CRE4
        else:
            logging.warning(f"No sort list for Stride={Stride}")
        
        logging.debug(f"Using multiline_sort_list with {len(multiline_sort_list)} elements")
        return multiline_sort_list

    ###################################################################################
    # Check if the ML sort list is valid for the requested parameters
    ###################################################################################
    def _check_ml_sort_validity(self, ML, multiline_sort_list, CRE, Stride):
        """Check if the ML sort list is valid for the requested parameters."""
        logging.debug(f"Checking ML sort list validity - ML: {ML}, CRE: {CRE}, Stride: {Stride}")
        
        # Check if sort list is empty
        if not multiline_sort_list:
            logging.warning(f"Empty multiline_sort_list for Stride={Stride}, CRE={CRE}")
            return
        
        # Log sort list properties
        logging.debug(f"multiline_sort_list - length: {len(multiline_sort_list)}, min: {min(multiline_sort_list)}, max: {max(multiline_sort_list)}")
        
        # Check if ML value exceeds what's available in the sort list
        if ((ML-1) > max(multiline_sort_list)):
            logging.warning(f"ML ({ML}) exceeds max value in multiline_sort_list ({max(multiline_sort_list)})")
        
        # Check for special configuration issues
        if (CRE == 4 and Stride < 16):
            logging.warning(f"Insufficient ML capture for CRE=4 with Stride={Stride} (should be >= 16)")
            
        if (CRE == 2 and Stride < 4):
            logging.warning(f"Insufficient ML capture for CRE=2 with Stride={Stride} (should be >= 4)")
        
        logging.debug(f"ML sort list validity check complete")

    ###################################################################################
    # Fill output arrays based on the sort list
    ###################################################################################
    def _fill_rf_sort_outputs(self, RFinput, out0, out1, out2, out3, multilines, multiline_sort_list, depth, Stride, ML, CRE):
        """Fill output arrays based on the sort list."""
        logging.info(f"Filling output arrays - ML: {ML}, CRE: {CRE}, depth: {depth}")
        
        # Skip if sort list is empty
        if not multiline_sort_list:
            logging.warning(f"Empty multiline_sort_list, unable to fill output arrays")
            return out0, out1, out2, out3
        
        # Log the first few items in the sort list
        preview_length = min(10, len(multiline_sort_list))
        logging.debug(f"Using multiline_sort_list (first {preview_length}): {multiline_sort_list[:preview_length]}")
        
        # Store matches for logging
        matches_found = 0
        ml_not_found = []
        
        # Process each multiline index
        for k in range(ML):
            logging.debug(f"Processing ML index {k} of {ML}")
            
            # Get indices in sort list that match current multiline index
            iML = np.where(np.array(multiline_sort_list) == multilines[k])[0]
            
            # Skip if no matching indices found
            if len(iML) == 0:
                logging.warning(f"No matching indices for ML={multilines[k]} in sort list")
                ml_not_found.append(multilines[k])
                continue
            
            matches_found += 1
            logging.debug(f"Found {len(iML)} matches for ML={multilines[k]} at indices {iML}")
            
            # Fill primary output array
            self._fill_output_array(out0, RFinput, depth, k, iML[0], Stride)
            
            # Fill additional output arrays based on CRE
            if CRE >= 2 and len(iML) > 1:
                logging.debug(f"Filling CRE={CRE} outputs for ML={multilines[k]}")
                self._fill_output_array(out1, RFinput, depth, k, iML[1], Stride)
                
                # These are duplicated for backward compatibility
                if out2.size > 0:
                    self._fill_output_array(out2, RFinput, depth, k, iML[1], Stride)
                if out3.size > 0:
                    self._fill_output_array(out3, RFinput, depth, k, iML[1], Stride)
            
            # Fill tertiary and quaternary output arrays for CRE=4
            if CRE == 4 and len(iML) > 3:
                logging.debug(f"Filling CRE=4 tertiary and quaternary outputs for ML={multilines[k]}")
                self._fill_output_array(out2, RFinput, depth, k, iML[2], Stride)
                self._fill_output_array(out3, RFinput, depth, k, iML[3], Stride)
        
        # Log summary statistics
        logging.info(f"Output array filling complete - {matches_found}/{ML} multilines processed")
        if ml_not_found:
            logging.warning(f"Missing multilines: {ml_not_found}")
        
        return out0, out1, out2, out3

    ###################################################################################
    # Fill a specific output array with data from the input array
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
    # Extract pulse wave data from RF data
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
    # Extract color data from RF data
    ###################################################################################
    def _extract_color_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract color data from RF data."""
        logging.info(f"Extracting color data, tap_point={tap_point}")
        logging.debug(f"Color data types: {self.DataType_COLOR}")
        
        multiline_capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if multiline_capture == 0:
            sample_rate = float(header_info.RF_Sample_Rate[0])
            multiline_capture = 16 if sample_rate == 0 else 32
            logging.debug(f"multiline_capture was 0, set to {multiline_capture} based on sample rate {sample_rate}")
            
        logging.debug(f"multiline_capture={multiline_capture}")
        
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
            color_data = self._prune_data(rfdata.lineData[:, color_index], rfdata.lineHeader[:, color_index], multiline_capture)
            if tap_point in [0, 1]:
                multiline_actual = rfdata.dbParams.azimuthMultilineFactorXbrInCf * rfdata.dbParams.elevationMultilineFactorCf
            else:
                multiline_actual = rfdata.dbParams.azimuthMultilineFactorXbrOutCf * rfdata.dbParams.elevationMultilineFactorCf
            cross_resolution_enhancement = 1
            rfdata.colorData = self._sort_rf(color_data, multiline_capture, multiline_actual, cross_resolution_enhancement, False)
        return rfdata

    ###################################################################################
    # Extract echo M-Mode data from RF data
    ###################################################################################
    def _extract_echo_mmode_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract echo M-Mode data from RF data."""
        logging.info(f"Extracting echo M-Mode data, tap_point={tap_point}")
        logging.debug(f"Echo M-Mode data type: {self.DataType_EchoMMode}")
        
        multiline_capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if multiline_capture == 0:
            sample_rate = float(header_info.RF_Sample_Rate[0])
            multiline_capture = 16 if sample_rate == 0 else 32
            logging.debug(f"multiline_capture was 0, set to {multiline_capture} based on sample rate {sample_rate}")
            
        logging.debug(f"multiline_capture={multiline_capture}")
        
        echo_mmode_index = (header_info.Data_Type == self.DataType_EchoMMode)
        echo_mmode_count = np.sum(echo_mmode_index)
        logging.debug(f"Found {echo_mmode_count} echo M-Mode data entries")
        
        if echo_mmode_count > 0:
            logging.debug("Processing echo M-Mode data")
            echo_mmode_data = self._prune_data(rfdata.lineData[:, echo_mmode_index], rfdata.lineHeader[:, echo_mmode_index], multiline_capture)
            
            multiline_actual = 1
            cross_resolution_enhancement = 1
            logging.debug(f"Echo M-Mode sorting: multiline_actual={multiline_actual}, cross_resolution_enhancement={cross_resolution_enhancement}")
            
            rfdata.echoMModeData = self._sort_rf(echo_mmode_data, multiline_capture, multiline_actual, cross_resolution_enhancement, False)
            logging.info(f"Echo M-Mode data extracted successfully, shape: {rfdata.echoMModeData[0].shape if hasattr(rfdata.echoMModeData, '__getitem__') else 'N/A'}")
        else:
            logging.debug("No echo M-Mode data found")
            
        return rfdata

    ###################################################################################
    # Extract color M-Mode data from RF data
    ###################################################################################
    def _extract_color_mmode_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract color M-Mode data from RF data."""
        logging.info(f"Extracting color M-Mode data, tap_point={tap_point}")
        logging.debug(f"Color M-Mode data types: {self.DataType_ColorMMode}")
        
        multiline_capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if multiline_capture == 0:
            sample_rate = float(header_info.RF_Sample_Rate[0])
            multiline_capture = 16 if sample_rate == 0 else 32
            logging.debug(f"multiline_capture was 0, set to {multiline_capture} based on sample rate {sample_rate}")
            
        logging.debug(f"multiline_capture={multiline_capture}")
        
        xmit_events = len(header_info.Data_Type)
        color_mmode_index = np.zeros(xmit_events).astype(bool)
        
        for dt in self.DataType_ColorMMode:
            index = (header_info.Data_Type == dt)
            color_mmode_index = np.bitwise_or(color_mmode_index, index)
            
        color_mmode_count = np.sum(color_mmode_index)
        logging.debug(f"Found {color_mmode_count} color M-Mode data entries")
        
        if color_mmode_count > 0:
            logging.debug("Processing color M-Mode data")
            color_mmode_data = self._prune_data(rfdata.lineData[:, color_mmode_index], rfdata.lineHeader[:, color_mmode_index], multiline_capture)
            
            multiline_actual = 1
            cross_resolution_enhancement = 1
            logging.debug(f"Color M-Mode sorting: multiline_actual={multiline_actual}, cross_resolution_enhancement={cross_resolution_enhancement}")
            
            rfdata.colorMModeData = self._sort_rf(color_mmode_data, multiline_capture, multiline_actual, cross_resolution_enhancement, False)
            logging.info(f"Color M-Mode data extracted successfully, shape: {rfdata.colorMModeData[0].shape if hasattr(rfdata.colorMModeData, '__getitem__') else 'N/A'}")
        else:
            logging.debug("No color M-Mode data found")
            
        return rfdata

    ###################################################################################
    # Extract dummy data from RF data
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
    # Extract SWI data from RF data
    ###################################################################################
    def _extract_swi_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract SWI data from RF data."""
        logging.info(f"Extracting SWI data, tap_point={tap_point}")
        logging.debug(f"SWI data types: {self.DataType_SWI}")
        
        multiline_capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if multiline_capture == 0:
            sample_rate = float(header_info.RF_Sample_Rate[0])
            multiline_capture = 16 if sample_rate == 0 else 32
            logging.debug(f"multiline_capture was 0, set to {multiline_capture} based on sample rate {sample_rate}")
            
        logging.debug(f"multiline_capture={multiline_capture}")
        
        xmit_events = len(header_info.Data_Type)
        swi_index = np.zeros(xmit_events).astype(bool)
        
        for dt in self.DataType_SWI:
            index = (header_info.Data_Type == dt)
            swi_index = np.bitwise_or(swi_index, index)
            
        swi_count = np.sum(swi_index)
        logging.debug(f"Found {swi_count} SWI data entries")
        
        if swi_count > 0:
            logging.debug("Processing SWI data")
            swi_data = self._prune_data(rfdata.lineData[:, swi_index], rfdata.lineHeader[:, swi_index], multiline_capture)
            
            multiline_actual = multiline_capture
            cross_resolution_enhancement = 1
            logging.debug(f"SWI sorting: multiline_actual={multiline_actual}, cross_resolution_enhancement={cross_resolution_enhancement}")
            
            rfdata.swiData = self._sort_rf(swi_data, multiline_capture, multiline_actual, cross_resolution_enhancement, False)
            logging.info(f"SWI data extracted successfully, shape: {rfdata.swiData[0].shape if hasattr(rfdata.swiData, '__getitem__') else 'N/A'}")
        else:
            logging.debug("No SWI data found")
            
        return rfdata

    ###################################################################################
    # Extract miscellaneous data from RF data
    ###################################################################################
    def _extract_misc_data(self, rfdata: 'PhilipsRfParser.Rfdata', header_info: 'PhilipsRfParser.HeaderInfoStruct', tap_point: int) -> 'PhilipsRfParser.Rfdata':
        """Extract miscellaneous data from RF data."""
        logging.info(f"Extracting miscellaneous data, tap_point={tap_point}")
        logging.debug(f"Misc data types: {self.DataType_Misc}")
        
        multiline_capture = 128 if tap_point == 7 else float(header_info.Multilines_Capture[0])
        if multiline_capture == 0:
            sample_rate = float(header_info.RF_Sample_Rate[0])
            multiline_capture = 16 if sample_rate == 0 else 32
            logging.debug(f"multiline_capture was 0, set to {multiline_capture} based on sample rate {sample_rate}")
            
        logging.debug(f"multiline_capture={multiline_capture}")
        
        xmit_events = len(header_info.Data_Type)
        misc_index = np.zeros(xmit_events).astype(bool)
        
        for dt in self.DataType_Misc:
            index = (header_info.Data_Type == dt)
            misc_index = np.bitwise_or(misc_index, index)
            
        misc_count = np.sum(misc_index)
        logging.debug(f"Found {misc_count} miscellaneous data entries")
        
        if misc_count > 0:
            logging.debug("Processing miscellaneous data")
            misc_data = self._prune_data(rfdata.lineData[:, misc_index], rfdata.lineHeader[:, misc_index], multiline_capture)
            
            multiline_actual = multiline_capture
            cross_resolution_enhancement = 1
            logging.debug(f"Misc sorting: multiline_actual={multiline_actual}, cross_resolution_enhancement={cross_resolution_enhancement}")
            
            rfdata.miscData = self._sort_rf(misc_data, multiline_capture, multiline_actual, cross_resolution_enhancement, False)
            logging.info(f"Miscellaneous data extracted successfully, shape: {rfdata.miscData[0].shape if hasattr(rfdata.miscData, '__getitem__') else 'N/A'}")
        else:
            logging.debug("No miscellaneous data found")
            
        return rfdata
    
    ###################################################################################
    # Save Header Summary
    ###################################################################################
    def _save_header_summary(self, numpy_folder: str):
        """Log a summary of header information."""
        logging.info("=== PHILIPS RF FILE HEADER SUMMARY ===")
        
        # Log each section of the summary
        self._write_header_info_section()
        self._write_db_params_section()
        self._write_calculated_params_section()
        self._write_data_shapes_section()
        self._write_available_data_section()

    ###################################################################################
    # Log header information section
    ###################################################################################
    def _write_header_info_section(self):
        """Log header information section."""
        if not hasattr(self.rfdata, 'headerInfo') or not self.rfdata.headerInfo:
            return
            
        h = self.rfdata.headerInfo
        
        # Number of headers/lines
        if hasattr(h, 'Data_Type') and h.Data_Type is not None:
            logging.info(f"Number of headers/lines: {len(h.Data_Type)}")
        
        # First header information
        logging.info("--- First Header Information ---")
        self._write_first_header_info(h)
        
        # Data types present
        self._write_data_types_info(h)
        
        # Frame and line information
        self._write_frame_info(h)
        
        # Time information
        self._write_time_info(h)

    ###################################################################################
    # Log first header information section
    ###################################################################################
    def _write_first_header_info(self, h):
        """Log information from the first header."""
        header_fields = [
            ('RF_CaptureVersion', 'RF Capture Version'),
            ('Tap_Point', 'Tap Point'),
            ('RF_Sample_Rate', 'RF Sample Rate'),
            ('Multilines_Capture', 'Multilines Capture'),
            ('Data_Gate', 'Data Gate')
        ]
        
        for attr, label in header_fields:
            if hasattr(h, attr) and getattr(h, attr) is not None:
                logging.info(f"{label}: {getattr(h, attr)[0]}")

    ###################################################################################
    # Data Types Information Section 
    ###################################################################################
    def _write_data_types_info(self, h):
        """Log information about data types."""
        logging.info("--- Data Types Present ---")
        if hasattr(h, 'Data_Type') and h.Data_Type is not None:
            unique_types = np.unique(h.Data_Type)
            logging.info(f"Unique Data Types: {unique_types}")
            
            # Count of each data type
            for dtype in unique_types:
                count = np.sum(h.Data_Type == dtype)
                logging.info(f"  Type {dtype}: {count} occurrences")

    ###################################################################################
    # Frame Information Section 
    ###################################################################################
    def _write_frame_info(self, h):
        """Log information about frames and lines."""
        logging.info("--- Frame Information ---")
        if hasattr(h, 'Frame_ID') and h.Frame_ID is not None:
            unique_frames = np.unique(h.Frame_ID)
            logging.info(f"Number of unique frames: {len(unique_frames)}")
            logging.info(f"Frame ID range: {unique_frames.min()} to {unique_frames.max()}")
        
        if hasattr(h, 'Line_Index') and h.Line_Index is not None:
            logging.info(f"Line index range: {h.Line_Index.min()} to {h.Line_Index.max()}")

    ###################################################################################
    # Time Information Section 
    ###################################################################################
    def _write_time_info(self, h):
        """Log time-related information."""
        logging.info("--- Time Information ---")
        if hasattr(h, 'Time_Stamp') and h.Time_Stamp is not None:
            logging.info(f"First timestamp: {h.Time_Stamp[0]}")
            logging.info(f"Last timestamp: {h.Time_Stamp[-1]}")

    ###################################################################################
    # Database Parameters Section 
    ###################################################################################
    def _write_db_params_section(self):
        """Log database parameters section."""
        if not hasattr(self.rfdata, 'dbParams') or not self.rfdata.dbParams:
            return
            
        logging.info("--- Database Parameters ---")
        db = self.rfdata.dbParams
        
        if hasattr(db, 'acqNumActiveScChannels2d') and db.acqNumActiveScChannels2d is not None:
            logging.info(f"Active scan channels: {db.acqNumActiveScChannels2d.shape}")
            logging.info(f"2D columns shape: {db.num2DCols.shape}")
            logging.info(f"2D columns first row: {db.num2DCols[0, :] if db.num2DCols.size > 0 else 'N/A'}")
        if hasattr(db, 'numOfSonoCTAngles2dActual') and db.numOfSonoCTAngles2dActual is not None:
            logging.info(f"SonoCT angles: {db.numOfSonoCTAngles2dActual.shape}")
        if hasattr(db, 'num2DCols') and db.num2DCols is not None:
            logging.info(f"2D columns shape: {db.num2DCols.shape}")
            logging.info(f"2D columns first row: {db.num2DCols[0, :] if db.num2DCols.size > 0 else 'N/A'}")

    ###################################################################################
    # Calculated Parameters Section 
    ###################################################################################
    def _write_calculated_params_section(self):
        """Log calculated parameters section."""
        if not hasattr(self, 'num_frames'):
            return
            
        logging.info("--- Calculated Parameters ---")
        logging.info(f"Number of frames: {self.num_frames}")
        logging.info(f"TX beams per frame: {self.tx_beams_per_frame}")
        logging.info(f"Number of SonoCT angles: {self.num_sonoct_angles}")
        logging.info(f"Multiline factor: {self.multiline_factor}")
        logging.info(f"Used OS: {self.offset_samples}")
        logging.info(f"PT: {self.points_per_line}")

    ###################################################################################
    # Data Shapes Section 
    ###################################################################################
    def _write_data_shapes_section(self):
        """Log data shapes section."""
        logging.info("--- Data Array Shapes ---")
        if hasattr(self.rfdata, 'lineData') and self.rfdata.lineData is not None:
            logging.info(f"Line data shape: {self.rfdata.lineData.shape}")
        if hasattr(self.rfdata, 'lineHeader') and self.rfdata.lineHeader is not None:
            logging.info(f"Line header shape: {self.rfdata.lineHeader.shape}")

    ###################################################################################
    # Available Data Types Section 
    ###################################################################################
    def _write_available_data_section(self):
        """Log available data types section."""
        logging.info("--- Available Data Arrays ---")
        data_attrs = ['echoData', 'cwData', 'pwData', 'colorData', 
                     'echoMModeData', 'colorMModeData', 'dummyData', 
                     'swiData', 'miscData']
        
        for attr in data_attrs:
            if hasattr(self.rfdata, attr):
                data = getattr(self.rfdata, attr)
                if data is not None:
                    if isinstance(data, (list, tuple)):
                        logging.info(f"{attr}: {len(data)} elements")
                        for i, elem in enumerate(data):
                            if hasattr(elem, 'shape'):
                                logging.info(f"  [{i}]: {elem.shape}")
                    elif hasattr(data, 'shape'):
                        logging.info(f"{attr}: {data.shape}")
                    else:
                        logging.info(f"{attr}: Available (unknown shape)")

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
    # Parameter Calculation
    ####################################################################################
    def _calculate_parameters(self) -> None:
        """Calculate and set main parameters as instance variables."""
        logging.info(f"Calculating parsing parameters...")
        
        # Calculate beam parameters
        self.tx_beams_per_frame = int(np.array(self.rfdata.dbParams.num2DCols).flat[0])
        self.num_sonoct_angles = int(self.rfdata.dbParams.numOfSonoCTAngles2dActual[0])
        logging.info(f"Beam parameters - tx_beams_per_frame: {self.tx_beams_per_frame}, num_sonoct_angles: {self.num_sonoct_angles}")
        
        # Calculate frame count
        self.num_frames = int(np.floor(self.rfdata.lineData.shape[1] / (self.tx_beams_per_frame * self.num_sonoct_angles)))
        self.multiline_factor = self.multiline_input
        logging.info(f"Calculated num_frames: {self.num_frames}, multiline_factor: {self.multiline_factor}")
        
        # Determine OS and PT parameters
        col = 0
        if np.any(self.rfdata.lineData[:, col] != 0):
            # Auto-detect based on data
            first_nonzero = np.where(self.rfdata.lineData[:, col] != 0)[0][0]
            last_nonzero = np.where(self.rfdata.lineData[:, col] != 0)[0][-1]
            self.offset_samples = first_nonzero  # Override default with detected value
            self.points_per_line = int(np.floor((last_nonzero - first_nonzero + 1) / self.multiline_factor))
            logging.info(f"Auto-detected: offset_samples={self.offset_samples}, points_per_line={self.points_per_line}")
        else:
            # Use the value provided in the constructor (default is 2256)
            logging.info(f"Using provided offset_samples={self.offset_samples}")
            self.points_per_line = int(np.floor((self.rfdata.lineData.shape[0] - self.offset_samples) / self.multiline_factor))
            logging.debug(f"Calculated points_per_line={self.points_per_line} based on shape {self.rfdata.lineData.shape[0]} and multiline_factor {self.multiline_factor}")

    ###################################################################################
    # Data Array Filling
    ####################################################################################
    def _fill_data_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fill RF data arrays for fundamental and harmonic signals."""
        logging.info(f"Filling RF data arrays for fundamental and harmonic signals...")
        
        # Preallocate arrays
        rftemp_all_harm = np.zeros((self.points_per_line, self.multiline_output * self.tx_beams_per_frame))
        rftemp_all_fund = np.zeros((self.points_per_line, self.multiline_output * self.tx_beams_per_frame))
        rf_data_all_harm = np.zeros((self.num_frames, self.num_sonoct_angles, self.points_per_line, self.multiline_output * self.tx_beams_per_frame))
        rf_data_all_fund = np.zeros((self.num_frames, self.num_sonoct_angles, self.points_per_line, self.multiline_output * self.tx_beams_per_frame))
        logging.debug(f"Preallocated arrays shapes - fund: {rf_data_all_fund.shape}, harm: {rf_data_all_harm.shape}")
        
        # Process each frame
        for k0 in range(self.num_frames):
            if k0 % max(1, self.num_frames // 10) == 0:
                logging.info(f"Processing frame {k0+1}/{self.num_frames}")
            
            # Process angles within frame
            for k1 in range(self.num_sonoct_angles):
                # Process beams within angle
                for k2 in range(self.tx_beams_per_frame):
                    bi = k0 * self.tx_beams_per_frame * self.num_sonoct_angles + k1 * self.tx_beams_per_frame + k2
                    
                    # Skip if beam index exceeds available data
                    if bi >= self.rfdata.lineData.shape[1]:
                        logging.warning(f"Skipping bi={bi} as it exceeds lineData columns {self.rfdata.lineData.shape[1]}")
                        continue
                    
                    # Extract data for this beam
                    idx0 = self.offset_samples + np.arange(self.points_per_line * self.multiline_factor)
                    idx1 = bi
                    
                    # Log first extraction for debugging
                    if k0 == 0 and k1 == 0 and k2 == 0:
                        logging.debug(f"First extraction - lineData[{idx0[0]}:{idx0[-1]+1}, {idx1}]")
                        logging.debug(f"lineData values sample: {self.rfdata.lineData[idx0, idx1][:10]}")
                    
                    # Reshape data for multiline
                    temp = np.transpose(
                        np.reshape(self.rfdata.lineData[idx0, idx1],
                                 (self.multiline_factor, self.points_per_line), order='F')
                    )
                    
                    # Log first reshape for debugging
                    if k0 == 0 and k1 == 0 and k2 == 0:
                        logging.debug(f"temp shape: {temp.shape}, temp sample: {temp.ravel()[:10]}")
                    
                    # Harmonic extraction
                    if temp.shape[1] > 2:
                        rftemp_all_harm[:, np.arange(self.multiline_output) + (k2 * self.multiline_output)] = temp[:, [0, 2]]
                    else:
                        logging.warning(f"temp has only {temp.shape[1]} columns, skipping harmonic assignment")
                    
                    # Fundamental extraction
                    if temp.shape[1] >= 12:
                        rftemp_all_fund[:, np.arange(self.multiline_output) + (k2 * self.multiline_output)] = temp[:, [9, 11]]
                    elif temp.shape[1] >= 2:
                        if k0 == 0 and k1 == 0 and k2 == 0:
                            logging.warning(f"temp has only {temp.shape[1]} columns, using last 2 columns for fundamental")
                        rftemp_all_fund[:, np.arange(self.multiline_output) + (k2 * self.multiline_output)] = temp[:, [-2, -1]]
                    else:
                        logging.warning(f"temp has only {temp.shape[1]} columns, skipping fundamental assignment")
                
                # Store arrays for this angle
                rf_data_all_harm[k0][k1] = rftemp_all_harm
                rf_data_all_fund[k0][k1] = rftemp_all_fund
        
        logging.info(f"RF data array filling complete")
        return rf_data_all_fund, rf_data_all_harm

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
        
        # Debug log the state of all variables before saving
        logging.debug("=== Data state before saving ===")
        logging.debug(f"data_to_save: {type(data_to_save) if data_to_save is not None else 'None'}")
        logging.debug(f"rf_data_all_fund: {type(rf_data_all_fund) if rf_data_all_fund is not None else 'None'}")
        logging.debug(f"rf_data_all_harm: {type(rf_data_all_harm) if rf_data_all_harm is not None else 'None'}")
        
        # Convert None values to empty numpy arrays
        data_to_save = data_to_save if data_to_save is not None else np.array([])
        rf_data_all_fund = rf_data_all_fund if rf_data_all_fund is not None else np.array([])
        rf_data_all_harm = rf_data_all_harm if rf_data_all_harm is not None else np.array([])
        
        # Convert HeaderInfoStruct and DbParams to dictionaries
        header_info_dict = {}
        if hasattr(self.rfdata, 'headerInfo'):
            for attr in dir(self.rfdata.headerInfo):
                if not attr.startswith('_'):  # Skip private attributes
                    value = getattr(self.rfdata.headerInfo, attr)
                    if value is None:
                        value = np.array([])
                    header_info_dict[attr] = value
                    
        db_params_dict = {}
        if hasattr(self.rfdata, 'dbParams'):
            for attr in dir(self.rfdata.dbParams):
                if not attr.startswith('_'):  # Skip private attributes
                    value = getattr(self.rfdata.dbParams, attr)
                    if value is None:
                        value = np.array([])
                    db_params_dict[attr] = value
        
        # Prepare contents dictionary with validation
        contents = {
            'echoData': data_to_save,
            'lineData': getattr(self.rfdata, 'lineData', np.array([])),
            'lineHeader': getattr(self.rfdata, 'lineHeader', np.array([])),
            'headerInfo': header_info_dict,
            'dbParams': db_params_dict,
            'rf_data_all_fund': rf_data_all_fund,
            'rf_data_all_harm': rf_data_all_harm,
            'NumFrame': self.num_frames,
            'NumSonoCTAngles': self.num_sonoct_angles,
            'pt': self.points_per_line,
            'multilinefactor': self.multiline_factor,
        }
        
        # Add optional data if available
        logging.debug(f"Adding optional data fields to MATLAB file")
        if hasattr(self.rfdata, 'echoData') and isinstance(self.rfdata.echoData, (list, tuple)) and len(self.rfdata.echoData) > 1:
            contents['echoData1'] = self.rfdata.echoData[1]
        if hasattr(self.rfdata, 'echoData') and isinstance(self.rfdata.echoData, (list, tuple)) and len(self.rfdata.echoData) > 2:
            contents['echoData2'] = self.rfdata.echoData[2]
        if hasattr(self.rfdata, 'echoData') and isinstance(self.rfdata.echoData, (list, tuple)) and len(self.rfdata.echoData) > 3:
            contents['echoData3'] = self.rfdata.echoData[3]
        if hasattr(self.rfdata, 'echoMModeData') and self.rfdata.echoMModeData is not None:
            contents['echoMModeData'] = self.rfdata.echoMModeData
        else:
            contents['echoMModeData'] = np.array([])
        if hasattr(self.rfdata, 'miscData') and self.rfdata.miscData is not None:
            contents['miscData'] = self.rfdata.miscData
        else:
            contents['miscData'] = np.array([])
        
        # Remove existing file if necessary
        if os.path.exists(destination):
            logging.debug(f"Removing existing file: {destination}")
            os.remove(destination)
        
        # Save the file
        try:
            savemat(destination, contents)
            logging.info(f"MATLAB file saved successfully: {destination}")
            return np.array(rf_data_all_fund).shape if rf_data_all_fund is not None else (0,)
        except Exception as e:
            logging.error(f"Error saving MATLAB file: {str(e)}")
            logging.error("Contents that failed to save:")
            for key, value in contents.items():
                logging.error(f"{key}: type={type(value)}")
            raise

    ###################################################################################
    # Restore logging configuration
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
# Main Execution
###################################################################################
if __name__ == "__main__":
    # === Single Logging Level Control ===
    # Set to True for debug mode (more detailed logs), False for info mode (less detailed)
    DEBUG_MODE = True
    
    # Set logging levels based on DEBUG_MODE
    LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    
    # Create a detailed formatter that includes function names
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s'
    )
    
    # Console handler only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(detailed_formatter)
    
    # Clear any existing handlers and add only console handler
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Log system information for diagnostic purposes
    logging.debug("==== Logging Activated ====")
    logging.debug(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    logging.debug(f"Log Level: {logging.getLevelName(LOG_LEVEL)}")
    logging.debug(f"Python version: {platform.python_version()}")
    logging.debug(f"Platform: {platform.platform()}")
    logging.debug(f"Current directory: {os.getcwd()}")
    
    # Hardcoded file path - no command line arguments needed
    #filepath = r"D:\Omid\0_samples\Philips\David\sample.rf"
    filepath = r"D:\Omid\0_samples\Philips\UKDFIBEPIC003\UKDFIBEPIC003INTER4D_20250424_094124.rf"

    logging.info(f"Starting main execution with file: {filepath}")
    parser = PhilipsRfParser()
    parser.philipsRfParser(filepath, save_numpy=False)
    logging.info("Main execution complete")
      
    ###################################################################################