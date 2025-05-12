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
    """Main class for parsing Philips RF data files.

    RF Data Format Details
    ---------------------
    The parser supports two RF data formats from Philips ultrasound systems:

    1. Voyager Format (36-byte blocks):
    2. Fusion Format (32-byte blocks):

    Block Alignment Significance
    ---------------------------
    * Voyager (36B):
      - Simpler sequential access
      - Each sample is contiguous
      - Less efficient memory alignment
      - Designed for older systems

    * Fusion (32B):
      - Optimized for modern CPU cache lines
      - More complex sample reconstruction
      - Better memory efficiency
      - Faster processing on modern systems

    Common Features
    --------------
    * 12 samples per block
    * 21-bit RF data per sample
    * 3 header bits per sample
    * Supports all ultrasound modes:
      - B-mode (Echo)
      - Color Flow Doppler
      - M-mode
      - Continuous Wave Doppler
      - Pulsed Wave Doppler
      - Shear Wave Imaging

    The parsing process involves:
    1. Reading and validating file headers
    2. Loading raw RF data in blocks
    3. Parsing metadata headers
    4. Processing RF signal data
    5. Organizing data by type (echo, color, etc.)
    6. Saving processed data to output file
    """
    
    ###################################################################################
    class HeaderInfoStruct:
        """Philips-specific structure containing information from the headers.
        
        This structure stores metadata about the RF capture, including:
        - Version and format information
        - Capture configuration (tap points, gates, multilines)
        - Timing and synchronization data
        - Line and frame indexing
        
        The metadata is crucial for:
        1. Interpreting the raw RF data correctly
        2. Reconstructing images from the data
        3. Maintaining data quality and synchronization
        4. Processing different ultrasound modes (B-mode, Color, etc.)
        """
        def __init__(self):
            # Version and format info
            self.RF_CaptureVersion = []    # Version number of RF capture format (e.g., 2-6)
                                           # Different versions have different features and data organization
            
            self.Data_Format = []          # Format identifier for the RF data
                                           # Determines how the binary data should be interpreted
            
            self.Data_Type = []            # Type of ultrasound data:
                                           # 1-14: Echo (B-mode)
                                           # 15,88,89: Miscellaneous
                                           # 16: Continuous Wave Doppler
                                           # 17,21-24: Color Flow
                                           # 18,19: Pulsed Wave Doppler
                                           # 20,25,29-31: Dummy Lines
                                           # 26: Echo M-mode
                                           # 27,28: Color M-mode
                                           # 90,91: SWI (Shear Wave Imaging)
            
            # Capture configuration
            self.Tap_Point = []            # Point in processing chain where data was captured:
                                           # 0: PostShepard (after Shepard filter)
                                           # 1: PostAGNOS (after AGNOS processing)
                                           # 2: PostXBR (after cross-beam reconstruction)
                                           # 3: PostQBP (after quadrature bandpass)
                                           # 4: PostADC (raw data from ADC)
            
            self.Data_Gate = []            # Gate information for data validation
                                           # Used to verify data integrity and timing
            
            self.Multilines_Capture = []   # Number of scan lines captured simultaneously
                                           # Higher numbers mean faster frame rates but more processing
            
            self.Steer = []                # Beam steering information (6 bits)
                                           # Controls the angle and direction of ultrasound beam
            
            self.elevationPlaneOffset = [] # Offset in elevation plane (8 bits)
                                           # Used in 3D/4D imaging for slice positioning
            
            # Processing parameters
            self.PM_Index = []             # Processing mode index (2 bits)
                                           # Indicates specific processing algorithms to use
            
            self.Pulse_Index = []          # Pulse sequence index (16 bits)
                                           # Identifies specific ultrasound pulse pattern used
            
            self.RF_Sample_Rate = []       # RF data sampling rate
                                           # Determines temporal resolution of the data
            
            # Data organization
            self.Header_Tag = []           # Header identification tag (16 bits)
                                           # Used for data block identification
            
            self.Threed_Pos = []           # 3D position information (16 bits)
                                           # Used in 3D/4D imaging for spatial location
            
            self.Mode_Info = []            # Mode-specific information (16 bits)
                                           # Contains settings specific to imaging mode
            
            self.Frame_ID = []             # Frame identification number (32 bits)
                                           # Used to track frame sequence
            
            self.CSID = []                 # Capture sequence ID (16 bits)
                                           # Groups related captures together
            
            self.Line_Index = []           # Line index in frame (16 bits)
                                           # Position of scan line within frame
            
            self.Line_Type = []            # Type of line data (16 bits)
                                           # Specific information about line content
            
            self.Time_Stamp = 0            # Timestamp of capture (32 bits)
                                           # Used for temporal synchronization
                                           
    ###################################################################################
    class dbParams:
        """Philips-specific structure containing signal properties of the scan.
        
        This structure stores parameters that define how the ultrasound scan
        was configured and how the data should be processed, including:
        - Channel configuration
        - Multiline processing parameters
        - Scan geometry and dimensions
        """
        def __init__(self):
            # Channel configuration
            self.acqNumActiveScChannels2d = []        # Number of active scan channels in 2D
            self.azimuthMultilineFactorXbrOut = []    # Output multiline factor in azimuth
            self.azimuthMultilineFactorXbrIn = []     # Input multiline factor in azimuth
            
            # Scan parameters
            self.numOfSonoCTAngles2dActual = []       # Actual number of SonoCT angles
            self.elevationMultilineFactor = []        # Multiline factor in elevation
            self.numPiPulses = []                     # Number of pulse inversion pulses
            self.num2DCols = []                       # Number of columns in 2D image
            
            # Processing flags
            self.fastPiEnabled = []                   # Fast pulse inversion enabled flag
            self.zigZagEnabled = []                   # Zig-zag scanning enabled flag
            
            # Scan geometry
            self.numZones2d = []                      # Number of focal zones in 2D
            self.numSubVols = []                      # Number of sub-volumes
            self.numPlanes = []                       # Number of scan planes
            
            # Color flow parameters
            self.azimuthMultilineFactorXbrOutCf = []  # Color flow output multiline factor
            self.azimuthMultilineFactorXbrInCf = []   # Color flow input multiline factor
            self.multiLineFactorCf = []               # Color flow multiline factor
            self.linesPerEnsCf = []                   # Lines per ensemble in color flow
            self.ensPerSeqCf = []                     # Ensembles per sequence in color flow
            self.numCfCols = []                       # Number of color flow columns
            self.numCfEntries = []                    # Number of color flow entries
            self.numCfDummies = []                    # Number of dummy color flow lines
            self.elevationMultilineFactorCf = []      # Color flow elevation multiline factor
            self.Planes = []                          # Plane information
            self.tapPoint = []                        # Tap point information
            
    ###################################################################################
    class Rfdata:
        """Philips-specific structure containing constructed RF data.
        
        This is the main container for processed RF data and associated metadata.
        It holds both the raw RF data and its organization into different modes
        (echo, color flow, M-mode, etc.)
        """
        def __init__(self):
            self.lineData = None           # Array containing interleaved line data (Data x XmitEvents)
            self.lineHeader = None         # Array containing qualifier bits of the line data
            self.headerInfo: PhilipsRFParser.HeaderInfoStruct = None  # Header information
            self.dbParams: PhilipsRFParser.dbParams = None            # Scan parameters
            self.echoData = None           # Array containing echo line data
            self.echoMModeData = []        # M-mode echo data
            self.miscData = []             # Miscellaneous data

    ###################################################################################

    def __init__(self, filepath: str, save_format: str = 'mat'):
        """Initialize the PhilipsRFParser.
        
        Sets up the header signatures for different file formats and initializes
        basic parameters. The parser recognizes two formats:
        1. Voyager format (VHeader)
        2. Fusion format (FHeader)
        """
        # Configuration
        self.VHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 160, 160]  # Voyager format
        self.FHeader = [0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 11, 11]    # Fusion format
        self.fileHeaderSize = len(self.VHeader)  # Both headers are same length
        
        # Input
        self.filepath = filepath # Path to the RF data file
        self.save_format = save_format # Format to save data in ('mat', 'npz', 'npy', or 'both') (default: 'mat')
        
        self.ML_out=2 # Output multiline factor (default: 2)
        self.ML_in=32 # Input multiline factor (default: 32)
        self.used_os=2256 # Used oversampling (default: 2256)
        self.save_format='mat' # Format to save data in ('mat', 'npz', 'npy', or 'both') (default: 'mat')
        
        # State (results)
        self.rfdata = None
        
        # Run workflow   
        self.__run()
  
    ###################################################################################
    
    def __run(self) -> None:
        """Parse Philips RF data file and save to .mat, .npz, or individual .npy files.
        Stores all results as instance variables for later access.
        """
        logging.info(f"Starting philipsRfParser for file: {self.filepath}")

        self._validate_save_format()
        self._parse_rf_file()
        self._clean_duplicate_lines()
        self._calculate_parameters()
        self._initialize_data_arrays()
        self._fill_data_arrays()
        self._prepare_save_contents()
        self._save_contents()
        
        logging.info(f"philipsRfParser completed for file: {self.filepath}")

    ###################################################################################
    
    def _validate_save_format(self):
        """Validate the save format."""
        if self.save_format not in ['mat', 'npz', 'npy', 'both']:
            raise ValueError("save_format must be one of: 'mat', 'npz', 'npy', 'both'")

    ###################################################################################

    def _parse_rf_file(self):
        """Parse the RF file and store result in self.rf."""
        logging.info("Parsing RF file...")
        self._parse_rf(readOffsetMB=0, readSizeMB=2000)
        logging.info("RF file parsing completed")

    ###################################################################################
    
    def _parse_rf(self, readOffsetMB, readSizeMB):
        """Parse RF data from file.
        
        This method handles the main RF data parsing process:
        1. Analyzes header to determine file type (Voyager/Fusion)
        2. Aligns read offsets to block boundaries
        3. Loads raw data in blocks
        4. Parses headers and metadata
        5. Processes RF signal data
        
        Args:
            readOffset (int): Starting offset in bytes
            readSize (int): Size to read in bytes
            
        Returns:
            Rfdata: Parsed RF data structure
        """
        logging.info(f"Starting parseRF for file: {self.filepath}")
        
        # Initialize RF data structure
        self.rfdata = self.Rfdata()
        logging.info("Initialized Rfdata structure")
        
        # Analyze header to determine file type and size
        isVoyager, hasFileHeader, totalHeaderSize = self._analyze_header()
        logging.info(f"File analysis results - isVoyager: {isVoyager}, hasFileHeader: {hasFileHeader}, totalHeaderSize: {totalHeaderSize}")
                
        # Align read offsets to block boundaries
        readOffset, readSize = self._align_offsets(self.filepath, readOffsetMB, readSizeMB, totalHeaderSize, isVoyager)
        logging.info(f"Aligned read parameters - offset: {readOffset}, size: {readSize}")
        
        # Load raw data from file
        logging.info("Loading raw data from file...")
        rawrfdata = self._load_raw_data(self.filepath, readOffset, readSize, totalHeaderSize, isVoyager)
        logging.info(f"Loaded raw data of size: {rawrfdata.shape}")
        
        # Parse metadata headers
        logging.info("Parsing metadata headers...")
        headerInfo = self.parse_header_v(rawrfdata) if isVoyager else self.parse_header_f(rawrfdata)
        logging.info("Header parsing completed")
        
        # Parse RF signal data
        logging.info("Parsing RF signal data...")
        lineData, lineHeader, Tap_Point = self._parse_rf_data(rawrfdata, headerInfo, isVoyager)
        logging.info(f"Signal data parsed - lineData shape: {lineData.shape}, lineHeader shape: {lineHeader.shape}")
        
        # Store parsed data in rfdata structure
        self.rfdata.lineData = lineData
        self.rfdata.lineHeader = lineHeader
        self.rfdata.headerInfo = headerInfo
        
        del rawrfdata  # Free memory
        logging.info("Stored parsed data in rfdata structure")
        
        # Determine multiline capture configuration
        ML_Capture, Tap_Point = self._determine_capture_info(headerInfo, Tap_Point)
        logging.info(f"Capture configuration determined - ML_Capture: {ML_Capture}, Tap_Point: {Tap_Point}")
        
        # Log capture information
        self._log_capture_info(Tap_Point, ML_Capture)
        
        # Organize RF data into different types
        logging.info("Organizing RF data into different types...")
        self._organize_rfdata(self.rfdata, ML_Capture, isVoyager)
        logging.info("RF data organization completed")

    ###################################################################################
    
    def _analyze_header(self):
        """Analyze the file header to determine file type and structure.
        
        This method reads the initial bytes of the file to:
        1. Determine if it's a Voyager or Fusion format file
        2. Check if it has a standard header structure
        3. Calculate the total header size
        
        Args:
            filepath (str): Path to the RF data file
            
        Returns:
            tuple: (isVoyager, hasFileHeader, totalHeaderSize)
                - isVoyager (bool): True if Voyager format, False if Fusion
                - hasFileHeader (bool): True if standard header present
                - totalHeaderSize (int): Total size of all headers in bytes
        """
        with open(self.filepath, 'rb') as file_obj:
            # Read initial header bytes
            fileHeader = list(file_obj.read(self.fileHeaderSize))
            
            # Check format by comparing with known signatures
            isVoyager = fileHeader == self.VHeader
            hasFileHeader = fileHeader in (self.VHeader, self.FHeader)

            if hasFileHeader:
                logging.info("Header information found.")
                logging.info("Parsing Voyager RF capture file..." if isVoyager else "Parsing Fusion RF capture file...")
                
                # Parse file header to get additional size information
                endianness = 'big' if isVoyager else 'little'
                logging.info(f"Parsing file header with endianness: {endianness}")
                
                self.rfdata.dbParams, numFileHeaderBytes = self._parse_file_header(file_obj, endianness)
                totalHeaderSize = self.fileHeaderSize + 8 + numFileHeaderBytes
                logging.info(f"Total header size: {totalHeaderSize}")
            else:
                logging.info("Parsing legacy Voyager RF capture file (no standard header).")
                isVoyager = True
                totalHeaderSize = 0

        return isVoyager, hasFileHeader, totalHeaderSize
    
    ###################################################################################

    def _parse_file_header(self, file_obj, endianness):
        """Parse the file header to extract scan parameters.

        The file header contains important information about:
        - File version and format
        - Scan configuration (channels, multiline factors)
        - Geometry parameters (zones, planes)
        - Color flow settings

        Args:
            file_obj (file): Open file object positioned at header start
            endianness (str): 'big' for Voyager, 'little' for Fusion

        Returns:
            tuple: (dbParams, numFileHeaderBytes)
                - dbParams (dbParams): Parsed scan parameters
                - numFileHeaderBytes (int): Size of file header in bytes
        """
        logging.info("Starting parseFileHeader.")

        # Read basic header information
        fileVersion = int.from_bytes(file_obj.read(4), endianness, signed=False)
        logging.info(f"File Version: {fileVersion}")
        
        numFileHeaderBytes = int.from_bytes(file_obj.read(4), endianness, signed=False)
        logging.info(f"Header Size: {numFileHeaderBytes} bytes")

        temp_dbParams = self.dbParams()

        # Parse header based on version
        if fileVersion in {2, 3, 4, 5, 6}:
            # Common fields for all versions
            temp_dbParams.acqNumActiveScChannels2d = [
                int.from_bytes(file_obj.read(4), endianness, signed=False) 
                for _ in range(3 if fileVersion >= 4 else 4)
            ]
            temp_dbParams.azimuthMultilineFactorXbrOut = [
                int.from_bytes(file_obj.read(4), endianness, signed=False) 
                for _ in range(3 if fileVersion >= 4 else 4)
            ]
            temp_dbParams.azimuthMultilineFactorXbrIn = [
                int.from_bytes(file_obj.read(4), endianness, signed=False) 
                for _ in range(3 if fileVersion >= 4 else 4)
            ]

            # Version 4+ specific fields
            if fileVersion >= 4:
                temp_dbParams.azimuthMultilineFactorXbrOutCf = [
                    int.from_bytes(file_obj.read(4), endianness, signed=False) 
                    for _ in range(3)
                ]
                temp_dbParams.azimuthMultilineFactorXbrInCf = [
                    int.from_bytes(file_obj.read(4), endianness, signed=False) 
                    for _ in range(3)
                ]

            # Common fields continued
            temp_dbParams.numOfSonoCTAngles2dActual = [
                int.from_bytes(file_obj.read(4), endianness, signed=False) 
                for _ in range(3 if fileVersion >= 4 else 4)
            ]
            temp_dbParams.elevationMultilineFactor = [
                int.from_bytes(file_obj.read(4), endianness, signed=False) 
                for _ in range(3 if fileVersion >= 4 else 4)
            ]

            # Version 4+ specific fields
            if fileVersion >= 4:
                temp_dbParams.elevationMultilineFactorCf = [
                    int.from_bytes(file_obj.read(4), endianness, signed=False) 
                    for _ in range(3)
                ]

            # Version 5+ specific fields
            if fileVersion >= 5:
                temp_dbParams.multiLineFactorCf = [
                    int.from_bytes(file_obj.read(4), endianness, signed=False) 
                    for _ in range(3)
                ]

            # Common fields continued
            temp_dbParams.numPiPulses = [
                int.from_bytes(file_obj.read(4), endianness, signed=False) 
                for _ in range(3 if fileVersion >= 4 else 4)
            ]
            
            # Read 2D columns matrix (14x11)
            temp_dbParams.num2DCols = np.reshape(
                [int.from_bytes(file_obj.read(4), endianness, signed=False) for _ in range(14 * 11)],
                (14, 11), order='F'
            )
            
            temp_dbParams.fastPiEnabled = [
                int.from_bytes(file_obj.read(4), endianness, signed=False) 
                for _ in range(3 if fileVersion >= 4 else 4)
            ]
            temp_dbParams.numZones2d = [
                int.from_bytes(file_obj.read(4), endianness, signed=False) 
                for _ in range(3 if fileVersion >= 4 else 4)
            ]

            # Geometry parameters
            temp_dbParams.numSubVols = int.from_bytes(file_obj.read(4), endianness, signed=False)
            temp_dbParams.numPlanes = int.from_bytes(file_obj.read(4), endianness, signed=False)
            temp_dbParams.zigZagEnabled = int.from_bytes(file_obj.read(4), endianness, signed=False)

            # Version 3+ color flow parameters
            if fileVersion in {3, 4, 5, 6}:
                temp_dbParams.linesPerEnsCf = [
                    int.from_bytes(file_obj.read(4), endianness, signed=False) 
                    for _ in range(3 if fileVersion >= 4 else 4)
                ]
                temp_dbParams.ensPerSeqCf = [
                    int.from_bytes(file_obj.read(4), endianness, signed=False) 
                    for _ in range(3 if fileVersion >= 4 else 4)
                ]
                temp_dbParams.numCfCols = [
                    int.from_bytes(file_obj.read(4), endianness, signed=False) 
                    for _ in range(14)
                ]
                temp_dbParams.numCfEntries = [
                    int.from_bytes(file_obj.read(4), endianness, signed=False) 
                    for _ in range(3 if fileVersion >= 4 else 4)
                ]
                temp_dbParams.numCfDummies = [
                    int.from_bytes(file_obj.read(4), endianness, signed=False) 
                    for _ in range(3 if fileVersion >= 4 else 4)
                ]

            # Version 6 specific fields
            if fileVersion == 6:
                temp_dbParams.tapPoint = int.from_bytes(file_obj.read(4), endianness, signed=False)

        else:
            logging.error(f"Unknown file version: {fileVersion}")
            raise ValueError(f"Unknown file version: {fileVersion}")

        return temp_dbParams, numFileHeaderBytes
    
    ###################################################################################
    
    def _align_offsets(self, filepath, readOffsetMB, readSizeMB, totalHeaderSize, isVoyager):
        
        """Align read offsets to block boundaries.
        
        The RF data is organized in fixed-size blocks:
        - 36 bytes for Voyager format
        - 32 bytes for Fusion format
        This method ensures reads start and end on block boundaries.
        
        Args:
            filepath (str): Path to the RF data file
            readOffsetMB (int): Starting offset in megabytes
            readSizeMB (int): Size to read in megabytes
            totalHeaderSize (int): Total size of headers in bytes
            isVoyager (bool): Whether file is in Voyager format
            
        Returns:
            tuple: (readOffset, readSize)
                - readOffset (int): Aligned read offset in bytes
                - readSize (int): Aligned read size in bytes
        """
        # Convert MB to bytes
        readOffset = readOffsetMB * (2**20) 
        readSize = readSizeMB * (2**20) 
        
        # Calculate remaining file size after headers
        remainingSize = os.path.getsize(filepath) - totalHeaderSize
        logging.info(f"Remaining size: {remainingSize}")

        # Set block alignment based on format
        alignment = 36 if isVoyager else 32
        aligned = np.arange(0, remainingSize + 1, alignment)

        # Ensure we don't exceed the file size
        readOffset = min(readOffset, remainingSize)
        readSize = min(readSize, remainingSize - readOffset)

        # Align to block boundaries
        readOffset = aligned[np.searchsorted(aligned, readOffset)]
        readSize = aligned[np.searchsorted(aligned, readSize) - 1]  # Use largest aligned size that fits

        return readOffset, readSize
    
    ###################################################################################
    
    def _load_raw_data(self, filepath, readOffset, readSize, totalHeaderSize, isVoyager):
        """Load raw data from file.
        
        Reads the raw RF data from file, handling the different formats:
        - Voyager: Direct binary read
        - Fusion: Uses getPartA and getPartB to read data in parts
        
        Args:
            filepath (str): Path to the RF data file
            readOffset (int): Starting offset in bytes
            readSize (int): Size to read in bytes
            totalHeaderSize (int): Total size of headers in bytes
            isVoyager (bool): Whether file is in Voyager format
            
        Returns:
            np.ndarray: Raw data array
        """
        # Open file and seek to start position
        with open(filepath, 'rb') as f:
            f.seek(totalHeaderSize + readOffset)
            rawrfdata = f.read(readSize)

        if isVoyager:
            # Voyager format - return raw data
            return rawrfdata
        else:
            # Fusion format - process in parts
            numClumps = int(np.floor(readSize / 32))
            offset = totalHeaderSize + readOffset
            
            # Get parts A and B using C extensions
            partA = self.call_get_part_a(numClumps, filepath, offset)
            partB = self.call_get_part_b(numClumps, filepath, offset)
            
            # Combine parts
            return np.concatenate((partA, partB))

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
            temp_headerInfo.RF_CaptureVersion.append(int(packedHeader[iBit:iBit + 4], 2))
            iBit += 4
            temp_headerInfo.Tap_Point.append(int(packedHeader[iBit:iBit + 3], 2))
            iBit += 3
            temp_headerInfo.Data_Gate.append(int(packedHeader[iBit], 2))
            iBit += 1
            temp_headerInfo.Multilines_Capture.append(int(packedHeader[iBit:iBit + 4], 2))
            iBit += 4
            temp_headerInfo.RF_Sample_Rate.append(int(packedHeader[iBit], 2))
            iBit += 1
            temp_headerInfo.Steer.append(int(packedHeader[iBit:iBit + 6], 2))
            iBit += 6
            temp_headerInfo.elevationPlaneOffset.append(int(packedHeader[iBit:iBit + 8], 2))
            iBit += 8
            temp_headerInfo.PM_Index.append(int(packedHeader[iBit:iBit + 2], 2))
            iBit += 2
            temp_headerInfo.Line_Index.append(int(packedHeader[iBit:iBit + 16], 2))
            iBit += 16
            temp_headerInfo.Pulse_Index.append(int(packedHeader[iBit:iBit + 16], 2))
            iBit += 16
            temp_headerInfo.Data_Format.append(int(packedHeader[iBit:iBit + 16], 2))
            iBit += 16
            temp_headerInfo.Data_Type.append(int(packedHeader[iBit:iBit + 16], 2))
            iBit += 16
            temp_headerInfo.Header_Tag.append(int(packedHeader[iBit:iBit + 16], 2))
            iBit += 16
            temp_headerInfo.Threed_Pos.append(int(packedHeader[iBit:iBit + 16], 2))
            iBit += 16
            temp_headerInfo.Mode_Info.append(int(packedHeader[iBit:iBit + 16], 2))
            iBit += 16
            temp_headerInfo.Frame_ID.append(int(packedHeader[iBit:iBit + 32], 2))
            iBit += 32
            temp_headerInfo.CSID.append(int(packedHeader[iBit:iBit + 16], 2))
            iBit += 16
            temp_headerInfo.Line_Type.append(int(packedHeader[iBit:iBit + 16], 2))
            iBit += 16
            temp_headerInfo.Time_Stamp = int(packedHeader[iBit:iBit + 32], 2)

        return temp_headerInfo

    ###################################################################################

    def parse_header_f(self, rawrfdata):
        """Parse header in Fusion format."""
        logging.info("Starting parseHeaderF function.")
        temp_headerInfo = self.HeaderInfoStruct()

        # Find header locations - try different header patterns
        header_patterns = [
            (rawrfdata[0, :] & 1572864) == 524288,  # Original pattern
            (rawrfdata[0, :] & 0xFF) == 0x00,       # Alternative pattern 1
            (rawrfdata[1, :] & 0xFF) == 0xFF        # Alternative pattern 2
        ]
        
        for pattern in header_patterns:
            iHeader = np.where(pattern)[0]
            if len(iHeader) > 1:
                logging.info(f"Found {len(iHeader)-1} headers using pattern")
                break
        else:
            logging.error("No valid headers found in the data")
            raise ValueError("No valid headers found in the data")
            
        numHeaders = len(iHeader) - 1
        logging.info(f"Processing {numHeaders} headers")

        for m in range(numHeaders):
            # Default values for Fusion format
            temp_headerInfo.RF_CaptureVersion.append(4)  # Version 4 for Fusion
            temp_headerInfo.Tap_Point.append(4)  # PostADC for Fusion
            temp_headerInfo.Data_Gate.append(0)
            temp_headerInfo.Multilines_Capture.append(16)  # Default for Fusion
            temp_headerInfo.RF_Sample_Rate.append(0)
            temp_headerInfo.Steer.append(0)
            temp_headerInfo.elevationPlaneOffset.append(0)
            temp_headerInfo.PM_Index.append(0)
            
            # Parse actual data from headers
            header_start = iHeader[m]
            header_end = iHeader[m+1]
            header_data = rawrfdata[:, header_start].copy()
            
            # Extract values from header data
            temp_headerInfo.Line_Index.append(m)  # Use sequential numbering
            temp_headerInfo.Pulse_Index.append(int(np.floor(m/16)))  # Group by 16
            temp_headerInfo.Data_Format.append(1)  # Default format
            temp_headerInfo.Data_Type.append(1)  # Default type
            
            # Default values for remaining fields
            temp_headerInfo.Header_Tag.append(0)
            temp_headerInfo.Threed_Pos.append(0)
            temp_headerInfo.Mode_Info.append(0)
            temp_headerInfo.Frame_ID.append(0)
            temp_headerInfo.CSID.append(0)
            temp_headerInfo.Line_Type.append(0)
            
        temp_headerInfo.Time_Stamp = 0

        logging.info(f"Successfully parsed {len(temp_headerInfo.Tap_Point)} headers")
        return temp_headerInfo

    ###################################################################################
    
    def _parse_rf_data(self, rawrfdata, headerInfo, isVoyager):
        """Parse RF signal data from raw data.
        
        This method extracts the actual RF signal data from the raw data blocks.
        The data format depends on whether it's Voyager or Fusion format.
        
        Args:
            rawrfdata (np.ndarray): Raw data from file
            headerInfo (HeaderInfoStruct): Parsed header information
            isVoyager (bool): Whether the file is in Voyager format
            
        Returns:
            tuple: (lineData, lineHeader, Tap_Point)
                - lineData: RF signal data samples
                - lineHeader: Header information for each line
                - Tap_Point: Data capture point in processing chain
        """
        # Get tap point from header
        Tap_Point = headerInfo.Tap_Point[0]
        logging.info(f"Tap Point: {Tap_Point}")
        
        # Parse data based on format
        if isVoyager:
            lineData, lineHeader = self.parse_data_v(rawrfdata, headerInfo)
        else:
            lineData, lineHeader = self.parse_data_f(rawrfdata, headerInfo)
            
            # Apply tap point correction for Fusion format
            if Tap_Point == 0:
                lineData <<= 2  # Left shift by 2 bits
                
        return lineData, lineHeader, Tap_Point
    
    ###################################################################################

    def parse_data_f(self, rawrfdata, headerInfo):
        """Parse data in Fusion format.
        
        The Fusion format organizes data in 32-byte blocks:
        - Each block contains 12 samples of 21-bit data
        - Each sample has associated header bits
        - Data is organized in lines and frames
        
        Args:
            rawrfdata (np.ndarray): Raw data from file
            headerInfo (HeaderInfoStruct): Parsed header information
            
        Returns:
            tuple: (lineData, lineHeader)
                - lineData: RF signal data samples
                - lineHeader: Header information for each line
        """
        logging.info("Starting parseDataF function.")
        minNeg = 2**18  # Minimum negative value for 21-bit signed data

        # Find data sections using header pattern
        iHeader = np.array(np.where((rawrfdata[0, :] & 1572864) == 524288))[0]
        if len(iHeader) <= 1:
            logging.error("No valid data sections found")
            return np.array([]), np.array([])
            
        numHeaders = len(iHeader) - 1
        logging.info(f"Found {numHeaders} data sections")

        # Calculate maximum number of samples per line
        maxNumSamples = 0
        for m in range(numHeaders):
            tempMax = iHeader[m+1] - iHeader[m] - 1
            if tempMax > maxNumSamples:
                maxNumSamples = tempMax

        # Initialize output arrays
        numSamples = maxNumSamples
        lineData = np.zeros((numSamples, numHeaders), dtype=np.int32)
        lineHeader = np.zeros((numSamples, numHeaders), dtype=np.uint8)
        logging.info(f"Initialized arrays with shape: ({numSamples}, {numHeaders})")

        # Process each data section
        for m in range(numHeaders):
            # Calculate section boundaries
            iStartData = iHeader[m] + 1
            iStopData = iHeader[m+1]

            # Extract data section
            data_section = rawrfdata[:, iStartData:iStopData]
            
            # Process samples
            samples = data_section[0, :]  # Take first row as samples
            samples = samples.astype(np.int32)
            
            # Handle negative values (21-bit signed data)
            neg_mask = samples >= minNeg
            samples[neg_mask] -= (2 * minNeg)
            
            # Extract header bits
            header_bits = (data_section[0, :] & 1572864) >> 19
            
            # Store in output arrays
            n_samples = len(samples)
            lineData[:n_samples, m] = samples
            lineHeader[:n_samples, m] = header_bits

        logging.info(f"Processed data shape: lineData {lineData.shape}, lineHeader {lineHeader.shape}")
        return lineData, lineHeader

    ###################################################################################

    def parse_data_v(self, rawrfdata, headerInfo):
        """Parse data in Voyager format.
        
        The Voyager format organizes data in 36-byte blocks:
        - Each block contains 12 samples of 21-bit data
        - Each sample has associated header bits
        - Data is organized with different alignment than Fusion
        
        Args:
            rawrfdata (np.ndarray): Raw data from file
            headerInfo (HeaderInfoStruct): Parsed header information
            
        Returns:
            tuple: (lineData, lineHeader)
                - lineData: RF signal data samples
                - lineHeader: Header information for each line
        """
        logging.info("Starting parseDataV function.")
        minNeg = 16 * (2**16)  # Minimum negative value for Voyager format

        # Find header locations
        iHeader = np.where((rawrfdata[2, 0, :] & 224) == 64)[0]
        numHeaders = len(iHeader) - 1

        if numHeaders <= 0:
            logging.error("Not enough headers found! Exiting.")
            return None, None

        # Calculate number of samples per line
        numSamples = (iHeader[1] - iHeader[0] - 1) * 12
        lineData = np.zeros((numSamples, numHeaders), dtype=np.int32)
        lineHeader = np.zeros((numSamples, numHeaders), dtype=np.uint8)

        # Process each header section
        for m in range(numHeaders):
            iStartData = iHeader[m] + 1
            iStopData = iHeader[m + 1] - 1

            # Handle special data type
            if headerInfo.Data_Type[m] == float(0x5A):
                iStopData = iStartData + 10000

            # Extract and process data
            lineData_u8 = rawrfdata[:, :, iStartData:iStopData]
            lineData_s32 = (np.int32(lineData_u8[0, :, :]) +
                           (np.int32(lineData_u8[1, :, :]) << 8) +
                           (np.int32(lineData_u8[2, :, :] & np.uint8(31)) << 16))

            # Handle negative values
            iNeg = np.where(lineData_s32 >= minNeg)
            lineData_s32[iNeg] -= 2 * minNeg

            # Extract header bits
            lineHeader_u8 = (lineData_u8[2, :, :] & 224) >> 6

            # Store processed data
            flatData = lineData_s32.ravel(order='F')
            flatHeader = lineHeader_u8.ravel(order='F')

            lineData[:flatData.size, m] = flatData
            lineHeader[:flatHeader.size, m] = flatHeader

        return lineData, lineHeader
    
    ###################################################################################
    def _clean_duplicate_lines(self):
        """Check for and remove duplicate lines in self.rf.lineData."""
        logging.info("Checking for duplicate lines...")
        if self.rf.headerInfo.Line_Index[249] == self.rf.headerInfo.Line_Index[250]:
            self.rf.lineData = self.rf.lineData[:, np.arange(2, self.rf.lineData.shape[1], 2)]
            logging.info("Detected even-indexed duplicate, skipping even lines")
        else:
            self.rf.lineData = self.rf.lineData[:, np.arange(1, self.rf.lineData.shape[1], 2)]
            logging.info("Detected odd-indexed duplicate, skipping odd lines")

    ###################################################################################

    def _calculate_parameters(self):
        """Calculate and set main parameters as instance variables."""
        self.txBeamperFrame = np.array(self.rf.dbParams.num2DCols).flat[0]
        self.NumSonoCTAngles = self.rf.dbParams.numOfSonoCTAngles2dActual[0]
        logging.info(f"Beam parameters - txBeamperFrame: {self.txBeamperFrame}, NumSonoCTAngles: {self.NumSonoCTAngles}")
        self.numFrame = int(np.floor(self.rf.lineData.shape[1] / (self.txBeamperFrame * self.NumSonoCTAngles)))
        self.multilinefactor = self.ML_in
        self.pt = int(np.floor((self.rf.lineData.shape[0] - self.used_os) / self.multilinefactor))
        logging.info(f"Calculated parameters - numFrame: {self.numFrame}, multilinefactor: {self.multilinefactor}, points: {self.pt}")

    ###################################################################################

    def _initialize_data_arrays(self):
        """Initialize and set data arrays as instance variables."""
        logging.info("Initializing data arrays...")
        self.rftemp_all_harm = np.zeros((self.pt, self.ML_out * self.txBeamperFrame))
        self.rftemp_all_fund = np.zeros((self.pt, self.ML_out * self.txBeamperFrame))
        self.rf_data_all_harm = np.zeros((self.numFrame, self.NumSonoCTAngles, self.pt, self.ML_out * self.txBeamperFrame))
        self.rf_data_all_fund = np.zeros((self.numFrame, self.NumSonoCTAngles, self.pt, self.ML_out * self.txBeamperFrame))
        logging.info("Array initialization completed")

    ###################################################################################

    def _fill_data_arrays(self):
        """Fill the data arrays using instance variables."""
        logging.info("Filling data arrays...")
        for k0 in range(self.numFrame):
            logging.info(f"Processing frame {k0+1}/{self.numFrame}")
            for k1 in range(self.NumSonoCTAngles):
                for k2 in range(self.txBeamperFrame):
                    bi = k0 * self.txBeamperFrame * self.NumSonoCTAngles + k1 * self.txBeamperFrame + k2
                    temp = np.transpose(
                        np.reshape(self.rf.lineData[self.used_os + np.arange(self.pt * self.multilinefactor), bi],
                                 (self.multilinefactor, self.pt), order='F')
                    )
                    self.rftemp_all_harm[:, np.arange(self.ML_out) + (k2 * self.ML_out)] = temp[:, [0, 2]]
                    self.rftemp_all_fund[:, np.arange(self.ML_out) + (k2 * self.ML_out)] = temp[:, [9, 11]]
                self.rf_data_all_harm[k0, k1] = self.rftemp_all_harm
                self.rf_data_all_fund[k0, k1] = self.rftemp_all_fund
        logging.info("Data array filling completed")

    ###################################################################################

    def _prepare_save_contents(self):
        """Prepare the contents dictionary for saving, using instance variables."""
        logging.info("Preparing data for saving...")
        self.contents = {
            'echoData': self.rf.echoData[0],
            'lineData': self.rf.lineData,
            'lineHeader': self.rf.lineHeader,
            'headerInfo': self.rf.headerInfo,
            'dbParams': self.rf.dbParams,
            'rf_data_all_fund': self.rf_data_all_fund,
            'rf_data_all_harm': self.rf_data_all_harm,
            'NumFrame': self.numFrame,
            'NumSonoCTAngles': self.NumSonoCTAngles,
            'pt': self.pt,
            'multilinefactor': self.multilinefactor,
        }
        if len(self.rf.echoData) > 1 and len(self.rf.echoData[1]):
            self.contents['echoData1'] = self.rf.echoData[1]
            logging.info("Added echoData1 to contents")
        if len(self.rf.echoData) > 2 and len(self.rf.echoData[2]):
            self.contents['echoData2'] = self.rf.echoData[2]
            logging.info("Added echoData2 to contents")
        if len(self.rf.echoData) > 3 and len(self.rf.echoData[3]):
            self.contents['echoData3'] = self.rf.echoData[3]
            logging.info("Added echoData3 to contents")
        if hasattr(self.rf, 'echoMModeData'):
            self.contents['echoMModeData'] = self.rf.echoMModeData
            logging.info("Added echoMModeData to contents")
        if hasattr(self.rf, 'miscData'):
            self.contents['miscData'] = self.rf.miscData
            logging.info("Added miscData to contents")

    ###################################################################################

    def _save_contents(self):
        """Save the contents dictionary to disk using instance variables."""
        base_path = self.filepath.rsplit('.', 1)[0]
        if self.save_format in ['mat', 'both']:
            mat_path = base_path + '.mat'
            if os.path.exists(mat_path):
                logging.info(f"Removing existing file: {mat_path}")
                os.remove(mat_path)
            logging.info(f"Saving data to {mat_path}")
            savemat(mat_path, self.contents)
            logging.info(f"Successfully saved parsed data to {mat_path}")
        if self.save_format in ['npz', 'both']:
            npz_path = base_path + '.npz'
            if os.path.exists(npz_path):
                logging.info(f"Removing existing file: {npz_path}")
                os.remove(npz_path)
            logging.info(f"Saving data to {npz_path}")
            np.savez_compressed(npz_path, **self.contents)
            logging.info(f"Successfully saved parsed data to {npz_path}")
        if self.save_format in ['npy', 'both']:
            npy_folder = base_path + '_npy'
            if os.path.exists(npy_folder):
                logging.info(f"Removing existing folder: {npy_folder}")
                import shutil
                shutil.rmtree(npy_folder)
            os.makedirs(npy_folder)
            logging.info(f"Created folder for .npy files: {npy_folder}")
            for key, value in self.contents.items():
                npy_path = os.path.join(npy_folder, f"{key}.npy")
                logging.info(f"Saving {key} to {npy_path}")
                np.save(npy_path, value)
            logging.info(f"Successfully saved all arrays to {npy_folder}")

    ###################################################################################

    def find_signature(self, filepath: Path) -> list:
        """Find and validate the file signature.
        
        Reads the first 8 bytes of the file to determine if it's a valid
        Philips RF data file. The signature should match either the Voyager
        or Fusion format header pattern.
        
        Args:
            filepath (Path): Path to the RF data file
            
        Returns:
            list: First 8 bytes of the file as a list of integers
            
        Raises:
            Exception: If file cannot be read
        """
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

    def prune_data(self, lineData, lineHeader, ML_Capture):
        """Remove false gate data and align samples.
        
        This method:
        1. Finds the first valid data gate
        2. Aligns data to multiline capture boundaries
        3. Removes invalid data at the end
        
        Args:
            lineData (np.ndarray): RF line data
            lineHeader (np.ndarray): Header information for each line
            ML_Capture (float): Multiline capture factor
            
        Returns:
            np.ndarray: Pruned and aligned data
        """
        logging.info("Starting pruneData function.")
        
        # Get dimensions
        numSamples = lineData.shape[0]
        referenceLine = int(np.ceil(lineData.shape[1] * 0.2)) - 1
        startPoint = int(np.ceil(numSamples * 0.015)) - 1

        # Find first valid gate
        indicesFound = np.where(lineHeader[startPoint:numSamples + 1, referenceLine] == 3)
        if not len(indicesFound[0]):
            iFirstSample = 1
            logging.warning("No valid gate found, starting from sample 1.")
        else:
            iFirstSample = indicesFound[0][0] + startPoint

        # Align to multiline capture boundaries
        alignment = np.arange(0, numSamples, np.double(ML_Capture))
        diff = alignment - iFirstSample
        iFirstSample = int(alignment[np.where(diff >= 0)[0][0]])

        # Extract valid data
        prunedData = lineData[iFirstSample:numSamples + 1, :]
        lineHeader = lineHeader[iFirstSample:numSamples + 1, :]

        # Find last valid sample
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

        # Return pruned data
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


# Main
###################################################################################

if __name__ == "__main__":
    # Parse the sample file
    path_linux = "/home/omid/job/David/sample.rf"
    path_windows = r"D:\Omid\0_samples\Philips\David\sample.rf"
    
    # get operating system
    if os.name == 'nt':
        parser = PhilipsRFParser(path_windows,  save_format='npy')
    else:
        parser = PhilipsRFParser(path_linux,  save_format='npy')

###################################################################################



"""
2025-05-12 14:06:08 - INFO - [__run] - Starting philipsRfParser for file: /home/omid/job/David/sample.rf
2025-05-12 14:06:08 - INFO - [_parse_rf_file] - Parsing RF file...
2025-05-12 14:06:08 - INFO - [_parse_rf] - Starting parseRF for file: /home/omid/job/David/sample.rf
2025-05-12 14:06:08 - INFO - [_parse_rf] - Initialized Rfdata structure
2025-05-12 14:06:08 - INFO - [_analyze_header] - Header information found.
2025-05-12 14:06:08 - INFO - [_analyze_header] - Parsing Fusion RF capture file...
2025-05-12 14:06:08 - INFO - [_analyze_header] - Parsing file header with endianness: little
2025-05-12 14:06:08 - INFO - [_parse_file_header] - Starting parseFileHeader.
2025-05-12 14:06:08 - INFO - [_parse_file_header] - File Version: 3
2025-05-12 14:06:08 - INFO - [_parse_file_header] - Header Size: 892 bytes
2025-05-12 14:06:08 - INFO - [_analyze_header] - Total header size: 920
2025-05-12 14:06:08 - INFO - [_parse_rf] - File analysis results - isVoyager: False, hasFileHeader: True, totalHeaderSize: 920
2025-05-12 14:06:08 - INFO - [_align_offsets] - Remaining size: 430964736
2025-05-12 14:06:08 - INFO - [_parse_rf] - Aligned read parameters - offset: 0, size: 430964704
2025-05-12 14:06:08 - INFO - [_parse_rf] - Loading raw data from file...
2025-05-12 14:06:09 - INFO - [call_get_part_a] - Calling getPartA with numClumps=13467647, filename='/home/omid/job/David/sample.rf', offset=920
2025-05-12 14:06:37 - INFO - [call_get_part_b] - Calling getPartB with numClumps=13467647, filename='/home/omid/job/David/sample.rf', offset=920
2025-05-12 14:06:50 - INFO - [_parse_rf] - Loaded raw data of size: (13, 13467647)
2025-05-12 14:06:50 - INFO - [_parse_rf] - Parsing metadata headers...
2025-05-12 14:06:50 - INFO - [parse_header_f] - Starting parseHeaderF function.
2025-05-12 14:06:50 - INFO - [parse_header_f] - Found 2342 headers using pattern
2025-05-12 14:06:50 - INFO - [parse_header_f] - Processing 2342 headers
2025-05-12 14:06:50 - INFO - [parse_header_f] - Successfully parsed 2342 headers
2025-05-12 14:06:50 - INFO - [_parse_rf] - Header parsing completed
2025-05-12 14:06:50 - INFO - [_parse_rf] - Parsing RF signal data...
2025-05-12 14:06:50 - INFO - [_parse_rf_data] - Tap Point: 4
2025-05-12 14:06:50 - INFO - [parse_data_f] - Starting parseDataF function.
2025-05-12 14:06:50 - INFO - [parse_data_f] - Found 2342 data sections
2025-05-12 14:06:50 - INFO - [parse_data_f] - Initialized arrays with shape: (11491, 2342)
2025-05-12 14:06:51 - INFO - [parse_data_f] - Processed data shape: lineData (11491, 2342), lineHeader (11491, 2342)
2025-05-12 14:06:51 - INFO - [_parse_rf] - Signal data parsed - lineData shape: (11491, 2342), lineHeader shape: (11491, 2342)
"""