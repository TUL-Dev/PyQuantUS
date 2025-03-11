# Standard Library Imports
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Tuple

# Third-Party Library Imports
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from tqdm import tqdm

# Local Module Imports
from pyquantus.parse.objects import DataOutputStruct, InfoStruct
from pyquantus.parse.transforms import scanConvert

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# info structures
###################################################################################
class ClariusInfo(DataOutputStruct, InfoStruct):
    
    def __init__(self):
        
        super().__init__()
        
        self.numLines: int
        self.samplesPerLine: int
        self.sampleSize: int # bytes
        
        # data from raw file
        self.raw_hdr: dict
        self.raw_timestamps: np.ndarray
        self.raw_data: np.ndarray 
        
        # outputs
        self.img_data: DataOutputStruct 
        self.img_info: ClariusInfo 
        self.scan_converted: bool        
        
###################################################################################





###################################################################################
class Clarius_yml_parser():
        
    def __init__(self, yml_path):
        
        self.yml_path: str = yml_path
        self.valid_software_versions: list[str] = ['12.0.1-673']
        
        # rf.yml
        self.software_version: str
        self.iso_time_date: str
        self.probe_version: str
        self.probe_elements: int
        self.probe_pitch: float
        self.probe_radius: float
        self.frames: int
        self.frame_rate: str
        self.transmit_frequency: str
        self.imaging_depth: str
        self.focal_depth: str
        self.auto_gain: str
        self.mla: str
        self.tgc: dict
        self.size_samples_per_line: int
        self.size_number_of_lines: int
        self.size_sample_size: str  
        self.type: str
        self.compression: str
        self.sampling_rate: float
        self.delay_samples: int
        self.lines: dict
        self.focus_depth: str
        self.focus_tx_frequency: str
        self.focus_receive_settings: dict 
        
        # env.tgc.yml
        self.frames: int
        self.timestamp: dict
        
        self.__load_rf_yml()
        self.__load_tgc_yml()
        
    ###################################################################################
        
    def __load_rf_yml(self):
        
        try:
            with open(self.yml_path, 'r') as file:
                data = yaml.safe_load(file)

            # Mapping YAML fields to class attributes
            self.software_version = data.get("software version", None)
            self.iso_time_date = data.get("iso time/date", None)
            probe = data.get("probe", {})
            self.probe_version = probe.get("version", None)
            self.probe_elements = probe.get("elements", None)
            self.probe_pitch = probe.get("pitch", None)
            self.probe_radius = probe.get("radius", None)
            self.frames = data.get("frames", None)
            self.frame_rate = data.get("frame rate", None)
            self.transmit_frequency = data.get("transmit frequency", None)
            self.imaging_depth = data.get("imaging depth", None)
            self.focal_depth = data.get("focal depth", None)
            self.auto_gain = data.get("auto gain", None)
            self.mla = data.get("mla", None)
            self.tgc = data.get("tgc", {})
            size = data.get("size", {})
            self.size_samples_per_line = size.get("samples per line", None)
            self.size_number_of_lines = size.get("number of lines", None)
            self.size_sample_size = size.get("sample size", None)
            self.type = data.get("type", None)
            self.compression = data.get("compression", None)
            self.sampling_rate = data.get("sampling rate", None)
            self.delay_samples = data.get("delay samples", None)
            self.lines = data.get("lines", {})
            focus_data = data.get("focus", {})
            self.focus_depth = focus_data.get("depth", None)
            self.focus_tx_frequency = focus_data.get("tx frequency", None)
            self.focus_receive = focus_data.get("receive", None)

        except Exception as e:
            print(f"Error loading YAML file: {e}")        
            
    ###################################################################################
    
    def __load_tgc_yml(self):
        pass

###################################################################################





# tar file unpacker    
###################################################################################  
class Clarius_tar_unpacker():
    """
    A class for extracting and processing `.tar` archives containing `.lzo` and `.raw` files.
    
    Attributes:
        tar_files_path (str): The path to the directory containing `.tar` files.
        extraction_mode (str): Extraction mode - either "single" or "multiple".
        lzo_exe_file_path (str): Path to the LZO executable for decompression (Windows only).
    """
    ###################################################################################
    
    def __init__(self, path: str, extraction_mode: str) -> None:  
        """
        Initializes the ClariusTarUnpacker class and starts the extraction process.
        
        Args:
            tar_files_path (str): The directory containing `.tar` files.
            extraction_mode (str): Mode of extraction - "single" or "multiple".
        
        Raises:
            ValueError: If `extraction_mode` is not "single" or "multiple".
        """
        
        self.path = path
        self.extraction_mode = extraction_mode
        self.lzo_exe_file_path = 'pyquantus/exe/lzop.exe'
        
        # single tar extraction attibutes
        self.single_tar_extraction: bool = None
        self.tar_path: str = None
        
        if self.extraction_mode == "single_sample":
            """Extracts data from a single sample containing multiple tar files.
            The provided path should point to a directory containing multiple tar files. 
            Each tar file within this directory will be processed sequentially, 
            extracting its contents into the appropriate output location."""
            if self.check_input_path():
                self.__run_single_sample_extraction()

        elif self.extraction_mode == "multiple_samples":
            """Processes multiple samples, where each sample is a separate directory 
            that potentially contains multiple tar files. The given path should be a 
            directory containing multiple subdirectories, each representing an individual 
            sample. Each subdirectory is processed independently, extracting the tar files 
            within it."""
            if self.check_input_path():
                self.__run_multiple_samples_extraction()

        elif self.extraction_mode == "single_tar":
            """Extracts data from a single tar file.
            The provided path should point directly to a single tar file. The file 
            will be extracted to a designated output directory, maintaining its internal 
            structure. This mode is useful when processing a standalone tar file rather 
            than multiple files in a batch."""
            if self.check_input_path():
                self.__run_single_tar_extraction()

        else:
            """Handles invalid extraction modes by raising an error."""
            raise ValueError(f"Invalid mode: {self.extraction_mode}")
        
    ###################################################################################
        
    def __repr__(self):
        return ""
    
    ###################################################################################
    
    def __run_single_sample_extraction(self):
        """Runs the extraction process for a single directory."""
        self.delete_extracted_folders()
        self.extract_tar_files()
        self.set_path_of_extracted_folders()
        self.set_path_of_lzo_files_inside_extracted_folders()
        self.read_lzo_files()
        self.set_path_of_raw_files_inside_extracted_folders()
        self.delete_hidden_files_in_extracted_folders()

    ###################################################################################

    def __run_multiple_samples_extraction(self):
        """Extracts data from all directories inside `self.path`."""
        try:
            # Retrieve all subdirectory paths
            folder_paths = [
                os.path.join(self.path, folder)
                for folder in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, folder))
            ]

            # Process each folder for data extraction
            for folder_path in folder_paths:
                self.path = folder_path  # Update path before extraction
                self.__run_single_sample_extraction()

        except Exception as e:
            logging.error(f"An error occurred while extracting data: {e}")

    ###################################################################################
    
    def __run_single_tar_extraction(self):
        """Handles extraction when a single tar file is provided."""
        self.single_tar_extraction = True
        self.tar_path = self.path
        self.path = self.get_folder_path_from_file_path(self.tar_path)
        self.__run_single_sample_extraction()
            
    ###################################################################################
           
    def delete_extracted_folders(self):
        """Deletes all extracted folders in the specified directory."""
        extracted_folders = [
            os.path.join(self.path, item)
            for item in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, item)) and "extracted" in item
        ]

        for folder in extracted_folders:
            try:
                shutil.rmtree(folder)
                logging.info(f"Deleted folder: {folder}")
            except OSError as e:
                logging.error(f"Error deleting folder {folder}: {e}")
                    
    ###################################################################################
        
    def extract_tar_files(self):
        """
        Extracts all non-hidden tar files in the specified sample folder.
        If `self.single_tar_extraction` is True, `self.path` should be a `.tar` file.
        """

        if not self.single_tar_extraction:
            # Ensure the given path is a directory
            if not os.path.isdir(self.path):
                logging.error(f"Path '{self.path}' is not a directory.")
                return

            for item_name in os.listdir(self.path):
                item_path = os.path.join(self.path, item_name)

                # Ignore hidden files
                if item_name.startswith('.'):
                    continue
                
                # Check if the item is a tar archive
                if os.path.isfile(item_path) and item_name.endswith('.tar') and tarfile.is_tarfile(item_path):
                    file_name = os.path.splitext(item_name)[0]
                    extracted_folder = os.path.join(self.path, f"{file_name}_extracted")
                    os.makedirs(extracted_folder, exist_ok=True)

                    try:
                        with tarfile.open(item_path, 'r') as tar:
                            tar.extractall(path=extracted_folder)
                            logging.info(f"Extracted '{item_name}' into '{extracted_folder}'")
                    except (tarfile.TarError, OSError) as e:
                        logging.error(f"Error extracting '{item_name}': {e}")

        elif self.single_tar_extraction:
            # Handle single tar extraction
            if os.path.isfile(self.tar_path) and self.tar_path.endswith('.tar') and tarfile.is_tarfile(self.tar_path):
                file_name = os.path.splitext(os.path.basename(self.tar_path))[0]
                extracted_folder = os.path.join(os.path.dirname(self.tar_path), f"{file_name}_extracted")
                os.makedirs(extracted_folder, exist_ok=True)

                try:
                    with tarfile.open(self.tar_path, 'r') as tar:
                        tar.extractall(path=extracted_folder)
                        logging.info(f"Extracted '{self.tar_path}' into '{extracted_folder}'")
                except (tarfile.TarError, OSError) as e:
                    logging.error(f"Error extracting '{self.tar_path}': {e}")
            else:
                logging.error(f"Invalid tar file: '{self.tar_path}'")

    ###################################################################################
    
    def set_path_of_extracted_folders(self):
        """Finds and stores paths of extracted folders inside `self.path`."""
        logging.info("Searching for extracted folders...")

        # Find all directories containing 'extracted' in their name
        self.extracted_folders_path_list = [
            os.path.join(self.path, item)
            for item in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, item)) and "extracted" in item
        ]

        # Log each extracted folder found
        for folder in self.extracted_folders_path_list:
            logging.info(f"Found extracted folder: {folder}")

        # Log summary
        logging.info(f"Total extracted folders found: {len(self.extracted_folders_path_list)}")
    
    ###################################################################################

    def set_path_of_lzo_files_inside_extracted_folders(self):
        """Finds and stores paths of .lzo files inside extracted folders."""
        logging.info("Starting to search for LZO files inside extracted folders...")

        # Ensure extracted folders list is available
        if not self.extracted_folders_path_list:
            logging.warning("No extracted folders found. Please check the extracted folders path list.")
            return

        # Now search for .lzo files in the extracted folders
        lzo_files = []
        for folder in self.extracted_folders_path_list:
            for root, dirs, files in os.walk(folder):  # Walk through each extracted folder
                for file in files:
                    if file.endswith('.lzo'):
                        lzo_file_path = os.path.join(root, file)
                        lzo_files.append(lzo_file_path)
                        logging.info(f"Found LZO file: {lzo_file_path}")

        self.lzo_files_path_list = lzo_files  # Store the paths in the class attribute

        # Log the number of LZO files found
        logging.info(f"Total LZO files found: {len(lzo_files)}")
            
    ###################################################################################

    def read_lzo_files(self):
        """
        Detects the operating system and decompresses `.lzo` files accordingly.

        **Workflow:**
        1. Determines whether the system is Windows or macOS.
        2. Logs the detected operating system.
        3. If running on **Windows**:
            - Constructs the path to the LZO executable.
            - Checks if the executable exists.
            - Iterates through the `.lzo` files and decompresses them using `lzop.exe`.
        4. If running on **macOS**:
            - Attempts to decompress `.lzo` files using the `lzop` command.
            - If `lzop` is missing, checks for Homebrew.
            - Installs `lzop` via Homebrew if necessary.
            - Decompresses `.lzo` files after ensuring `lzop` is installed.
        5. Logs successes and failures, handling potential errors like:
            - `FileNotFoundError`
            - `PermissionError`
            - `subprocess.CalledProcessError`
            - Other unexpected exceptions.

        **Returns:**
        - Logs the status of each decompression attempt.
        - Exits the program if `lzop` is missing on macOS and cannot be installed.
        """
        # Set self.os based on the platform
        os_name = platform.system().lower()           
        
        if 'windows' in os_name:
            self.os = "windows"
        elif 'darwin' in os_name:
            self.os = "mac"
        elif 'linux' in os_name:
            self.os = "linux"
        else:
            self.os = "unknown"
            
        logging.info(f'Detected operating system: {self.os}')
               
        if self.os == "windows":
            # Get the path of the current working directory
            working_space_path = os.getcwd()
            
            # Construct the full path to the LZO executable, adding the .exe extension
            path_of_lzo_exe_file = os.path.join(working_space_path, self.lzo_exe_file_path)

            # Log the path being checked
            logging.info(f'Checking path for LZO executable: {path_of_lzo_exe_file}')

            # Check if the executable exists
            if not os.path.isfile(path_of_lzo_exe_file):
                logging.error(f'LZO executable not found: {path_of_lzo_exe_file}')
                return

            for lzo_file_path in self.lzo_files_path_list:
                logging.info(f'Starting decompression for: {lzo_file_path}')
                try:
                    # Run the lzop command to decompress the LZO file
                    subprocess.run([path_of_lzo_exe_file, '-d', lzo_file_path], check=True)
                    logging.info(f'Successfully decompressed: {lzo_file_path}')
                except subprocess.CalledProcessError as e:
                    logging.error(f'Error decompressing {lzo_file_path}: {e}')
                except PermissionError as e:
                    logging.error(f'Permission denied for {lzo_file_path}: {e}')
                except Exception as e:
                    logging.error(f'Unexpected error occurred with {lzo_file_path}: {e}')
                    
        elif self.os == "mac":
            for lzo_file_path in self.lzo_files_path_list:
                logging.info(f'Starting decompression for: {lzo_file_path}')
                try:
                    # Run the lzop command to decompress the LZO file
                    subprocess.run(['lzop', '-d', lzo_file_path], check=True)
                    logging.info(f'Successfully decompressed: {lzo_file_path}')
                except FileNotFoundError:
                    # check if homebrew is installed
                    brew_path = shutil.which("brew")
                    # if homebrew is not installed, tell the user to install it manually and exit 
                    if brew_path is None:
                        logging.error("Homebrew is required to install lzop. A description how to install Homebrew can be found in the README.md file.")
                        sys.exit()
                   
                    # install lzop using homebrew
                    subprocess.run(['arch', '-arm64', 'brew', 'install', 'lzop'], check=True)
                    logging.info("Successfully installed lzop using Homebrew.")
                    # Run the lzop command to decompress the LZO file
                    subprocess.run(['lzop', '-d', lzo_file_path], check=True)
                    logging.info(f'Successfully decompressed: {lzo_file_path}')

                except subprocess.CalledProcessError as e:
                    logging.error(f'Error decompressing {lzo_file_path}: {e}')
                except PermissionError as e:
                    logging.error(f'Permission denied for {lzo_file_path}: {e}')
                except Exception as e:
                    logging.error(f'Unexpected error occurred with {lzo_file_path}: {e}')            
                    logging.error(f'Unexpected error occurred with {lzo_file_path}: {e}')    
                    
        elif self.os == "linux":
            logging.warning("LZO decompression on Linux has not been fully developed yet.")

    ###################################################################################

    def set_path_of_raw_files_inside_extracted_folders(self):
        """Searches for .raw files inside extracted folders and stores their paths."""
        logging.info("Starting to search for RAW files inside extracted folders...")

        # Ensure extracted folders list is available
        if not self.extracted_folders_path_list:
            logging.warning("No extracted folders found. Please check the extracted folders path list.")
            return

        # Now search for .raw files in the extracted folders
        raw_files = []
        for folder in self.extracted_folders_path_list:
            for root, dirs, files in os.walk(folder):  # Walk through each extracted folder
                for file in files:
                    if file.endswith('.raw'):
                        raw_file_path = os.path.join(root, file)
                        raw_files.append(raw_file_path)
                        logging.info(f"Found RAW file: {raw_file_path}")

        self.raw_files_path_list = raw_files  # Store the paths in the class attribute

        # Log the number of RAW files found
        logging.info(f"Total RAW files found: {len(raw_files)}")

    ###################################################################################
    
    def delete_hidden_files_in_extracted_folders(self):
        """Deletes hidden files (starting with a dot) in extracted folders."""
        # Iterate through each extracted folder path
        for folder_path in self.extracted_folders_path_list:
            try:
                # List all files in the folder
                for filename in os.listdir(folder_path):
                    # Check if the file is hidden (starts with a dot)
                    if filename.startswith('.'):
                        file_path = os.path.join(folder_path, filename)
                        os.remove(file_path)  # Delete the hidden file
                        logging.info(f"Deleted hidden file: {file_path}")
            except Exception as e:
                logging.error(f"Error while deleting hidden files in {folder_path}: {e}")
          
    ###################################################################################
    @staticmethod
    def get_folder_path_from_file_path(file_path: str) -> str:
        """Returns the absolute directory path of the given file."""
        return os.path.dirname(os.path.abspath(file_path))
    
    ###################################################################################
    
    def check_input_path(self):
        """
        Validates the input path based on the specified extraction mode.

        This method checks whether the provided path exists and conforms to 
        the expected format for different extraction modes:

        - "single_sample": The path must be a directory containing at least 
        one `.tar` file.
        - "multiple_samples": The path must be a directory containing at least 
        one subdirectory.
        - "single_tar": The path must be a valid `.tar` file.

        Returns:
            bool: True if the path is valid for the specified extraction mode, 
                otherwise False with a warning message.
        """
        if not os.path.exists(self.path):
            print(f"Warning: The path '{self.path}' does not exist.")
            return False

        if self.extraction_mode == "single_sample":
            if not os.path.isdir(self.path):
                print(f"Warning: The path '{self.path}' is not a directory.")
                return False
            tar_files = [f for f in os.listdir(self.path) if f.endswith(".tar")]
            if not tar_files:
                print(f"Warning: No .tar files found in '{self.path}'.")
                return False

        elif self.extraction_mode == "multiple_samples":
            if not os.path.isdir(self.path):
                print(f"Warning: The path '{self.path}' is not a directory.")
                return False
            subfolders = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
            if not subfolders:
                print(f"Warning: No subdirectories found in '{self.path}'.")
                return False

        elif self.extraction_mode == "single_tar":
            if not os.path.isfile(self.path) or not self.path.endswith(".tar"):
                print(f"Warning: The path '{self.path}' is not a valid .tar file.")
                return False
            if not tarfile.is_tarfile(self.path):
                print(f"Warning: The file '{self.path}' is not a valid tar archive.")
                return False

        else:
            print(f"Warning: Unknown extraction mode '{self.extraction_mode}'.")
            return False

        return True
    
    ###################################################################################

###################################################################################



# parser
###################################################################################  
class Clarius_raw_parser(ClariusInfo):

    ###################################################################################
    
    def __init__(self, 
                 main_raw_path: str,
                 main_tgc_yml_path: str | None,
                 main_rf_yml_path: str,
                 isPhantom: bool):
    
        # inputs
        self.main_raw_path = main_raw_path
        self.main_tgc_yml_path = main_tgc_yml_path
        self.main_rf_yml_path = main_rf_yml_path
        self.isPhantom = isPhantom
        
        # predefined versions
        self.supporting_versions_list: list = ["6.0.3"]
        
        self.__run()
        
    
        
    ###################################################################################
    
    def __run(self):
        
        self.read_raw_files()
        
    ###################################################################################
    
    def read_raw_files(self, 
                        raw_file_path):
        """
        Reads a raw binary file, extracts header information, timestamps, and data,
        and returns them as a dictionary and NumPy arrays.

        Args:
            raw_file_path (str): Path to the raw binary file.

        Returns:
            tuple: (hdr (dict), timestamps (np.ndarray), data (np.ndarray))
            or (None, None, None) in case of an error.
        """
        logging.info(f'Reading raw file: {raw_file_path}')

        # Define header fields
        hdr_info = ('id', 'frames', 'lines', 'samples', 'samplesize')
        hdr = {}

        try:
            with open(raw_file_path, 'rb') as raw_bytes:
                # Read header (4 bytes each field)
                for info in hdr_info:
                    hdr[info] = int.from_bytes(raw_bytes.read(4), byteorder='little')

                # Validate header values
                if any(value <= 0 for value in hdr.values()):
                    logging.error(f'Invalid header values: {hdr}')
                    return None, None, None

                # Initialize arrays
                timestamps = np.zeros(hdr['frames'], dtype=np.int64)
                sz = hdr['lines'] * hdr['samples'] * hdr['samplesize']
                data = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype=np.int16)

                # Read timestamps and data for each frame
                for frame in range(hdr['frames']):
                    # Read timestamp (8 bytes per frame)
                    timestamps[frame] = int.from_bytes(raw_bytes.read(8), byteorder='little')

                    # Read frame data and ensure correct number of bytes
                    frame_data = raw_bytes.read(sz)
                    if len(frame_data) != sz:
                        logging.error(f'Unexpected frame size at frame {frame}: Expected {sz} bytes, got {len(frame_data)}')
                        return None, None, None

                    data[:, :, frame] = np.frombuffer(frame_data, dtype=np.int16).reshape((hdr['lines'], hdr['samples']))

            logging.info(f'Loaded {data.shape[2]} raw frames of size {data.shape[0]} x {data.shape[1]} (lines x samples)')

            self.raw_hdr, self.raw_timestamps, self.raw_data = hdr, timestamps, data

        except Exception as e:
            logging.error(f'Error reading file {raw_file_path}: {e}')
            return None, None, None

    ###################################################################################
    
###################################################################################




# functions
###################################################################################

def read_tgc_file(file_timestamp: str, rf_timestamps: np.ndarray) -> list | None:
    """Read TGC file and extract TGC data for inputted file.

    Args:
        file_timestamp (str): Timestamp of the inputted RF file.
        rf_timestamps (np.ndarray): Given RF timestamps.

    Returns:
        list | None: Extracted TGC data corresponding to the RF timestamps, or None if the TGC file is not found.
    """
    tgc_file_name_dottgc = file_timestamp + "_env.tgc"
    tgc_file_name_dotyml = file_timestamp + "_env.tgc.yml"

    if os.path.isfile(tgc_file_name_dottgc):
        tgc_file_name = tgc_file_name_dottgc
    elif os.path.isfile(tgc_file_name_dotyml):
        tgc_file_name = tgc_file_name_dotyml
    else:
        return None

    with open(tgc_file_name, "r") as file:
        data_str = file.read()

    frames_data = data_str.split("timestamp:")[1:]
    frames_data = [
        frame
        if "{" in frame
        else frame + "  - { 0.00mm, 15.00dB }\n  - { 120.00mm, 35.00dB }"
        for frame in frames_data
    ]
    frames_dict = {
        timestamp: frame
        for frame in frames_data
        for timestamp in rf_timestamps
        if str(timestamp) in frame
    }
    filtered_frames_data = [
        frames_dict.get(timestamp)
        for timestamp in rf_timestamps
        if frames_dict.get(timestamp) is not None
    ]

    return filtered_frames_data

###################################################################################

def clean_and_convert(value):
    """Clean and convert a string value to a float."""
    clean_value = ''.join([char for char in value if char.isdigit() or char in ['.', '-']])
    return float(clean_value)

###################################################################################

def extract_tgc_data_from_line(line):
    """Extract TGC data from a line."""
    tgc_pattern = r'\{([^}]+)\}'
    return re.findall(tgc_pattern, line)

###################################################################################

def read_tgc_file_v2(tgc_path, rf_timestamps):
    """Read TGC file and extract TGC data for inputted file.
    
    Args:
        tgc_path (str): Path to the TGC file.
        rf_timestamps (np.ndarray): Given RF timestamps.
        
    Returns:
        list | None: Extracted TGC data corresponding to the RF timestamps, or None if no corresponding data is found.
    """
    with open(tgc_path, 'r') as file:
        data_str = file.read()
    
    frames_data = data_str.split('timestamp:')[1:]
    frames_data = [frame if "{" in frame else frame + "  - { 0.00mm, 15.00dB }\n  - { 120.00mm, 35.00dB }" for frame in frames_data]
    frames_dict = {timestamp: frame for frame in frames_data for timestamp in rf_timestamps if str(timestamp) in frame}
    missing_timestamps = [ts for ts in rf_timestamps if ts not in frames_dict]
    if len(missing_timestamps) >= 2:
        print("The number of missing timestamps for " + tgc_path + " is: " + str(len(missing_timestamps)) + ". Skipping this scan with current criteria.")
        return None
    elif len(missing_timestamps) == 1:
        missing_ts = missing_timestamps[0]
        print("missing timestamp is: ")
        print(missing_ts)
        index = np.where(rf_timestamps == missing_ts)[0][0]
        prev_ts = rf_timestamps[index - 1]
        next_ts = rf_timestamps[index + 1]
        prev_data = frames_dict[prev_ts]
        next_data = frames_dict[next_ts]
        interpolated_data = f" {missing_ts} "
        prev_tgc_entries = extract_tgc_data_from_line(prev_data)
        next_tgc_entries = extract_tgc_data_from_line(next_data)
        for prev_val, next_val in zip(prev_tgc_entries, next_tgc_entries):
            prev_mm_str, prev_dB_str = prev_val.split(",")
            next_mm_str, next_dB_str = next_val.split(",")
            prev_dB = clean_and_convert(prev_dB_str)
            next_dB = clean_and_convert(next_dB_str)
            prev_mm = clean_and_convert(prev_mm_str)
            next_mm = clean_and_convert(next_mm_str)
            if abs(prev_dB - next_dB) <= 4:
                interpolated_dB = (prev_dB + next_dB) / 2
            else:
                print("Difference in dB values too large for interpolation. Skipping this Scan with current criteria.")
                return None
            interpolated_data += f"{{ {prev_mm}mm, {interpolated_dB:.2f}dB }}"
        print("prev data for " + str(prev_ts) + " is: ")
        print(prev_data)
        print("interpolated data for " + str(missing_ts) + " is: ")
        print(interpolated_data)
        print("next data for " + str(next_ts) + " is: ")
        print(next_data)
        frames_dict[missing_ts] = interpolated_data
    filtered_frames_data = [frames_dict.get(timestamp) for timestamp in rf_timestamps if frames_dict.get(timestamp) is not None]
    

    return filtered_frames_data

###################################################################################

def generate_default_tgc_matrix(num_frames: int, info: ClariusInfo) -> np.ndarray:
    """Generate a default TGC matrix for the inputted number of frames and Clarius file metadata.
    
    Args:
        num_frames (int): Number of frames.
        info (ClariusInfo): Clarius file metadata.
        
    Returns:
        numpy.ndarray: Default TGC matrix.
    """
    # image_depth_mm = 150
    # num_samples = 2928
    image_depth_mm = info.endDepth1
    num_samples = info.samplesPerLine
    depths_mm = np.linspace(0, image_depth_mm, num_samples)
    default_mm_values = [0.00, 120.00]
    default_dB_values = [15.00, 35.00]

    default_interpolation_func = interp1d(
        default_mm_values,
        default_dB_values,
        bounds_error=False,
        fill_value=(default_dB_values[0], default_dB_values[-1]),
    )
    default_tgc_matrix = default_interpolation_func(depths_mm)[None, :]
    default_tgc_matrix = np.repeat(default_tgc_matrix, num_frames, axis=0)

    default_tgc_matrix_transpose = default_tgc_matrix.T
    linear_default_tgc_matrix_transpose = 10 ** (default_tgc_matrix_transpose / 20)
    linear_default_tgc_matrix_transpose = linear_default_tgc_matrix_transpose[None, ...]
    linear_default_tgc_matrix_transpose = np.repeat(
        linear_default_tgc_matrix_transpose, info.numLines, axis=0
    )

    print(
        "A default TGC matrix of size {} is generated.".format(
            linear_default_tgc_matrix_transpose.shape
        )
    )
    #     for depth, tgc_value in zip(depths_mm, default_tgc_matrix[0]):
    #         print(f"Depth: {depth:.2f}mm, TGC: {tgc_value:.2f}dB")

    return linear_default_tgc_matrix_transpose

###################################################################################

def generate_tgc_matrix(file_timestamp: str, tgc_path: str | None, rf_timestamps: np.ndarray, num_frames: int, 
                        info: InfoStruct, isPhantom: bool) -> np.ndarray:
    """Generate a TGC matrix for the inputted file timestamp, TGC path, RF timestamps, number of frames, and Clarius file metadata.

    Args:
        file_timestamp (str): Timestamp of the inputted RF file.
        tgc_path (str): Path to the TGC file.
        rf_timestamps (np.ndarray): Given RF timestamps.
        num_frames (int): Number of frames.
        info (InfoStruct): Clarius file metadata.
        isPhantom (bool): Indicates if the data is phantom (True) or patient data (False).

    Returns:
        np.ndarray: TGC matrix.
    """
    # image_depth_mm = 150
    # num_samples = 2928
    image_depth_mm = info.endDepth1
    num_samples = info.samplesPerLine
    if tgc_path is not None:
        if isPhantom:
            tgc_data = read_tgc_file(file_timestamp, rf_timestamps)
        else:
            tgc_data = read_tgc_file_v2(tgc_path, rf_timestamps)
    else:
        tgc_data = None
        

    if tgc_data == None:
        return generate_default_tgc_matrix(num_frames, info)

    tgc_matrix = np.zeros((len(tgc_data), num_samples))
    depths_mm = np.linspace(0, image_depth_mm, num_samples)

    for i, frame in enumerate(tgc_data):
        mm_values = [float(x) for x in re.findall(r"{ (.*?)mm,", frame)]
        dB_values = [float(x) for x in re.findall(r", (.*?)dB }", frame)]
        fill_value = (dB_values[0], dB_values[-1])
        interpolation_func = interp1d(
            mm_values, dB_values, bounds_error=False, fill_value=fill_value
        )
        tgc_matrix[i, :] = interpolation_func(depths_mm)

    tgc_matrix_transpose = tgc_matrix.T
    linear_tgc_matrix_transpose = 10 ** (tgc_matrix_transpose / 20)
    linear_tgc_matrix_transpose = linear_tgc_matrix_transpose[None, ...]
    linear_tgc_matrix_transpose = np.repeat(linear_tgc_matrix_transpose, info.numLines, axis=0)

    print(
        "A TGC matrix of size {} is generated for {} timestamp ".format(
            linear_tgc_matrix_transpose.shape, file_timestamp
        )
    )
    #     for depth, tgc_value in zip(depths_mm, tgc_matrix[0]):
    #         print(f"Depth: {depth:.2f}mm, TGC: {tgc_value:.2f}dB")

    return linear_tgc_matrix_transpose

###################################################################################

def checkLengthEnvRF(rfa, rfd, rfn, env, db):
    lenEnv = env.shape[2]
    lenRf = rfa.shape[2]

    if lenEnv == lenRf:
        pass
    elif lenEnv > lenRf:
        env = env[:, :, :lenRf]
    else:
        db = db[:, :, :lenEnv]
        rfa = rfa[:, :, :lenEnv]
        rfd = rfd[:, :, :lenEnv]
        rfn = rfn[:, :, :lenEnv]

    return rfa, rfd, rfn, env, db

###################################################################################

def convert_env_to_rf_ntgc(x, linear_tgc_matrix):
    y1 =  47.3 * x + 30
    y = 10**(y1/20)-1
    y = y / linear_tgc_matrix
    return y 

###################################################################################

def readImg(filename: str, tgc_path: str | None, info_path: str, 
            version="6.0.3", isPhantom=False) -> Tuple[DataOutputStruct, ClariusInfo, bool]:
    """Read RF data contained in Clarius file
    Args:
        filename (string)): where is the Clarius file
        version (str, optional): indicates Clarius file version. Defaults to '6.0.3'. Currently not used.
        isPhantom (bool, optional): indicated if it is phantom (True) or patient data (False)

    Returns:
        Tuple: Corrected RF data processed from RF data contained in filename and as well as metadata
    """

    if version != "6.0.3":
        print("Unrecognized version")
        return []

    # read the header info
    hinfo = np.fromfile(
        filename, dtype="uint32", count=5
    )  # int32 and uint32 appear to be equivalent in memory -> a = np.int32(1); np.dtype(a).itemsize
    header = {"id": 0, "nframes": 0, "w": 0, "h": 0, "ss": 0}
    header["id"] = hinfo[0]
    header["nframes"] = hinfo[1]  # frames
    header["w"] = hinfo[2]  # lines
    header["h"] = hinfo[3]  # samples
    header["ss"] = hinfo[4]  # sampleSize

    # % ADDED BY AHMED EL KAFFAS - 22/09/2018
    frames = header["nframes"]

    id = header["id"]
    if id == 2:  # RF
        ts = np.zeros(shape=(frames,), dtype="uint64")
        data = np.zeros(shape=(header["h"], header["w"], frames))
        #  read RF data
        for f in range(frames):
            ts[f] = np.fromfile(filename, dtype="uint64", count=1)[0]
            v = np.fromfile(filename, count=header["h"] * header["w"], dtype="int16")
            data[:, :, f] = np.flip(
                v.reshape(header["h"], header["w"], order="F").astype(np.int16), axis=1
            )
            
    #######################################################################################################
    else:
        print(
            "The file does not contain RF data. Make sure RF mode is turned on while taking scans."
        )
        return []

    # # Check if the ROI is full
    # if header["w"] != 192 or header["h"] != 2928:
    #     print(
    #         "The ROI is not full. The size of RF matrix is {}*{} thus returning an empty list.".format(
    #             header["w"], header["h"]
    #         )
    #     )
    #     return []
    

    info = ClariusInfo()
    with open(info_path, 'r') as file:
        infoYml = yaml.safe_load(file)
    try:
        info.width1 = infoYml["probe"]["radius"] * 2
        scanConverted = True
    except KeyError:
        scanConverted = False
    info.endDepth1 = float(infoYml["imaging depth"][:-2]) / 1000 #m
    info.startDepth1 = info.endDepth1 / 4 #m
    info.samplingFrequency = int(infoYml["sampling rate"][:-3]) * 1e6
    info.tilt1 = 0
    info.samplesPerLine = infoYml["size"]["samples per line"]
    info.numLines = infoYml["size"]["number of lines"]
    info.sampleSize = infoYml["size"]["sample size"]
    info.centerFrequency = float(infoYml["transmit frequency"][:-3]) * 1e6

    info.minFrequency = 0
    info.maxFrequency = info.centerFrequency*2
    info.lowBandFreq = int(info.centerFrequency/2)
    info.upBandFreq = int(info.centerFrequency*1.5)
    
    info.clipFact = 0.95
    info.dynRange = 50

    data = data.astype(np.float64)
    file_timestamp = filename.split("_rf.raw")[0]
    linear_tgc_matrix = generate_tgc_matrix(file_timestamp, tgc_path, ts, header["nframes"], info, isPhantom)
    linear_tgc_matrix = np.transpose(linear_tgc_matrix, (1, 0, 2))

    if data.shape[2] != linear_tgc_matrix.shape[2]:
        print(
            "\033[31m"
            + "The timestamps for file_timestamp {} does not match between rf.raw and tgc file. Skipping this scan and returning an empty array.".format(
                file_timestamp
            )
            + "\033[0m"
        )
        return []

    rf_matrix_corrected_B = data / linear_tgc_matrix

    linear_default_tgc_matrix = generate_default_tgc_matrix(header["nframes"], info)
    linear_default_tgc_matrix = np.transpose(linear_default_tgc_matrix, (1, 0, 2))
    rf_matrix_corrected_A = rf_matrix_corrected_B * linear_default_tgc_matrix
    dB_tgc_matrix = 20*np.log10(linear_tgc_matrix)
    rf_ntgc = rf_matrix_corrected_B
    rf_dtgc = rf_matrix_corrected_A
    rf_atgc = data

    # rf_atgc, rf_dtgc, rf_ntgc, dataEnv, dB_tgc_matrix = checkLengthEnvRF(rf_atgc,rf_dtgc,rf_ntgc,dataEnv,dB_tgc_matrix)
    linear_tgc_matrix = linear_tgc_matrix[0:dB_tgc_matrix.shape[0],0:dB_tgc_matrix.shape[1],0:dB_tgc_matrix.shape[2]]
    
    bmode = np.zeros_like(rf_atgc)
    for f in range(rf_atgc.shape[2]):
        for i in range(rf_atgc.shape[1]):
            bmode[:,i, f] = 20*np.log10(abs(hilbert(rf_atgc[:,i, f])))   
            
    clippedMax = info.clipFact*np.amax(bmode)
    bmode = np.clip(bmode, clippedMax-info.dynRange, clippedMax) 
    bmode -= np.amin(bmode)
    bmode *= (255/np.amax(bmode))
    
    data = DataOutputStruct()
    
    if scanConverted:
        scBmodeStruct, hCm1, wCm1 = scanConvert(bmode[:,:,0], info.width1, info.tilt1, info.startDepth1, 
                                            info.endDepth1, desiredHeight=2000)
        scBmodes = np.array([scanConvert(bmode[:,:,i], info.width1, info.tilt1, info.startDepth1, 
                                     info.endDepth1, desiredHeight=2000)[0].scArr for i in tqdm(range(rf_atgc.shape[2]))])

        info.yResRF =  info.endDepth1*1000 / scBmodeStruct.scArr.shape[0]
        info.xResRF = info.yResRF * (scBmodeStruct.scArr.shape[0]/scBmodeStruct.scArr.shape[1]) # placeholder
        info.axialRes = hCm1*10 / scBmodeStruct.scArr.shape[0]
        info.lateralRes = wCm1*10 / scBmodeStruct.scArr.shape[1]
        info.depth = hCm1*10 #mm
        info.width = wCm1*10 #mm
        data.scBmodeStruct = scBmodeStruct
        data.scBmode = scBmodes
        
    else:
        info.yResRF = info.endDepth1*1000 / bmode.shape[0] # mm/pixel
        info.xResRF = info.yResRF * (bmode.shape[0]/bmode.shape[1]) # placeholder
        info.axialRes = info.yResRF #mm
        info.lateralRes = info.xResRF #mm
        info.depth = info.endDepth1*1000 #mm
        info.width = info.endDepth1*1000 #mm

    
    data.bMode = np.transpose(bmode, (2, 0, 1))
    data.rf = np.transpose(rf_atgc, (2, 0, 1))

    return data, info, scanConverted

###################################################################################

def clariusRfParser(imgFilename: str, imgTgcFilename: str, infoFilename: str, 
            phantomFilename: str, phantomTgcFilename: str, phantomInfoFilename: str, 
            version="6.0.3") -> Tuple[DataOutputStruct, ClariusInfo, DataOutputStruct, ClariusInfo, bool]:
    """Parse Clarius RF data and metadata from inputted files.

    Args:
        imgFilename (str): File path of the RF data.
        imgTgcFilename (str): File path of the TGC data.
        infoFilename (str): File path of the metadata.
        phantomFilename (str): File path of the phantom RF data.
        phantomTgcFilename (str): File path of the phantom TGC data.
        phantomInfoFilename (str): File path of the phantom metadata.
        version (str, optional): Defaults to "6.0.3".

    Returns:
        Tuple: Image data, image metadata, phantom data, and phantom metadata.
    """
    imgData, imgInfo, scanConverted = readImg(imgFilename, imgTgcFilename, infoFilename, version, isPhantom=False)
    refData, refInfo, scanConverted = readImg(phantomFilename, phantomTgcFilename, phantomInfoFilename, version, isPhantom=False)
    return imgData, imgInfo, refData, refInfo, scanConverted

###################################################################################
