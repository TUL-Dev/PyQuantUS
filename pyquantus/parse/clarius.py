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
import matplotlib.pyplot as plt

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
        
        # outputs
        self.img_data: DataOutputStruct 
        self.img_info: ClariusInfo 
        self.scan_converted: bool        
        
###################################################################################



# tar file unpacker    
###################################################################################  
class ClariusTarUnpacker():
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
        """
        Returns a string representation of the object.

        This method provides a developer-friendly representation of the instance,
        typically including key attributes for debugging purposes.

        Returns:
            str: A string representation of the instance.
        """

        return f"{self.__class__.__name__}"

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
            for lzo_file_path in self.lzo_files_path_list:
                logging.info(f'Starting decompression for: {lzo_file_path}')
                try:
                    # Run the lzop command to decompress the LZO file
                    subprocess.run(['lzop', '-d', lzo_file_path], check=True)
                    logging.info(f'Successfully decompressed: {lzo_file_path}')
                except FileNotFoundError as e:
                    logging.error(f"lzop must be installed to decompress LZO files. Please install lzop and try again.")
                    sys.exit()

                except subprocess.CalledProcessError as e:
                    logging.error(f'Error decompressing {lzo_file_path}: {e}')
                except PermissionError as e:
                    logging.error(f'Permission denied for {lzo_file_path}: {e}')
                except Exception as e:
                    logging.error(f'Unexpected error occurred with {lzo_file_path}: {e}')            
                    logging.error(f'Unexpected error occurred with {lzo_file_path}: {e}')    

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
            logging.warning(f"The path '{self.path}' does not exist.")
            return False

        if self.extraction_mode == "single_sample":
            if not os.path.isdir(self.path):
                logging.warning(f"The path '{self.path}' is not a directory.")
                return False
            tar_files = [f for f in os.listdir(self.path) if f.endswith(".tar")]
            if not tar_files:
                logging.warning(f"No .tar files found in '{self.path}'.")
                return False

        elif self.extraction_mode == "multiple_samples":
            if not os.path.isdir(self.path):
                logging.warning(f"The path '{self.path}' is not a directory.")
                return False
            subfolders = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
            if not subfolders:
                logging.warning(f"No subdirectories found in '{self.path}'.")
                return False

        elif self.extraction_mode == "single_tar":
            if not os.path.isfile(self.path) or not self.path.endswith(".tar"):
                logging.warning(f"The path '{self.path}' is not a valid .tar file.")
                return False
            if not tarfile.is_tarfile(self.path):
                logging.warning(f"The file '{self.path}' is not a valid tar archive.")
                return False

        else:
            logging.warning(f"Unknown extraction mode '{self.extraction_mode}'.")
            return False

        return True
    
    ###################################################################################
    
###################################################################################



# parser
###################################################################################  
class ClariusParser(ClariusInfo):

    ###################################################################################
    
    def __init__(self, extracted_sample_folder_path: str):
    
        self.path = extracted_sample_folder_path
        
        # yml files path
        self.rf_yml_path: str | None
        self.env_yml_path: str | None
        self.env_tgc_yml_path: str | None
        
        # raw files path
        self.rf_raw_path: str | None
        self.env_raw_path: str | None

        # yml objects
        self.rf_yml_obj: object
        self.env_yml_obj: object
        self.env_tgc_yml_obj: object
        
        # data from rf.raw file
        self.rf_raw_hdr: dict
        self.rf_raw_timestamps_1d: np.ndarray
        self.rf_raw_data_3d: np.ndarray 
        
        # data from env.raw file
        self.env_raw_hdr: dict
        self.env_raw_timestamps_1d: np.ndarray
        self.env_raw_data_3d: np.ndarray 
        
        # tgc
        self.default_tgc_data: dict = {}
        self.clean_tgc_data: dict = {}
        self.rf_no_tgc_raw_data_3d: np.ndarray
        
        # depth
        self.trimmed_imaging_depth_array_1d_cm:np.ndarray

        self.__run()
                
    ###################################################################################
    
    def __run(self):
        if not self.check_required_files():
            return

        # Process YAML files
        self.set_ymls_path()
        self.read_ymls()

        # Process raw files
        self.set_raws_path()
        self.read_rf_raw()
        self.read_env_raw()

        # Create no_tgc
        self.set_default_tgc_data()
        self.create_clean_no_tgc_data()
        self.create_no_tgc_raw()
        
        self.set_trimmed_imaging_depth_array()
        #self.visualize_raw()

    ###################################################################################
    
    def check_required_files(self):
        """Checks if all required files exist in the directory but ignores extra files or folders."""
        if not os.path.isdir(self.path):
            logging.error(f"The path '{self.path}' is NOT a valid folder.")
            return False

        # Define the required file suffixes after '_'
        required_files = {
            "env.raw.lzo",
            "rf.raw.lzo",
            "env.raw",
            "rf.raw",
            "env.tgc.yml",
            "env.yml",
            "rf.yml"
        }

        # Get all files in the directory
        files_in_directory = os.listdir(self.path)

        # Extract parts after the first '_'
        found_files = set()
        for file in files_in_directory:
            parts = file.split("_", 1)
            if len(parts) > 1:
                found_files.add(parts[1])  # Add the suffix part

        # Check if all required files are present
        missing_files = required_files - found_files

        if missing_files:
            logging.warning("The following required files are missing:")
            for missing in missing_files:
                logging.warning(f"- {missing}")
            return False

        logging.info("All required files are present.")
        return True

    ###################################################################################
    
    def set_ymls_path(self):
        """
        Scans the directory for files ending with `_rf.yml`, `_env.yml`, and `_env.tgc.yml`,
        then sets their paths as instance attributes.

        This method ensures that even if the files have dynamic prefixes, 
        they will be correctly identified based on their endings.

        Attributes:
            rf_yml_path (str): Full path to the file ending with `_rf.yml`, else None.
            env_yml_path (str): Full path to the file ending with `_env.yml`, else None.
            env_tgc_yml_path (str): Full path to the file ending with `_env.tgc.yml`, else None.

        Returns:
            None
        """

        # Required YAML file suffixes and corresponding attribute names
        required_suffixes = {
            "_rf.yml": "rf_yml_path",
            "_env.yml": "env_yml_path",
            "_env.tgc.yml": "env_tgc_yml_path"
        }

        logging.info(f"Scanning directory: {self.path}")

        # Get all files in the directory
        try:
            files_in_dir = os.listdir(self.path)
        except FileNotFoundError:
            logging.error(f"Directory not found: {self.path}")
            return

        # Check if any file matches the required suffix
        for suffix, attr_name in required_suffixes.items():
            matching_files = [f for f in files_in_dir if f.endswith(suffix)]
            
            if matching_files:
                # Take the first matching file (assuming there's only one per suffix)
                selected_file = matching_files[0]
                file_path = os.path.join(self.path, selected_file)
                setattr(self, attr_name, file_path)
                logging.info(f"Found {suffix}: {file_path}")
            else:
                setattr(self, attr_name, None)
                logging.warning(f"Missing file ending with {suffix}")

        logging.info("YAML file path setting completed.")
    
    ###################################################################################

    def read_ymls(self):
        
        self.rf_yml_obj = ClariusParser.YmlParser(self.rf_yml_path)
        self.env_yml_obj = ClariusParser.YmlParser(self.env_yml_path)
        self.env_tgc_yml_obj = ClariusParser.YmlParser(self.env_tgc_yml_path)
        
    ###################################################################################
    
    def set_raws_path(self):
        """
        Scans the directory for files ending with `_rf.raw` and `_env.raw`,
        then sets their paths as instance attributes.

        This method ensures that even if the files have dynamic prefixes, 
        they will be correctly identified based on their endings.

        Attributes:
            rf_raw_path (str): Full path to the file ending with `_rf.raw`, else None.
            env_raw_path (str): Full path to the file ending with `_env.raw`, else None.

        Returns:
            None
        """

        # Required RAW file suffixes and corresponding attribute names
        required_suffixes = {
            "_rf.raw": "rf_raw_path",
            "_env.raw": "env_raw_path"
        }

        logging.info(f"Scanning directory: {self.path} for RAW files")

        # Get all files in the directory
        try:
            files_in_dir = os.listdir(self.path)
        except FileNotFoundError:
            logging.error(f"Directory not found: {self.path}")
            return

        # Check if any file matches the required suffix
        for suffix, attr_name in required_suffixes.items():
            matching_files = [f for f in files_in_dir if f.endswith(suffix)]
            
            if matching_files:
                # Take the first matching file (assuming there's only one per suffix)
                selected_file = matching_files[0]
                file_path = os.path.join(self.path, selected_file)
                setattr(self, attr_name, file_path)
                logging.info(f"Found {suffix}: {file_path}")
            else:
                setattr(self, attr_name, None)
                logging.warning(f"Missing file ending with {suffix}")

        logging.info("RAW file path setting completed.")
        
    ###################################################################################
    
    def read_rf_raw(self):
        """
        Reads raw RF data from a binary file, extracting header information, timestamps, and frame data.
        
        This function follows the format used by Clarius' ultrasound raw data, as found in their GitHub repository:
        https://github.com/clariusdev/raw/blob/master/common/python/rdataread.py
        
        The function parses:
        - A 5-field header (id, frames, lines, samples, samplesize)
        - Timestamps for each frame
        - RF data for each frame
        
        The loaded data is stored as instance attributes:
        - `self.rf_raw_hdr`: Dictionary containing header information
        - `self.rf_raw_timestamps_1d`: NumPy array of frame timestamps
        - `self.rf_raw_data_3d`: NumPy array of the RF data
        
        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If the file format is incorrect.
        """
        logging.info("Reading raw RF file: %s", self.rf_raw_path)

        # Define header fields and initialize dictionary
        hdr_keys = ('id', 'frames', 'lines', 'samples', 'samplesize')
        hdr = {}

        try:
            with open(self.rf_raw_path, 'rb') as raw_bytes:
                logging.info("Opened file successfully.")
                
                # Read and parse header
                hdr = {key: int.from_bytes(raw_bytes.read(4), byteorder='little') for key in hdr_keys}
                logging.info("Parsed header: %s", hdr)
                
                # Validate header values
                if any(value <= 0 for value in hdr.values()):
                    logging.error("Invalid header values: %s", hdr)
                    return
                
                frames, lines, samples, samplesize = hdr['frames'], hdr['lines'], hdr['samples'], hdr['samplesize']
                frame_size = lines * samples * samplesize
                
                # Initialize arrays
                timestamps = np.zeros(frames, dtype=np.int64)
                data = np.zeros((lines, samples, frames), dtype=np.int16)
                
                # Read timestamps and data for each frame
                for frame in range(frames):
                    timestamps[frame] = int.from_bytes(raw_bytes.read(8), byteorder='little')
                    frame_data = raw_bytes.read(frame_size)
                    
                    if len(frame_data) != frame_size:
                        logging.error("Unexpected frame size at frame %d: Expected %d bytes, got %d", frame, frame_size, len(frame_data))
                        return
                    
                    data[:, :, frame] = np.frombuffer(frame_data, dtype=np.int16).reshape((lines, samples))
                
                logging.info("Successfully read %d RF frames.", frames)
                
        except FileNotFoundError:
            logging.error("File not found: %s", self.rf_raw_path)
            raise
        except Exception as e:
            logging.error("Error reading RF data: %s", str(e))
            raise

        logging.info("Loaded %d raw RF frames of size %d x %d (lines x samples)", data.shape[2], data.shape[0], data.shape[1])
        
        self.rf_raw_hdr, self.rf_raw_timestamps_1d, self.rf_raw_data_3d = hdr, timestamps, data

    ###################################################################################
    
    def read_env_raw(self):
        """
        Reads raw environmental data from a binary file, extracting header information, timestamps, and frame data.
        
        This function reads data in the format used by Clarius' ultrasound raw data, as found in their GitHub repository:
        https://github.com/clariusdev/raw/blob/master/common/python/rdataread.py
        
        The function parses:
        - A 5-field header (id, frames, lines, samples, samplesize)
        - Timestamps for each frame
        - Image data for each frame
        
        The loaded data is stored as instance attributes:
        - `self.env_raw_hdr`: Dictionary containing header information
        - `self.env_raw_timestamps_1d`: NumPy array of frame timestamps
        - `self.env_raw_data_3d`: NumPy array of the image data
        
        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If the file format is incorrect.
        """
        logging.info("Reading raw environmental data from: %s", self.env_raw_path)
        
        hdr_info = ('id', 'frames', 'lines', 'samples', 'samplesize')
        hdr, timestamps, data = {}, None, None

        try:
            with open(self.env_raw_path, 'rb') as raw_bytes:
                logging.info("Opened file successfully.")
                
                # Read 4-byte header values
                for info in hdr_info:
                    hdr[info] = int.from_bytes(raw_bytes.read(4), byteorder='little')
                logging.info("Parsed header: %s", hdr)
                
                # Prepare timestamp and data arrays
                timestamps = np.zeros(hdr['frames'], dtype='int64')
                sz = hdr['lines'] * hdr['samples'] * hdr['samplesize']
                data = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype='uint8')
                
                for frame in range(hdr['frames']):
                    # Read 8-byte timestamp
                    timestamps[frame] = int.from_bytes(raw_bytes.read(8), byteorder='little')
                    
                    # Read and reshape frame data
                    data[:, :, frame] = np.frombuffer(raw_bytes.read(sz), dtype='uint8').reshape([hdr['lines'], hdr['samples']])
                
                logging.info("Successfully read %d frames.", hdr['frames'])
                
        except FileNotFoundError:
            logging.error("File not found: %s", self.env_raw_path)
            raise
        except Exception as e:
            logging.error("Error reading raw data: %s", str(e))
            raise

        logging.info("Loaded %d raw frames of size %d x %d (lines x samples)", data.shape[2], data.shape[0], data.shape[1])
        
        self.env_raw_hdr, self.env_raw_timestamps_1d, self.env_raw_data_3d = hdr, timestamps, data
           
    ###################################################################################
    
    def set_default_tgc_data(self):
        rf_tgc = self.rf_yml_obj.rf_tgc  # Extract the original data
        self.default_tgc_data = []  # Initialize empty list

        for entry in rf_tgc:
            keys = list(entry.keys())  # Extract keys dynamically
            if len(keys) == 2:
                length_key = keys[0]  # e.g., "0.00mm"
                db_key = keys[1]  # e.g., "5.00dB"

                # Extract depth from length_key
                depth_match = re.findall(r"[\d.]+", length_key)
                depth_value = float(depth_match[0]) if depth_match else None  # Convert to float

                # Extract dB from db_key
                db_match = re.findall(r"[\d.]+", db_key)
                db_value = float(db_match[0]) if db_match else None  # Convert to float

                if depth_value is not None and db_value is not None:
                    self.default_tgc_data.append({'depth': depth_value, 'dB': db_value})
    
    ###################################################################################

    def create_clean_no_tgc_data(self):
        previous_data_dict = None

        for timestamp, data_list in self.env_tgc_yml_obj.timestamps.items():
            logging.debug(f"Processing {timestamp}: {data_list}")

            if isinstance(data_list, list) and data_list:  # Check if it's a non-empty list
                data_dict = data_list  
                previous_data_dict = data_dict 
            else:
                logging.warning(f"No valid data for {timestamp}, using previous valid data.")
                data_dict = previous_data_dict if previous_data_dict else self.default_tgc_data

            self.clean_tgc_data[timestamp] = data_dict
            logging.info(f"Final data stored for {timestamp}: {data_dict}")

    ###################################################################################

    def create_no_tgc_raw(self, visualize=False):
        """
        Processes RF data to remove the time-gain compensation (TGC) effect.

        This function computes and applies a correction factor to the RF signal
        to counteract attenuation introduced by TGC. The TGC values are interpolated
        based on depth and applied frame-wise.

        Parameters:
        -----------
        visualize : bool, optional
            If True, plots attenuation versus depth for each frame (default is False).
        """
        full_depth_mm, signal_length, delay_samples = self.create_full_imaging_depth_array()
        num_frames = self.rf_raw_data_3d.shape[2]
        
        for frame in range(num_frames):
            # Initialize TGC array with zeros
            tgc_array_dB = np.zeros(len(full_depth_mm), dtype=np.float16)
            values = list(self.clean_tgc_data.values())[frame]

            # Extract depth and dB values
            depths_mm = np.array([entry['depth'] for entry in values])
            tgc_dB = np.array([entry['dB'] for entry in values])

            # Perform interpolation
            for i, depth in enumerate(full_depth_mm):
                mask = depths_mm > depth
                if mask.any():
                    idx = mask.argmax()
                    if idx == 0:
                        tgc_array_dB[i] = tgc_dB[idx]
                    else:
                        x1, y1 = depths_mm[idx - 1], tgc_dB[idx - 1]
                        x2, y2 = depths_mm[idx], tgc_dB[idx]
                        tgc_array_dB[i] = y1 + (y2 - y1) * (depth - x1) / (x2 - x1)
                else:
                    tgc_array_dB[i] = tgc_dB[-1]

            # Visualization (optional)
            if visualize:
                plt.figure(figsize=(10, 5))
                plt.plot(full_depth_mm, tgc_array_dB, label=f'Frame {frame}', color='blue')
                plt.title(f'Attenuation vs Depth for Frame {frame}')
                plt.xlabel('Depth (mm)')
                plt.ylabel('Attenuation (dB)')
                plt.grid()
                plt.legend()
                plt.show()

            # Apply correction
            trimmed_tgc_dB = tgc_array_dB[delay_samples: delay_samples + signal_length]
            tgc_coefficient = 10 ** (trimmed_tgc_dB / 20)

            rf_no_tgc_raw = self.rf_raw_data_3d.astype(np.float16)
            for line in range(self.rf_raw_data_3d.shape[0]):
                rf_no_tgc_raw[line, :, frame] /= tgc_coefficient

            self.rf_no_tgc_raw_data_3d = rf_no_tgc_raw
        
    ###################################################################################

    def create_full_imaging_depth_array(self):
        """
        Generates an array representing the full imaging depth in millimeters.

        This function calculates the depth array based on the imaging depth and delay samples,
        ensuring accurate mapping of signal depth.

        Returns:
        --------
        tuple: (full_imaging_depth_array_1d_mm, trimmed_signal_length, delay_samples)
            - full_imaging_depth_array_1d_mm: 1D NumPy array of depth values in mm.
            - trimmed_signal_length: Length of the trimmed RF signal.
            - delay_samples: Number of delay samples before the valid signal starts.
        """
        trimmed_signal_length = self.rf_raw_data_3d.shape[1]
        delay_samples = self.rf_yml_obj.rf_delay_samples
        imaging_depth_mm = self.extract_float(self.rf_yml_obj.rf_imaging_depth)

        # Compute full depth array
        full_signal_length = trimmed_signal_length + delay_samples
        depth_array_mm = np.linspace(0, imaging_depth_mm, full_signal_length, dtype=np.float16)

        return depth_array_mm, trimmed_signal_length, delay_samples

    ###################################################################################
    
    def set_trimmed_imaging_depth_array(self):
        """
        Sets the trimmed imaging depth array in centimeters.

        This function extracts the full imaging depth array, removes the delay samples,
        and converts the values from millimeters to centimeters.

        Updates:
        --------
        self.trimmed_imaging_depth_array_1d_cm : 1D NumPy array
            Trimmed imaging depth array in cm.
        """
        full_depth_array_mm, _, delay_samples = self.create_full_imaging_depth_array()
        self.trimmed_imaging_depth_array_1d_cm = full_depth_array_mm[delay_samples:] * 0.1  # Convert mm to cm

    ###################################################################################
    @staticmethod
    def extract_float(input_string):
        """
        Extracts the first floating-point number from a string and returns it as a float.
        If no number is found, returns None.
        """
        match = re.search(r"\d+\.\d+", input_string)
        return float(match.group()) if match else None

    ###################################################################################
    class YmlParser():
        """
        This class reads YAML file data related to ultrasound imaging parameters. 
        It extracts information from two YAML files:
        
        1. `rf.yml` - Contains data such as sampling frequency, probe details, imaging parameters, 
        and transmission settings.
        2. `env.tgc.yml` - Contains time gain compensation (TGC) data.

        The extracted data is used to generate new imaging data without TGC, 
        requiring TGC values for processing.
        """
        ###################################################################################
        
        def __init__(self, yml_path):
            
            self.path: str = yml_path
            self.valid_versions: list[str] = ['12.0.1-673']
            self.extension: str
            
            # rf.yml
            self.rf_software_version: str
            self.rf_iso_time_date: str
            self.rf_probe: dict
            self.rf_frames: int
            self.rf_frame_rate: str
            self.rf_transmit_frequency: str
            self.rf_imaging_depth: str
            self.rf_focal_depth: str
            self.rf_auto_gain: bool
            self.rf_mla: bool
            self.rf_tgc: dict
            self.rf_size: dict
            self.rf_type: str
            self.rf_compression: str
            self.rf_sampling_rate: float
            self.rf_delay_samples: int
            self.rf_lines: dict
            self.rf_focus: dict
            
            # env.yml
            self.env_software_version: str
            self.env_iso_time_date: str
            self.env_probe: dict
            self.env_frames: int
            self.env_frame_rate: str
            self.env_transmit_frequency: str
            self.env_imaging_depth: str
            self.env_focal_depth: str
            self.env_auto_gain: bool
            self.env_mla: bool
            self.env_tgc: dict
            self.env_size: dict
            self.env_type: str
            self.env_compression: str
            self.env_sampling_rate: float
            self.env_delay_samples: int
            self.env_lines: dict
            self.env_focus: dict
            self.env_compound: dict
            self.env_roi: dict
            
            # env.tgc.yml
            self.frames: int
            self.timestamps: dict
            
            self.__run()
            
        ###################################################################################
        
        def __run(self):
            """Executes the internal workflow of the class."""
            self.set_file_extension()
            self.load_rf_yml()
            self.load_env_yml()
            self.load_env_tgc_yml()
            self.check_version()

        ###################################################################################
        
        def set_file_extension(self):
            """
            Extracts and sets the file format based on the part after the last underscore `_`.
            Logs any issues encountered.
            """
            if not isinstance(self.path, str):
                logging.error("Invalid path: Path should be a string.")
                return None

            filename = os.path.basename(self.path)  # Get the filename only
            parts = filename.rsplit("_", 1)  # Split at the last underscore

            if len(parts) < 2:
                logging.warning("No underscore found in filename: %s", self.path)
                return None

            self.extension = parts[-1]  # Get the part after the last underscore
            logging.info("File format detected after last underscore: %s", self.extension)
    
        ###################################################################################
        
        def load_rf_yml(self):
            """
            Loads and parses an RF YAML file if the file extension is "rf.yml".
            The method reads the YAML file, extracts relevant fields, and maps them to class attributes.
            
            Attributes Populated:
            - software_version (str): Software version from the YAML file.
            - iso_time_date (str): ISO formatted timestamp.
            - probe_version (str): Version of the probe.
            - probe_elements (int): Number of probe elements.
            - probe_pitch (float): Probe pitch value.
            - probe_radius (float): Probe radius value.
            - frames (int): Number of frames in the file.
            - frame_rate (float): Frame rate value.
            - transmit_frequency (float): Transmit frequency.
            - imaging_depth (float): Imaging depth.
            - focal_depth (float): Focal depth.
            - auto_gain (bool): Whether auto gain is enabled.
            - mla (bool): Multi-line acquisition status.
            - tgc (dict): Time gain compensation settings.
            - size (dict): Size specifications.
            - type (str): Type information.
            - compression (str): Compression method.
            - sampling_rate (float): Sampling rate value.
            - delay_samples (int): Number of delay samples.
            - lines (dict): Line specifications.
            - focus (list): List of focus parameters.
            
            Raises:
            - Exception: If an error occurs while loading the YAML file.
            """
            if self.extension == "rf.yml":
                try:
                    with open(self.path, 'r') as file:
                        data = yaml.safe_load(file)

                    # Mapping YAML fields to class attributes
                    self.rf_software_version = data.get("software version", None)
                    self.rf_iso_time_date = data.get("iso time/date", None)
                    self.rf_probe = data.get("probe", {})
                    self.rf_frames = data.get("frames", None)
                    self.rf_frame_rate = data.get("frame rate", None)
                    self.rf_transmit_frequency = data.get("transmit frequency", None)
                    self.rf_imaging_depth = data.get("imaging depth", None)
                    self.rf_focal_depth = data.get("focal depth", None)
                    self.rf_auto_gain: bool = data.get("auto gain", None)
                    self.rf_mla: bool = data.get("mla", None)
                    self.rf_tgc = data.get("tgc", {})
                    self.rf_size = data.get("size", {})
                    self.rf_type = data.get("type", None)
                    self.rf_compression = data.get("compression", None)
                    self.rf_sampling_rate = data.get("sampling rate", None)
                    self.rf_delay_samples = data.get("delay samples", None)
                    self.rf_lines = data.get("lines", {})
                    self.rf_focus = data.get("focus", [{}])
                   
                except Exception as e:
                    logging.error(f"Error loading YAML file: {e}")
  
        ###################################################################################
        
        def load_env_yml(self):              
            """
            Loads environment configuration from a YAML file (`env.yml`) and maps its contents 
            to class attributes.

            This method reads a YAML file specified by `self.path` and extracts various 
            parameters related to the environment, such as software version, imaging settings, 
            probe details, and acquisition parameters. The extracted data is stored in 
            corresponding instance variables.

            Attributes Set:
                - env_software_version (str or None): Software version from YAML.
                - env_iso_time_date (str or None): ISO formatted time/date.
                - env_probe (dict): Probe details.
                - env_frames (int or None): Number of frames.
                - env_frame_rate (float or None): Frame rate.
                - env_transmit_frequency (float or None): Transmit frequency.
                - env_imaging_depth (float or None): Imaging depth.
                - env_focal_depth (float or None): Focal depth.
                - env_auto_gain (bool or None): Auto gain setting.
                - env_mla (bool or None): Multi-line acquisition setting.
                - env_tgc (dict): Time Gain Compensation settings.
                - env_size (dict): Image size parameters.
                - env_type (str or None): Data type.
                - env_compression (str or None): Compression type.
                - env_sampling_rate (float or None): Sampling rate.
                - env_delay_samples (int or None): Number of delay samples.
                - env_lines (dict): Line-related parameters.
                - env_focus (list of dicts): Focus parameters.
                - env_compound (dict): Compound imaging settings.
                - env_roi (dict): Region of Interest settings.

            Error Handling:
                - Logs an error message if the YAML file cannot be read or parsed.

            """  
            if self.extension == "env.yml":
                try:
                    with open(self.path, 'r') as file:
                        data = yaml.safe_load(file)

                    # Mapping YAML fields to class attributes
                    self.env_software_version = data.get("software version", None)
                    self.env_iso_time_date = data.get("iso time/date", None)
                    self.env_probe = data.get("probe", {})
                    self.env_frames = data.get("frames", None)
                    self.env_frame_rate = data.get("frame rate", None)
                    self.env_transmit_frequency = data.get("transmit frequency", None)
                    self.env_imaging_depth = data.get("imaging depth", None)
                    self.env_focal_depth = data.get("focal depth", None)
                    self.env_auto_gain: bool = data.get("auto gain", None)
                    self.env_mla: bool = data.get("mla", None)
                    self.env_tgc = data.get("tgc", {})
                    self.env_size = data.get("size", {})
                    self.env_type = data.get("type", None)
                    self.env_compression = data.get("compression", None)
                    self.env_sampling_rate = data.get("sampling rate", None)
                    self.env_delay_samples = data.get("delay samples", None)
                    self.env_lines = data.get("lines", {})
                    self.env_focus = data.get("focus", [{}])
                    self.env_compound = data.get("compound", {})
                    self.env_roi = data.get("roi", {})
                    
                except Exception as e:
                    logging.error(f"Error loading YAML file: {e}")

        ###################################################################################

        def load_env_tgc_yml(self):
            """
            Loads and parses an environmental TGC YAML file if the file extension is "env.tgc.yml".
            The method reads the YAML file, extracts relevant fields, and maps them to class attributes.
            
            Attributes Populated:
            - frames (int): Number of frames specified in the file.
            - timestamps (dict): Dictionary mapping timestamps to lists of depth and dB values.
            Each entry in the list is a dictionary with:
                - depth (float): Depth value in millimeters.
                - dB (float): Gain value in decibels.
            
            Raises:
            - Exception: If an error occurs while loading the YAML file.
            """
            if self.extension == "env.tgc.yml":
                try:
                    with open(self.path, 'r') as file:
                        lines = file.readlines()

                    self.timestamps = {}
                    current_timestamp = None
                    self.frames = None

                    for line in lines:
                        line = line.strip()

                        if line.startswith("frames:"):
                            self.frames = int(line.split(":")[1].strip())

                        elif line.startswith("timestamp:"):
                            # Extract timestamp value
                            current_timestamp = int(line.split(":")[1].strip())
                            self.timestamps[current_timestamp] = []

                        elif current_timestamp is not None and line.startswith("- {"):
                            # Extract depth and dB values
                            values = line.replace("- {", "").replace("}", "").split(", ")
                            depth = float(values[0].replace("mm", "").strip())
                            dB = float(values[1].replace("dB", "").strip())

                            # Store in list under the current timestamp
                            self.timestamps[current_timestamp].append({"depth": depth, "dB": dB})
                   
                except Exception as e:
                    logging.error(f"Error loading YAML file: {e}")

        ###################################################################################

        def check_version(self):
            """
            Checks if the current software version (from env.yml or rf.yml) 
            is in the list of valid versions.

            Logs an informational message if the version is valid and a warning if it is not.

            :return: True if the version is valid, False otherwise.
            """
            
            if self.extension == "env.tgc.yml":
                return  # Ignoriere diese spezielle Datei

            # Prüfen, ob die Attribute existieren, bevor sie verwendet werden
            env_version = getattr(self, "env_software_version", None)
            rf_version = getattr(self, "rf_software_version", None)

            version_mapping = {
                "env.yml": env_version,
                "rf.yml": rf_version
            }

            software_version = version_mapping.get(self.extension)

            if software_version in self.valid_versions:
                logging.info(f"Version {software_version} is valid.")
                return True
            else:
                logging.warning(f"Version {software_version} is not valid. This might cause some problems.")
                return False

        ###################################################################################
        
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
    # Check yml files
    YmlParser(imgTgcFilename); YmlParser(phantomTgcFilename); YmlParser(infoFilename); YmlParser(phantomInfoFilename)
    
    imgData, imgInfo, scanConverted = readImg(imgFilename, imgTgcFilename, infoFilename, version, isPhantom=False)
    refData, refInfo, scanConverted = readImg(phantomFilename, phantomTgcFilename, phantomInfoFilename, version, isPhantom=False)
    return imgData, imgInfo, refData, refInfo, scanConverted

###################################################################################
