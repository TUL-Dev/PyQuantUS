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
#logging.disable(logging.CRITICAL)


# tar file unpacker    
###################################################################################  
class ClariusTarUnpacker():
    """
    A class for extracting and processing `.tar` archives containing `.lzo` and `.raw` files.
    
    Attributes:
        tar_files_path (str): The path to the directory containing `.tar` files.
        extraction_mode (str): Extraction mode - either "single" or "multiple".
        lzo_py_file_path (str): Path to the LZO executable for decompression (Windows only).
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
        #self.lzo_py_file_path = 'pyquantus/lzop/lzop.py'
        
        # Using lzop.py file for Windows renamed to lzop.py to make it pip-accessible
        self.lzo_py_file_path = rf"{os.path.join(os.path.abspath(__file__), os.pardir, os.pardir)}\lzop\lzop.py"
        
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
            - Iterates through the `.lzo` files and decompresses them using `lzop.py`.
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
            # working_space_path = os.getcwd()
            
            # Construct the full path to the LZO executable, adding the .py extension
            # path_of_lzo_py_file = os.path.join(working_space_path, self.lzo_py_file_path)
            
            # Log the path being checked
            logging.info(f'Checking path for LZO executable: {self.lzo_py_file_path}')

            # Check if the executable exists
            if not os.path.isfile(self.lzo_py_file_path):
                logging.error(f'LZO executable not found: {self.lzo_py_file_path}')
                return

            for lzo_file_path in self.lzo_files_path_list:
                logging.info(f'Starting decompression for: {lzo_file_path}')
                try:
                    # Run the lzop command to decompress the LZO file
                    subprocess.run([self.lzo_py_file_path, '-d', lzo_file_path], check=True)
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
                    subprocess.run(['brew', 'update'], check=True)
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
class ClariusParser():

    ###################################################################################
    
    def __init__(self, extracted_sample_folder_path: str, visualize: bool=False):
    
        self.path = extracted_sample_folder_path
        self.visualize = visualize
        
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
        
        # no tgc
        self.rf_no_tgc_raw_data_3d: np.ndarray
        
        # depth and time
        self.trimmed_depth_array_1d_cm: np.ndarray
        self.trimmed_time_array_1d_s: np.ndarray
        
        # visualization
        self.default_frame: int = 0
        self.SPEED_OF_SOUND: int = 1540 # [m/s]
        self.hilbert_transform_axis: int = 1
        
        # data structure
        self.clarius_info_struct: object = ClariusParser.ClariusInfoStruct()
        self.clarius_data_struct: object = DataOutputStruct()
        
        # nd signal and envelope
        self.tgc_signal_2d: np.ndarray 
        self.tgc_signal_1d: np.ndarray             
        self.no_tgc_signal_2d: np.ndarray 
        self.no_tgc_signal_1d: np.ndarray 
        self.tgc_envelope_2d: np.ndarray
        self.no_tgc_envelope_3d: np.ndarray
        self.no_tgc_envelope_2d: np.ndarray
        
        self.scan_converted = None

        self.__run()
                
    ###################################################################################
    
    def __run(self):
        if not self.check_required_files(): return

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
        
        # time and deoth array
        self.set_trimmed_depth_and_time_array()
        
        # set signal and envelope
        self.set_tgc_signal_nd()
        self.set_tgc_envelope_nd()
        self.set_no_tgc_signal_nd()
        self.set_no_tgc_envelope_nd()
        
        # et data to structure       
        self.set_data_of_clarius_info_struct()
        
        # normalize envelope
        # self.tgc_envelope_2d = self.normalize_envelope(self.tgc_envelope_2d) # !!!
        # self.no_tgc_envelope_2d = self.normalize_envelope(self.no_tgc_envelope_2d) # !!!
        
        # convert envelope
        self.convert_envelope() # !!!
        
        # visualize
        if not self.visualize: return
        self.image_envelope_2d(self.tgc_envelope_2d, title="rf_raw")
        self.image_envelope_2d(self.no_tgc_envelope_2d, title="rf_no_tgc_raw")
        self.plot_1d_signal_and_fft(self.tgc_signal_1d, title="rf_raw")
        self.plot_1d_signal_and_fft(self.no_tgc_signal_1d, title="rf_no_tgc_raw")

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

    def read_ymls(self) -> None:
        """
        Reads and parses multiple YAML files using the ClariusParser.YmlParser.

        This function initializes YAML parser objects for different configuration files 
        and assigns them to instance attributes.

        Attributes:
            self.rf_yml_obj: Parses the RF YAML file located at `self.rf_yml_path`.
            self.env_yml_obj: Parses the environment YAML file located at `self.env_yml_path`.
            self.env_tgc_yml_obj: Parses the environment TGC YAML file located at `self.env_tgc_yml_path`.

        Returns:
            None
        """
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
    
    def set_default_tgc_data(self) -> None:
        """
        Extracts and processes TGC (Time Gain Compensation) data from the RF YAML object.

        This function parses key-value pairs from the RF TGC data, extracts depth and 
        dB values, and stores them in a structured format within `self.default_tgc_data`.

        Attributes:
            self.default_tgc_data (list[dict]): A list of dictionaries, each containing:
                - 'depth' (float): The extracted depth value from the length key.
                - 'dB' (float): The extracted dB value from the dB key.

        Returns:
            None
        """
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

    def create_clean_no_tgc_data(self) -> None:
        """
        Creates a cleaned version of TGC (Time Gain Compensation) data by ensuring 
        each timestamp has valid data. If a timestamp lacks valid data, it uses the 
        most recent valid data or falls back to default TGC data.

        This function processes TGC data from `self.env_tgc_yml_obj.timestamps` and 
        stores the cleaned version in `self.clean_tgc_data`.

        Attributes:
            self.clean_tgc_data (dict): A dictionary where:
                - Keys (str): Timestamps from `env_tgc_yml_obj.timestamps`.
                - Values (list[dict]): Processed TGC data lists, falling back to 
                previous valid data or default TGC data if necessary.

        Returns:
            None
        """
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
        
        rf_no_tgc_raw = np.copy(self.rf_raw_data_3d).astype(np.float64)

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
            trimmed_tgc_coefficient = 10 ** (trimmed_tgc_dB / 20)   
            
            for line in range(rf_no_tgc_raw.shape[0]):
                rf_no_tgc_raw[line, :, frame] /= trimmed_tgc_coefficient

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
        imaging_depth_mm = self.extract_digit(self.rf_yml_obj.rf_imaging_depth)

        # Compute full depth array
        full_signal_length = trimmed_signal_length + delay_samples
        depth_array_mm = np.linspace(0, imaging_depth_mm, full_signal_length, dtype=np.float16)

        return depth_array_mm, trimmed_signal_length, delay_samples

    ###################################################################################
    
    def set_trimmed_depth_and_time_array(self):
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
        self.trimmed_depth_array_1d_cm = full_depth_array_mm[delay_samples:] * 0.1  # Convert mm to cm
        self.trimmed_time_array_1d_s = (self.trimmed_depth_array_1d_cm * 0.1) / self.SPEED_OF_SOUND
        
        self.trimmed_depth_array_1d_cm = np.array(self.trimmed_depth_array_1d_cm, dtype=np.float64)
        self.trimmed_time_array_1d_s = np.array(self.trimmed_time_array_1d_s, dtype=np.float64)
        
    ###################################################################################
    
    def set_tgc_signal_nd(self) -> None:
        """
        Extracts and sets the TGC (Time Gain Compensation) signal in different dimensions.

        This function extracts a 2D slice from the 3D RF raw data using `self.default_frame`, 
        and then extracts a 1D signal from the center row of the 2D slice.

        Attributes:
            self.tgc_signal_2d (np.ndarray): A 2D array extracted from `self.rf_raw_data_3d` 
                at the specified default frame.
            self.tgc_signal_1d (np.ndarray): A 1D signal extracted from the center row 
                of `self.tgc_signal_2d`.

        Returns:
            None
        """
        self.tgc_signal_2d = self.rf_raw_data_3d[:, :, self.default_frame]
        self.tgc_signal_1d = self.tgc_signal_2d[self.tgc_signal_2d.shape[0] // 2, :]

    ###################################################################################
    
    def set_no_tgc_signal_nd(self) -> None:
        """
        Extracts and sets the non-TGC (Time Gain Compensation) signal in different dimensions.

        This function extracts a 2D slice from the 3D RF raw data without TGC using `self.default_frame`, 
        and then extracts a 1D signal from the center row of the 2D slice.

        Attributes:
            self.no_tgc_signal_2d (np.ndarray): A 2D array extracted from `self.rf_no_tgc_raw_data_3d` 
                at the specified default frame.
            self.no_tgc_signal_1d (np.ndarray): A 1D signal extracted from the center row 
                of `self.no_tgc_signal_2d`.

        Returns:
            None
        """
        self.no_tgc_signal_2d = self.rf_no_tgc_raw_data_3d[:, :, self.default_frame]
        self.no_tgc_signal_1d = self.no_tgc_signal_2d[self.no_tgc_signal_2d.shape[0] // 2, :]
        
    ###################################################################################
    
    def set_tgc_envelope_nd(self) -> None:
        """
        Computes and sets the TGC (Time Gain Compensation) envelope in multiple dimensions.

        This function applies the Hilbert transform to the 2D TGC signal to compute its 
        envelope and stores the result.

        Attributes:
            self.tgc_envelope_2d (np.ndarray): A 2D array representing the envelope 
                of `self.tgc_signal_2d`, computed using the Hilbert transform.

        Returns:
            None
        """
        self.tgc_envelope_2d = self.get_signal_envelope_xd(self.tgc_signal_2d,
                                                           hilbert_transform_axis=self.hilbert_transform_axis)

    ###################################################################################

    def set_no_tgc_envelope_nd(self) -> None:
        """
        Computes and sets the non-TGC (Time Gain Compensation) envelope in multiple dimensions.

        This function applies the Hilbert transform to both the 3D and 2D non-TGC signals 
        to compute their envelopes and stores the results.

        Attributes:
            self.no_tgc_envelope_3d (np.ndarray): A 3D array representing the envelope 
                of `self.rf_no_tgc_raw_data_3d`, computed using the Hilbert transform.
            self.no_tgc_envelope_2d (np.ndarray): A 2D array representing the envelope 
                of `self.no_tgc_signal_2d`, computed using the Hilbert transform.

        Returns:
            None
        """
        self.no_tgc_envelope_3d = self.get_signal_envelope_xd(self.rf_no_tgc_raw_data_3d,
                                                              hilbert_transform_axis=self.hilbert_transform_axis)
        
        self.no_tgc_envelope_2d = self.get_signal_envelope_xd(self.no_tgc_signal_2d,
                                                              hilbert_transform_axis=self.hilbert_transform_axis)

    ###################################################################################
    
    def get_signal_envelope_xd(self,
                               signal_xd: np.ndarray,
                               hilbert_transform_axis: int) -> np.ndarray:
        """
        Computes the envelope of an x-dimensional signal using the Hilbert transform.

        This function mirrors the input signal along all its dimensions, applies the 
        Hilbert transform along the specified axis, and then extracts the original 
        signal envelope from the analytic signal.

        Args:
            signal_xd (np.ndarray): The input x-dimensional signal represented as a NumPy array.
            hilbert_transform_axis (int): The axis along which the Hilbert transform is applied.

        Returns:
            np.ndarray: The computed signal envelope of the input signal.
        """
        logging.debug("Starting to set the xD signal envelope.")

        # Step 1: Mirror the signal
        logging.debug("Mirroring the signal.")
        signal_xd_mirrored = signal_xd.copy()
        
        for axis in range(signal_xd_mirrored.ndim):
            logging.debug(f"Reversing signal along axis {axis}.")
            # Reverse the signal along the current axis
            reversed_signal = np.flip(signal_xd_mirrored, axis)
            
            logging.debug(f"Concatenating original and reversed signal along axis {axis}.")
            # Concatenate the original and reversed signal along the current axis
            signal_xd_mirrored = np.concatenate((signal_xd_mirrored, reversed_signal), axis=axis)

        # Step 2: Apply the Hilbert transform to the mirrored signal
        logging.debug("Applying Hilbert transform to the mirrored signal.")
        hilbert_signal_mirrored = np.apply_along_axis(hilbert, arr=signal_xd_mirrored, axis=hilbert_transform_axis)
        
        # Step 3: Restore the original signal from the mirrored Hilbert signal
        logging.debug("Restoring the original signal from the mirrored Hilbert signal.")
        restored_signal = hilbert_signal_mirrored.copy()
        
        # Calculate the original size of the signal along each axis
        original_sizes = tuple(restored_signal.shape[axis] // 2 for axis in range(restored_signal.ndim))
        logging.debug(f"Original sizes of the signal: {original_sizes}.")
        
        # Slice to get the original signal
        logging.debug("Slicing to get the original signal.")
        signal_xd_unmirrored = restored_signal[tuple(slice(0, size) for size in original_sizes)]
        
        # Step 4: Compute the envelope of the analytic signal
        logging.debug("Computing the envelope of the analytic signal.")
        signal_envelope_xd = np.abs(signal_xd_unmirrored)

        logging.debug("Finished setting the xD signal envelope.")
        
        return signal_envelope_xd
                
    ###################################################################################

    def image_envelope_2d(self, envelope: np.ndarray, title: str) -> None:
        """
        Plots a 2D signal envelope in decibels.

        This function takes a 2D array representing a signal envelope, applies 
        a rotation and flipping transformation, converts it to a logarithmic 
        decibel scale, and displays it as an image plot.

        Args:
            envelope (np.ndarray): A 2D NumPy array representing the signal envelope.
            title (str): The title of the plot.

        Returns:
            None
        """
        logging.info("Starting the plot function.")
        
        # Flip the rotated array horizontally
        rotated_flipped_array = self.rotate_flip(envelope)

        log_envelope_2d = 20 * np.log10(np.abs(1 + rotated_flipped_array))
        
        logging.debug("Calculated log envelope 2D.")

        plt.figure(figsize=(8, 6))
        plt.imshow(log_envelope_2d, cmap='gray', aspect='auto')
        plt.title(title)
        plt.colorbar(label='dB')
        logging.info("Displayed 2D Signal Envelope.")

        plt.tight_layout()
        plt.show()
        logging.info("Plotting completed and displayed.")

    ###################################################################################

    def plot_1d_signal_and_fft(self, signal_1d: np.ndarray, title: str):
        """
        Plots a 1D signal and its corresponding FFT.
        
        Args:
            signal_1d (np.ndarray): The 1D signal to be analyzed.
            title (str): Title for the plots.
        """
        sampling_rate_Hz = self.extract_digit(self.rf_yml_obj.rf_sampling_rate) * 1e6
                        
        # Get envelope
        envelope_1d = self.get_signal_envelope_xd(signal_1d, hilbert_transform_axis=0)

        # Calculate FFT
        fft_signal = np.fft.fft(signal_1d)
        fft_magnitude = np.abs(fft_signal)

        # Frequency axis in MHz using the sampling frequency
        num_samples = len(signal_1d)
        freq_Hz = np.fft.fftfreq(num_samples, d=(1 / (sampling_rate_Hz)))  # Frequency in Hz

        # Only take positive frequencies
        positive_freq_indices = freq_Hz >= 0
        freq_positive_MHz = freq_Hz[positive_freq_indices] / 1e6
        fft_magnitude_positive = fft_magnitude[positive_freq_indices]

        # Create figure
        plt.figure(figsize=(14, 4))

        # Plot the original signal with envelope
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.plot(self.trimmed_depth_array_1d_cm, signal_1d, color='blue', label='Signal')  
        plt.plot(self.trimmed_depth_array_1d_cm, envelope_1d, color='red', linestyle='dashed', label='Envelope')  
        plt.title(f'1D Signal Plot {title}')
        plt.xlabel('Depth (cm)')  
        plt.ylabel('Amplitude')
        plt.legend()  # Add legend
        plt.grid(True)

        # Plot the positive FFT
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        plt.plot(freq_positive_MHz, fft_magnitude_positive, color='green')  
        plt.title(f'FFT of Signal {title}')
        plt.xlabel('Frequency (MHz)')  
        plt.ylabel('Magnitude')
        plt.grid(True)

        plt.tight_layout()  
        plt.show()

    ###################################################################################
    
    def set_data_of_clarius_info_struct(self):
        
        self.clarius_info_struct.width1 = self.rf_yml_obj.rf_probe['radius'] * 2
        self.clarius_info_struct.endDepth1 = self.extract_digit(self.rf_yml_obj.rf_imaging_depth) / 100 # [m]
        self.clarius_info_struct.startDepth1 = self.clarius_info_struct.endDepth1 / 4 # [m]
        self.clarius_info_struct.samplingFrequency = self.extract_digit(self.rf_yml_obj.rf_sampling_rate) * 1e6 # [Hz]
        self.clarius_info_struct.tilt1 = 0
        self.clarius_info_struct.samplesPerLine = self.rf_yml_obj.rf_size['samples per line']
        self.clarius_info_struct.numLines = self.rf_yml_obj.rf_size['number of lines']
        self.clarius_info_struct.sampleSize = self.rf_yml_obj.rf_size['sample size']        
        self.clarius_info_struct.centerFrequency = self.extract_digit(self.rf_yml_obj.rf_transmit_frequency) * 1e6 # [Hz]
        self.clarius_info_struct.minFrequency = 0
        self.clarius_info_struct.maxFrequency = self.clarius_info_struct.centerFrequency * 2
        self.clarius_info_struct.lowBandFreq = int(self.clarius_info_struct.centerFrequency / 2)
        self.clarius_info_struct.upBandFreq = int(self.clarius_info_struct.centerFrequency * 1.5)
        self.clarius_info_struct.clipFact = 0.95
        self.clarius_info_struct.dynRange = 50
        
    ###################################################################################

    def normalize_envelope(self, envelope):

        max_value = np.amax(envelope)
        clip_max = self.clarius_info_struct.clipFact * max_value

        # Clip the envelope: causing prolem, given minimum value is the way more higher than real minimum
        # therefore destroying envelope
        envelope = np.clip(envelope, clip_max - self.clarius_info_struct.dynRange, clip_max)
        
        # Shift to zero
        envelope -= np.amin(envelope)
        
        # Scale to 0-255
        envelope *= 255 / np.amax(envelope)

        return envelope
        
    ###################################################################################
    
    def convert_envelope(self):
        
        if self.clarius_info_struct.width1:
            
            bmode = self.no_tgc_envelope_3d
            rf_atgc = self.rf_raw_data_3d
            
            bmode = np.transpose(bmode, (1, 0, 2))
            rf_atgc = np.transpose(rf_atgc, (1, 0, 2))
            
            scBmodeStruct, hCm1, wCm1 = scanConvert(bmode[:,:,0],
                                                    self.clarius_info_struct.width1,
                                                    self.clarius_info_struct.tilt1,
                                                    self.clarius_info_struct.startDepth1, 
                                                    self.clarius_info_struct.endDepth1,
                                                    desiredHeight=2000)
            
            scBmodes = np.array([scanConvert(bmode[:,:,i],
                                             self.clarius_info_struct.width1,
                                             self.clarius_info_struct.tilt1,
                                             self.clarius_info_struct.startDepth1, 
                                             self.clarius_info_struct.endDepth1,
                                             desiredHeight=2000)[0].scArr for i in tqdm(range(rf_atgc.shape[2]))])
            
            self.clarius_info_struct.yResRF =  self.clarius_info_struct.endDepth1*1000 / scBmodeStruct.scArr.shape[0]
            self.clarius_info_struct.xResRF = self.clarius_info_struct.yResRF * (scBmodeStruct.scArr.shape[0]/scBmodeStruct.scArr.shape[1]) # placeholder
            self.clarius_info_struct.axialRes = hCm1*10 / scBmodeStruct.scArr.shape[0]
            self.clarius_info_struct.lateralRes = wCm1*10 / scBmodeStruct.scArr.shape[1]
            self.clarius_info_struct.depth = hCm1*10 #mm
            self.clarius_info_struct.width = wCm1*10 #mm
            
            self.clarius_info_struct.scBmodeStruct = scBmodeStruct
            self.clarius_info_struct.scBmode = scBmodes
            
            self.clarius_data_struct.bMode = np.transpose(bmode, (2, 0, 1))
            self.clarius_data_struct.rf = np.transpose(rf_atgc, (2, 0, 1))
            
            self.scan_converted = True
            
        else:
            self.clarius_info_struct.yResRF = self.clarius_info_struct.endDepth1*1000 / bmode.shape[0] # mm/pixel
            self.clarius_info_struct.xResRF = self.clarius_info_struct.yResRF * (bmode.shape[0]/bmode.shape[1]) # placeholder
            self.clarius_info_struct.axialRes = self.clarius_info_struct.yResRF #mm
            self.clarius_info_struct.lateralRes = self.clarius_info_struct.xResRF #mm
            self.clarius_info_struct.depth = self.clarius_info_struct.endDepth1*1000 #mm
            self.clarius_info_struct.width = self.clarius_info_struct.endDepth1*1000 #mm
            
            self.clarius_data_struct.bMode = np.transpose(bmode, (2, 0, 1))
            self.clarius_data_struct.rf = np.transpose(rf_atgc, (2, 0, 1))
            
            self.scan_converted = False
            
    ###################################################################################
    @staticmethod
    def extract_digit(input_string: str) -> float | int | None:
        """
        Extracts the first number (integer or floating-point) from a string.
        Returns an int if the number is an integer, otherwise returns a float.
        If no number is found, returns None.
        """
        match = re.search(r"\d+\.\d+|\d+", input_string)  # Match float first, then integer
        if match:
            number = match.group()
            return float(number) if '.' in number else int(number)  # Convert appropriately
        return None

    ###################################################################################
    @staticmethod
    def rotate_flip(two_dimension_array: np.ndarray) -> np.ndarray:
        """
        Rotates the input 2D array counterclockwise by 90 degrees and then flips it vertically.
        
        Args:
            two_dimension_array (np.ndarray): The input 2D array to be transformed.
        
        Returns:
            np.ndarray: The rotated and flipped 2D array.
        """
        # Rotate the input array counterclockwise by 90 degrees
        rotated_array = np.rot90(two_dimension_array)
        
        # Flip the rotated array horizontally
        rotated_flipped_array = np.flipud(rotated_array)
        
        return rotated_flipped_array
    
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
    class ClariusInfoStruct(InfoStruct):
        """
        ClariusInfoStruct is a subclass of InfoStruct that represents information 
        about an ultrasound scan's data structure. 

        Attributes:
            samplesPerLine (int): The number of samples per line in the scan.
            numLines (int): The total number of lines in the scan.
            sampleSize (str): The size of each sample, typically represented as a string.
        """
        
        def __init__(self):
            """
            Initializes a ClariusInfoStruct object with essential attributes related 
            to ultrasound scan data structure.
            """
            super().__init__()
            
            self.samplesPerLine: int  # Number of samples per line
            self.numLines: int  # Total number of lines
            self.sampleSize: str  # Size of each sample
    
###################################################################################



# function
###################################################################################

def clariusRfParser(imgFilename: str,
                    imgTgcFilename: str,
                    infoFilename: str, 
                    phantomFilename: str,
                    phantomTgcFilename: str,
                    phantomInfoFilename: str):

    def get_folder_path_from_file_path(file_path: str) -> str:
        """Returns the absolute directory path of the given file."""
        return os.path.dirname(os.path.abspath(file_path))
    
    main_sample_folder_path    = get_folder_path_from_file_path(imgFilename)
    pahntom_sample_folder_path = get_folder_path_from_file_path(phantomFilename)

    main_sample_obj    = ClariusParser(main_sample_folder_path)
    phantom_sample_obj = ClariusParser(pahntom_sample_folder_path)
      
    imgData       = main_sample_obj.clarius_data_struct   
    imgInfo       = main_sample_obj.clarius_info_struct
    scanConverted = main_sample_obj.scan_converted
    
    refData       = phantom_sample_obj.clarius_data_struct   
    refInfo       = phantom_sample_obj.clarius_info_struct
    scanConverted = phantom_sample_obj.scan_converted
    
    # imgData, imgInfo, scanConverted = readImg(imgFilename, imgTgcFilename, infoFilename, version, isPhantom=False)
    # refData, refInfo, scanConverted = readImg(phantomFilename, phantomTgcFilename, phantomInfoFilename, version, isPhantom=False)
    
    return imgData, imgInfo, refData, refInfo, scanConverted

###################################################################################




# no need for them
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

def generate_default_tgc_matrix(num_frames: int, info: ClariusParser.ClariusInfoStruct) -> np.ndarray:
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
            version="6.0.3", isPhantom=False) -> Tuple[DataOutputStruct, ClariusParser.ClariusInfoStruct, bool]:
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
    

    info = ClariusParser.ClariusInfoStruct()
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

    print(bmode.shape)
    print(rf_atgc.shape)
    
    data.bMode = np.transpose(bmode, (2, 0, 1))
    data.rf = np.transpose(rf_atgc, (2, 0, 1))

    return data, info, scanConverted

###################################################################################

def clariusRfParser_old(imgFilename: str, imgTgcFilename: str, infoFilename: str, 
            phantomFilename: str, phantomTgcFilename: str, phantomInfoFilename: str, 
            version="6.0.3") -> Tuple[DataOutputStruct, ClariusParser.ClariusInfoStruct, DataOutputStruct, ClariusParser.ClariusInfoStruct, bool]:
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
    # YmlParser(imgTgcFilename); YmlParser(phantomTgcFilename); YmlParser(infoFilename); YmlParser(phantomInfoFilename)
    
    imgData, imgInfo, scanConverted = readImg(imgFilename, imgTgcFilename, infoFilename, version, isPhantom=False)
    refData, refInfo, scanConverted = readImg(phantomFilename, phantomTgcFilename, phantomInfoFilename, version, isPhantom=False)
    return imgData, imgInfo, refData, refInfo, scanConverted

###################################################################################