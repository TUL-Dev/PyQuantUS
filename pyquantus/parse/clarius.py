# Standard Library Imports
import os
import re
import sys
import logging
import platform
import shutil
import tarfile
import subprocess
from typing import Tuple

# Third-Party Library Imports
import yaml
import numpy as np
from tqdm import tqdm
from scipy.signal import hilbert
from scipy.interpolate import interp1d

# Local Module Imports
from pyquantus.parse.objects import DataOutputStruct, InfoStruct
from pyquantus.parse.transforms import scanConvert

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#############################################################################################
class ClariusInfo(InfoStruct):
    def __init__(self):
        
        """
        Initializes the ClariusInfo class, inheriting from InfoStruct.
        Defines attributes related to ultrasound imaging data.
        """
        
        super().__init__()
        self.numLines: int  
        self.samplesPerLine: int 
        self.sampleSize: int 

#############################################################################################

def clarius_rf_parser(img_filename: str,
                      img_tgc_filename: str,
                      info_filename: str,
                      phantom_filename: str,
                      phantom_tgc_filename: str,
                      phantom_info_filename: str,
                      version: str = "6.0.3") -> Tuple[DataOutputStruct, ClariusInfo, DataOutputStruct, ClariusInfo, bool]:
    
    """Parses Clarius RF data and metadata from input files.

    Args:
        img_filename (str): File path of the RF data.
        img_tgc_filename (str): File path of the TGC data.
        info_filename (str): File path of the metadata.
        phantom_filename (str): File path of the phantom RF data.
        phantom_tgc_filename (str): File path of the phantom TGC data.
        phantom_info_filename (str): File path of the phantom metadata.
        version (str, optional): Clarius software version. Defaults to "6.0.3".

    Returns:
        Tuple[DataOutputStruct, ClariusInfo, DataOutputStruct, ClariusInfo, bool]: 
            Image data, image metadata, phantom data, phantom metadata, and scan conversion flag.
    """
    
    img_data, img_info, scan_converted = read_img(img_filename,
                                                 img_tgc_filename,
                                                 info_filename,
                                                 version,
                                                 isPhantom=False
                                                 )
    
    ref_data, ref_info, _ = read_img(phantom_filename,
                                    phantom_tgc_filename,
                                    phantom_info_filename,
                                    version,
                                    isPhantom=False
                                    )

    return img_data, img_info, ref_data, ref_info, scan_converted

#############################################################################################

def read_img(filename: str,
             tgc_path: str | None,
             info_path: str,
             version: str,
             is_phantom: bool) -> Tuple[DataOutputStruct, ClariusInfo, bool]:
    
    """
    Reads RF data from a Clarius ultrasound file.

    Args:
        filename (str): Path to the Clarius file.
        tgc_path (str | None): Path to the TGC file (if available).
        info_path (str): Path to the metadata file.
        version (str): Clarius file version (only '6.0.3' is supported).
        is_phantom (bool): True if the data is from a phantom, False if from a patient.

    Returns:
        Tuple: A tuple containing processed RF data, metadata, and a boolean flag.
    """

    if version != "6.0.3":
        print("Unrecognized version")
        return []

    # Read the header information
    try:
        hinfo = np.fromfile(filename, dtype="uint32", count=5)
        header = {
            "id":      hinfo[0],
            "nframes": hinfo[1],  # Number of frames
            "w":       hinfo[2],  # Width (lines)
            "h":       hinfo[3],  # Height (samples)
            "ss":      hinfo[4],  # Sample size
        }
    except Exception as e:
        print(f"Error reading header: {e}")
        return []

    frames = header["nframes"]

    # Check if file contains RF data
    if header["id"] != 2:
        print("File does not contain RF data. Ensure RF mode is enabled during scans.")
        return []

    # Initialize RF data storage
    ts = np.zeros(frames, dtype="uint64")
    data = np.zeros((header["h"], header["w"], frames))

    # Read RF data frame by frame
    try:
        for f in range(frames):
            ts[f] = np.fromfile(filename, dtype="uint64", count=1)[0]
            v = np.fromfile(filename, dtype="int16", count=header["h"] * header["w"])
            data[:, :, f] = np.flip(
                v.reshape(header["h"], header["w"], order="F").astype(np.int16), axis=1
            )
    except Exception as e:
        print(f"Error reading RF data: {e}")
        return []

    # Validate ROI size
    if header["w"] != 192 or header["h"] != 2928:
        print(f"Invalid ROI size: {header['w']}x{header['h']}. Returning empty list.")
        return []

    # Load metadata
    try:
        with open(info_path, 'r') as file:
            info_yml = yaml.safe_load(file)

        info = ClariusInfo()
        info.width1 = info_yml["probe"]["radius"] * 2 if "probe" in info_yml else 0
        scan_converted = "probe" in info_yml
        info.endDepth1 = float(info_yml["imaging depth"][:-2]) / 1000  # meters
        info.startDepth1 = info.endDepth1 / 4
        info.samplingFrequency = int(info_yml["sampling rate"][:-3]) * 1e6
        info.tilt1 = 0
        info.samplesPerLine = info_yml["size"]["samples per line"]
        info.numLines = info_yml["size"]["number of lines"]
        info.sampleSize = info_yml["size"]["sample size"]
        info.centerFrequency = float(info_yml["transmit frequency"][:-3]) * 1e6
        info.minFrequency = 0
        info.maxFrequency = info.centerFrequency * 2
        info.lowBandFreq = int(info.centerFrequency / 2)
        info.upBandFreq = int(info.centerFrequency * 1.5)
        info.clipFact = 0.95
        info.dynRange = 50
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return []

    # Apply TGC correction
    file_timestamp = filename.split("_rf.raw")[0]
    linear_tgc_matrix = generate_tgc_matrix(file_timestamp, tgc_path, ts, frames, info, is_phantom)
    linear_tgc_matrix = np.transpose(linear_tgc_matrix, (1, 0, 2))

    if data.shape[2] != linear_tgc_matrix.shape[2]:
        print(f"Timestamps mismatch in {file_timestamp}. Returning empty array.")
        return []

    rf_matrix_corrected_B = data / linear_tgc_matrix

    # Apply default TGC
    linear_default_tgc_matrix = generate_default_tgc_matrix(frames, info)
    linear_default_tgc_matrix = np.transpose(linear_default_tgc_matrix, (1, 0, 2))
    rf_matrix_corrected_A = rf_matrix_corrected_B * linear_default_tgc_matrix

    dB_tgc_matrix = 20 * np.log10(linear_tgc_matrix)

    # B-mode processing
    bmode = np.zeros_like(rf_matrix_corrected_A)
    for f in range(frames):
        for i in range(header["w"]):
            bmode[:, i, f] = 20 * np.log10(abs(hilbert(rf_matrix_corrected_A[:, i, f])))

    clipped_max = info.clipFact * np.amax(bmode)
    bmode = np.clip(bmode, clipped_max - info.dynRange, clipped_max)
    bmode -= np.amin(bmode)
    bmode *= 255 / np.amax(bmode)

    data_output = DataOutputStruct()

    # Perform scan conversion if applicable
    if scan_converted:
        sc_bmode_struct, h_cm, w_cm = scanConvert(
            bmode[:, :, 0], info.width1, info.tilt1, info.startDepth1, info.endDepth1, desiredHeight=2000
        )

        sc_bmodes = np.array([
            scanConvert(bmode[:, :, i], info.width1, info.tilt1, info.startDepth1, info.endDepth1, desiredHeight=2000)[0].scArr
            for i in range(frames)
        ])

        info.yResRF = info.endDepth1 * 1000 / sc_bmode_struct.scArr.shape[0]
        info.xResRF = info.yResRF * (sc_bmode_struct.scArr.shape[0] / sc_bmode_struct.scArr.shape[1])
        info.axialRes = h_cm * 10 / sc_bmode_struct.scArr.shape[0]
        info.lateralRes = w_cm * 10 / sc_bmode_struct.scArr.shape[1]
        info.depth = h_cm * 10  # mm
        info.width = w_cm * 10  # mm
        data_output.scBmodeStruct = sc_bmode_struct
        data_output.scBmode = sc_bmodes
    else:
        info.yResRF = info.endDepth1 * 1000 / bmode.shape[0]
        info.xResRF = info.yResRF * (bmode.shape[0] / bmode.shape[1])
        info.axialRes = info.yResRF
        info.lateralRes = info.xResRF
        info.depth = info.endDepth1 * 1000
        info.width = info.endDepth1 * 1000

    data_output.bMode = np.transpose(bmode, (2, 0, 1))
    data_output.rf = np.transpose(rf_matrix_corrected_A, (2, 0, 1))

    return data_output, info, scan_converted

#############################################################################################

def read_tgc_file(file_timestamp: str, rf_timestamps: np.ndarray) -> list | None:
    """
    Reads a TGC file and extracts the TGC data corresponding to the provided RF timestamps.

    Args:
        file_timestamp (str): Timestamp of the inputted RF file.
        rf_timestamps (np.ndarray): Array of RF timestamps.

    Returns:
        list | None: Extracted TGC data matching the RF timestamps, or None if no file is found.
    """
    # Define possible filenames
    possible_files = [f"{file_timestamp}_env.tgc", f"{file_timestamp}_env.tgc.yml"]

    # Find an existing TGC file
    tgc_file_name = next((file for file in possible_files if os.path.isfile(file)), None)
    if not tgc_file_name:
        return None

    # Read file content
    with open(tgc_file_name, "r") as file:
        data_str = file.read()

    # Process frames data
    frames_data = data_str.split("timestamp:")[1:]
    
    # Ensure each frame has default values if missing
    frames_data = [
        frame if "{" in frame else frame + "  - { 0.00mm, 15.00dB }\n  - { 120.00mm, 35.00dB }"
        for frame in frames_data
    ]

    # Create a dictionary mapping timestamps to frames
    frames_dict = {
        timestamp: frame
        for frame in frames_data
        for timestamp in rf_timestamps
        if str(timestamp) in frame
    }

    # Extract frames matching given timestamps
    return [frames_dict[timestamp] for timestamp in rf_timestamps if timestamp in frames_dict]

#############################################################################################

def clean_and_convert(value):
    """Clean and convert a string value to a float."""
    clean_value = ''.join([char for char in value if char.isdigit() or char in ['.', '-']])
    return float(clean_value)

#############################################################################################

def extract_tgc_data_from_line(line):
    """Extract TGC data from a line."""
    tgc_pattern = r'\{([^}]+)\}'
    return re.findall(tgc_pattern, line)

#############################################################################################

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

#############################################################################################

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

#############################################################################################

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

#############################################################################################

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

#############################################################################################

def convert_env_to_rf_ntgc(x, linear_tgc_matrix):
    y1 =  47.3 * x + 30
    y = 10**(y1/20)-1
    y = y / linear_tgc_matrix
    return y 

#############################################################################################




# raw files generator from tar files
#############################################################################################
class Clarius_tar_unpacker():
    
    def __init__(self, tar_files_path: str, extraction_mode: str) -> None:  

        self.tar_files_path = tar_files_path
        self.extraction_mode = extraction_mode
        
        if   self.extraction_mode == "single":   self.__run_single_extraction()
        elif self.extraction_mode == "multiple": self.__run_multiple_extraction()    
        else:
            raise ValueError(f"Invalid mode: {self.extraction_mode}")
        
    #########################################################################################
        
    def __run_single_extraction(self):
        
        self.delete_hidden_files_in_sample_folder()
        self.delete_extracted_folders()
        self.extract_tar_files()
        self.set_path_of_extracted_folders()
        self.set_path_of_lzo_files_inside_extracted_folders()
        self.read_lzo_files()
        self.set_path_of_raw_files_inside_extracted_folders()
        self.read_raw_files()
        self.delete_hidden_files_in_extracted_folders()

    #########################################################################################

    def __run_multiple_extraction(self):
        """Extracts data from all directories inside `self.tar_files_path`."""
        try:
            # Retrieve all subdirectory paths
            folder_paths = [
                os.path.join(self.tar_files_path, folder)
                for folder in os.listdir(self.tar_files_path)
                if os.path.isdir(os.path.join(self.tar_files_path, folder))
            ]

            # Process each folder for data extraction
            for folder_path in folder_paths:
                self.tar_files_path = folder_path  # Update path before extraction
                self.__run_single_extraction()

        except Exception as e:
            logging.error(f"An error occurred while extracting data: {e}")

    #########################################################################################
    
    def delete_hidden_files_in_sample_folder(self):
        """
        Deletes hidden files (starting with a dot) from the sample folder.

        Returns:
            bool: True if files were successfully deleted, False otherwise.
        """
        if not os.path.exists(self.tar_files_path):
            logging.error(f"Sample folder path does not exist: {self.tar_files_path}")
            return False  # Indicate failure due to non-existing path

        try:
            deleted_files_count = 0  # Count of deleted files
            # Iterate over the files in the sample folder
            for filename in os.listdir(self.tar_files_path):
                # Check if the file is hidden (starts with a dot)
                if filename.startswith('.'):
                    file_path = os.path.join(self.tar_files_path, filename)
                    os.remove(file_path)  # Delete the hidden file
                    logging.info(f"Deleted hidden file: {file_path}")
                    deleted_files_count += 1

            if deleted_files_count > 0:
                logging.info(f"Total hidden files deleted: {deleted_files_count}")
            else:
                logging.info("No hidden files to delete.")

            return True  # Indicate success
        except Exception as e:
            logging.error(f"An error occurred while deleting hidden files: {e}")
            return False  # Indicate failure
        
    #########################################################################################
       
    def delete_extracted_folders(self):
        """Deletes all extracted folders in the specified directory."""
        extracted_folders = [
            os.path.join(self.tar_files_path, item)
            for item in os.listdir(self.tar_files_path)
            if os.path.isdir(os.path.join(self.tar_files_path, item)) and "extracted" in item
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
        Extracts all tar files in the specified sample folder.

        The extracted files are placed in a subdirectory named after each tar file with '_extracted' appended.
        """
        # Iterate over files in the sample folder
        for item_name in os.listdir(self.tar_files_path):
            item_path = os.path.join(self.tar_files_path, item_name)
            
            # Check if the item is a file
            if os.path.isfile(item_path):
                # Check if the file is a tar file
                if tarfile.is_tarfile(item_path):
                    # Use the file name without the extension for the extracted folder
                    file_name = os.path.splitext(item_name)[0]

                    # Create a new folder for extracted files
                    extracted_folder = os.path.join(self.tar_files_path, f'{file_name}.tar_extracted')
                    os.makedirs(extracted_folder, exist_ok=True)
                    
                    try:
                        # Extract the tar file into the new folder
                        with tarfile.open(item_path, 'r') as tar:
                            tar.extractall(path=extracted_folder)
                            logging.info(f"Extracted '{item_name}' into '{extracted_folder}'")
                    except (tarfile.TarError, OSError) as e:
                        logging.error(f"Error extracting '{item_name}': {e}")

    ###################################################################################
    
    def set_path_of_extracted_folders(self):
        """Finds and stores paths of extracted folders inside `self.tar_files_path`."""
        logging.info("Searching for extracted folders...")

        # Find all directories containing 'extracted' in their name
        self.extracted_folders_path_list = [
            os.path.join(self.tar_files_path, item)
            for item in os.listdir(self.tar_files_path)
            if os.path.isdir(os.path.join(self.tar_files_path, item)) and "extracted" in item
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
        
        # Set self.os based on the platform
        os_name = platform.system().lower()
        if 'windows' in os_name:
            self.os = "windows"
        elif 'darwin' in os_name:
            self.os = "mac"
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
                    result = subprocess.run([path_of_lzo_exe_file, '-d', lzo_file_path], check=True)
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
                    result = subprocess.run(['lzop', '-d', lzo_file_path], check=True)
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

    ###################################################################################

    def set_path_of_raw_files_inside_extracted_folders(self):
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

    def read_raw_files(self) -> None:
        """
        Function to read raw files from the instance's list of file paths,
        extract header information, timestamps, and data, and save them as '.npy' files.

        Returns:
            None
        """
        for raw_file_path in self.raw_files_path_list:
            logging.info(f'Reading raw file: {raw_file_path}')
            
            # Define header information fields
            hdr_info = ('id', 'frames', 'lines', 'samples', 'samplesize')

            # Initialize dictionaries and arrays to store header, timestamps, and data
            hdr, timestamps, data = {}, None, None
            
            # Open the raw file in binary mode
            try:
                with open(raw_file_path, 'rb') as raw_bytes:
                    # Read header information (4 bytes each)
                    for info in hdr_info:
                        hdr[info] = int.from_bytes(raw_bytes.read(4), byteorder='little')
                    
                    # Read timestamps and data
                    timestamps = np.zeros(hdr['frames'], dtype='int64')
                                    
                    # Calculate the size of each frame
                    sz = hdr['lines'] * hdr['samples'] * hdr['samplesize']
                    
                    # Initialize data array based on file type
                    if "_rf.raw" in raw_file_path:
                        data = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype='int16')
                    elif "_env.raw" in raw_file_path:
                        data = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype='int8')

                    # Loop over frames
                    for frame in range(hdr['frames']):
                        
                        # Read timestamp for each frame (8 bytes)
                        timestamps[frame] = int.from_bytes(raw_bytes.read(8), byteorder='little')
                        
                        # Read frame data and reshape it to match dimensions specified in the header
                        if "_rf.raw" in raw_file_path:
                            data[:, :, frame] = np.frombuffer(raw_bytes.read(sz), dtype='int16').reshape([hdr['lines'], hdr['samples']])
                        elif "_env.raw" in raw_file_path:
                            data[:, :, frame] = np.frombuffer(raw_bytes.read(sz), dtype='uint8').reshape([hdr['lines'], hdr['samples']])

                # Print message indicating the number of frames loaded and their size
                logging.info('Loaded %d raw frames of size %d x %d (lines x samples)', data.shape[2], data.shape[0], data.shape[1])
                
                # Save data as numpy array
                np.save(raw_file_path, data)
                logging.info(f'Saved data as: {raw_file_path}.npy')

            except Exception as e:
                logging.error(f'Error reading file {raw_file_path}: {e}')

    ###################################################################################
    
    def delete_hidden_files_in_extracted_folders(self):
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