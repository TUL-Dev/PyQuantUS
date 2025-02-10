# Standard Library Imports
import os
import re
from typing import Tuple

# Third-Party Library Imports
import yaml
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from tqdm import tqdm

# Local Module Imports
from pyquantus.parse.objects import DataOutputStruct, InfoStruct
from pyquantus.parse.transforms import scanConvert

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

