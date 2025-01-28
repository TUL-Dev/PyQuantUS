import os
import re

import yaml
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from typing import Tuple
from tqdm import tqdm

from pyquantus.parse.objects import DataOutputStruct, InfoStruct
from pyquantus.parse.transforms import scanConvert


class ClariusInfo(InfoStruct):
    def __init__(self):
        super().__init__()
        self.numLines: int
        self.samplesPerLine: int
        self.sampleSize: int # bytes


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


def clean_and_convert(value):
    """Clean and convert a string value to a float."""
    clean_value = ''.join([char for char in value if char.isdigit() or char in ['.', '-']])
    return float(clean_value)

def extract_tgc_data_from_line(line):
    """Extract TGC data from a line."""
    tgc_pattern = r'\{([^}]+)\}'
    return re.findall(tgc_pattern, line)

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

def convert_env_to_rf_ntgc(x, linear_tgc_matrix):
    y1 =  47.3 * x + 30
    y = 10**(y1/20)-1
    y = y / linear_tgc_matrix
    return y 

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
