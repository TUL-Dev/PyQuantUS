import pickle
from pathlib import Path

import numpy as np
from typing import Tuple

# DEFAULT
def load_pkl_roi(roi_path: str, scan_path: str, phantom_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ROI data from a pickle file saved from the QuantUS UI
    
    Args:
        roi_path (str): Path to the ROI pickle file
        scan_path (str): Path to the scan file
        phantom_path (str): Path to the phantom file
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the X and Y coordinates of the 
        ROI in the coordinates of the B-mode image.
    """
    with open(roi_path, 'rb') as f:
        roi_info: dict = pickle.load(f)
    return np.array(roi_info["Spline X"]), np.array(roi_info["Spline Y"])

def load_roi_assert_scan(roi_path: str, scan_path: str, phantom_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ROI data from a pickle file saved from the QuantUS UI 
    and assert that the scan and ROI scans match.

    Args:
        roi_path (str): Path to the ROI pickle file
        scan_path (str): Path to the scan file
        phantom_path (str): Path to the phantom file

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the X and Y coordinates of the
        ROI in the coordinates of the B-mode image.
    """
    with open(roi_path, 'rb') as f:
        roi_info: dict = pickle.load(f)
    assert roi_info["Image Name"] == Path(scan_path).name, 'Scan file name mismatch'
    return np.array(roi_info["Spline X"]), np.array(roi_info["Spline Y"])

def load_roi_assert_scan_phantom(roi_path: str, scan_path: str, phantom_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ROI data from a pickle file saved from the QuantUS UI 
    and assert that the scan and ROI scans match.

    Args:
        roi_path (str): Path to the ROI pickle file
        scan_path (str): Path to the scan file
        phantom_path (str): Path to the phantom file

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the X and Y coordinates of the
        ROI in the coordinates of the B-mode image.
    """
    with open(roi_path, 'rb') as f:
        roi_info: dict = pickle.load(f)
    assert roi_info["Image Name"] == Path(scan_path).name, 'Scan file name mismatch'
    assert roi_info["Phantom Name"] == Path(phantom_path).name, 'Phantom file name mismatch'
    
    return np.array(roi_info["Spline X"]), np.array(roi_info["Spline Y"])