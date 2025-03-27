import pickle
from pathlib import Path
from pyquantus.utc import AnalysisConfig

from typing import Tuple

# DEFAULT
def load_pkl_analysis(analysis_path: str, scan_path: str, phantom_path: str) -> AnalysisConfig:
    """Load analysis configuration data from a pickle file saved from the QuantUS UI.
    
    Args:
        analysis_path (str): Path to the analysis config pickle file
        scan_path (str): Path to the scan file
        phantom_path (str): Path to the phantom file
        
    Returns:
        AnalysisConfig: Analysis configuration data.
    """
    with open(analysis_path, 'rb') as f:
        config_pkl: dict = pickle.load(f)
    return config_pkl['Config']

def load_analysis_assert_phantom(analysis_path: str, scan_path: str, phantom_path: str) -> AnalysisConfig:
    """Load analysis configuration data from a pickle file saved from the QuantUS UI 
    and assert that the scan and analysis config phantoms match.

    Args:
        analysis_path (str): Path to the analysis config pickle file
        scan_path (str): Path to the scan file
        phantom_path (str): Path to the phantom file

    Returns:
        AnalysisConfig: Analysis configuration data.
    """
    with open(analysis_path, 'rb') as f:
        config_pkl: dict = pickle.load(f)
    assert config_pkl['Phantom Name'] == Path(phantom_path).name, 'Phantom file name mismatch'
    return config_pkl['Config']

def load_analysis_assert_scan_phantom(analysis_path: str, scan_path: str, phantom_path: str) -> AnalysisConfig:
    """Load analysis configuration data from a pickle file saved from the QuantUS UI 
    and assert that the scan and analysis configuration scans and phantoms match.

    Args:
        analysis_path (str): Path to the analysis config pickle file
        scan_path (str): Path to the scan file
        phantom_path (str): Path to the phantom file

    Returns:
        AnalysisConfig: Analysis configuration data.
    """
    with open(analysis_path, 'rb') as f:
        config_pkl: dict = pickle.load(f)
    assert config_pkl['Image Name'] == Path(scan_path).name, 'Scan file name mismatch'
    assert config_pkl['Phantom Name'] == Path(phantom_path).name, 'Phantom file name mismatch'
    return config_pkl['Config']
