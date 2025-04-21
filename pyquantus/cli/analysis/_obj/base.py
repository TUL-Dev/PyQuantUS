import pickle
from abc import ABC

from ...seg_loaders._obj.base import SegData
from ...scan_loaders._obj.base import ImageData
from ...config_loaders._obj.base import AnalysisConfig
    
class BaseAnalysis(ABC):
    """
    Abstract base class for ultrasound analysis.
    """

    def __init__(self):
        pass

    def load(self, path: str):
        """
        Load RF image data from a given path.
        """
        if path.endswith('.pkl'):
            with open(path, 'rb') as file:
                self = pickle.load(file)
        else:
            raise ValueError("Unsupported file format. Only .pkl files are supported.")

    def save(self, path: str):
        """
        Save RF image data to a given path.
        """
        assert path.endswith('.pkl'), "Only .pkl files are supported."
        with open(path, 'wb') as file:
            pickle.dump(self, file)
