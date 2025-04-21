import pickle
from abc import ABC

import pandas as pd
    
class BaseDataExport(ABC):
    """
    Abstract base class for ultrasound visualizations.
    """

    def __init__(self, output_path: str):
        assert output_path.endswith(".csv") or output_path.endswith(".pkl"), "Can only save data to CSV or PKL files!"
        self.output_path = output_path
        self.df: pd.DataFrame

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
