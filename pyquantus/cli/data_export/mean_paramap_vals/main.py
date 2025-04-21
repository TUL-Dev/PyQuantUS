import numpy as np
import pandas as pd

from ...analysis._obj.paramap_base import ParamapAnalysis
from .._obj.base import BaseDataExport
        

class EntryClass(BaseDataExport):
    """Facilitate parametric map-centric analysis of ultrasound images.
    """
    def __init__(self, analysis_obj: ParamapAnalysis, output_path: str, **kwargs):
        super().__init__(output_path)
        
        self.analysis_obj = analysis_obj
        
    def save_data(self):
        """Create pandas dataframe to export results."""
        params = self.analysis_obj.roi_windows[0].results.__dict__.keys()
        row = {}
        for param in params:
            if isinstance(getattr(self.analysis_obj.roi_windows[0].results, param), (str, list, np.ndarray)):
                continue
            param_arr = [getattr(window.results, param) for window in self.analysis_obj.roi_windows]
            row[param] = [np.mean(param_arr)]
        
        self.df = pd.DataFrame(row)
        
        if self.output_path.endswith(".csv"):
            self.df.to_csv(self.output_path)
        elif self.output_path.endswith(".pkl"):
            self.df.to_pickle(self.output_path)
        else:
            raise RuntimeError("Output path not CSV or PKL!")