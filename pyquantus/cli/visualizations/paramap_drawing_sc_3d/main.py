from pathlib import Path

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ...analysis._obj.paramap_base_3d import ParamapAnalysis3d
from ..paramap_drawing_3d.main import EntryClass as ParamapVisualizations3d
from ...scan_loaders._obj.us_rf_3d import ScUltrasoundRfImage3d
from .transforms import condenseArr, expandArr, scanConvert3dVolumeSeries
        

class EntryClass(ParamapVisualizations3d):
    """Facilitate parametric map-centric analysis of ultrasound images.
    """
    
    def __init__(self, analysis_obj: ParamapAnalysis3d, **kwargs):
        super().__init__(analysis_obj, **kwargs)
        assert isinstance(analysis_obj.image_data, ScUltrasoundRfImage3d), 'image_data must be an ScUltrasoundRfImage child class'
        
        self.ps_plot_output_path = kwargs.get('ps_plot_output_path', None)
        self.paramap_folder_path = kwargs.get('paramap_folder_path', None)
        
        self.analysis_obj = analysis_obj
        self.paramaps = []
        self.legend_paramaps = []
        self.ps_plot = None
    
    def scan_convert_paramaps(self):
        """Scan converts the parametric maps to match the B-mode image.
        """
        if not len(self.paramaps):
            print("Generate cmaps first")
            return
        if hasattr(self, 'sc_paramaps'):
            del self.sc_paramaps
            
        self.sc_paramaps = [scanConvert3dVolumeSeries(paramap, self.analysis_obj.image_data, 
                                                      scale=False, interp='nearest', normalize=False) for paramap in self.paramaps]
        
    def draw_paramaps(self):
        """Generates parametric maps for midband fit, spectral slope, and spectral intercept.
        """
        super().draw_paramaps()
        self.scan_convert_paramaps()

    def compute_visualizations(self):
        """Used to specify which visualizations to compute.
        """
        self.draw_paramaps()
        self.plot_ps_data()
        self.export_visualizations(self.sc_paramaps)
