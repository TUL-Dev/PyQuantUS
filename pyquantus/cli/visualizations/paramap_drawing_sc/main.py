from pathlib import Path

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ...analysis._obj.paramap_base import ParamapAnalysis
from ..paramap_drawing.main import EntryClass as ParamapVisualizations
from ...scan_loaders._obj.us_rf import ScUltrasoundRfImage
from .transforms import condenseArr, expandArr, scanConvert
        

class EntryClass(ParamapVisualizations):
    """Facilitate parametric map-centric analysis of ultrasound images.
    """
    
    def __init__(self, analysis_obj: ParamapAnalysis, **kwargs):
        super().__init__(analysis_obj, **kwargs)
        assert isinstance(analysis_obj.image_data, ScUltrasoundRfImage), 'image_data must be an ScUltrasoundRfImage child class'
        
        self.ps_plot_output_path = kwargs.get('ps_plot_output_path', None)
        self.paramap_folder_path = kwargs.get('paramap_folder_path', None)
        
        self.analysis_obj = analysis_obj
        self.paramaps = []
        self.legend_paramaps = []
        self.ps_plot = None
        
    def scanConvertRGB(self, image: np.ndarray) -> np.ndarray:
        """Converts a scan-converted grayscale image to RGB.

        Args:
            image (np.ndarray): Grayscale image to convert

        Returns:
            np.ndarray: RGB image
        """
        condensedIm = condenseArr(image)

        width = self.analysis_obj.image_data.width
        tilt = self.analysis_obj.image_data.tilt
        start_depth = self.analysis_obj.image_data.start_depth
        end_depth = self.analysis_obj.image_data.end_depth
        sc_bmode = self.analysis_obj.image_data.sc_bmode
        sc_struct, _, _ = scanConvert(condensedIm, width, tilt,
                                        start_depth, end_depth, desiredHeight=sc_bmode.shape[0])

        return expandArr(sc_struct.scArr)
    
    def scanConvertParamaps(self):
        """Scan converts the parametric maps to match the B-mode image.
        """
        if not len(self.paramaps):
            print("Generate cmaps first")
            return
        if hasattr(self, 'scParamaps'):
            del self.scParamaps
            
        self.scParamaps = [self.scanConvertRGB(paramap) for paramap in self.paramaps]

        width = self.analysis_obj.image_data.width
        tilt = self.analysis_obj.image_data.tilt
        start_depth = self.analysis_obj.image_data.start_depth
        end_depth = self.analysis_obj.image_data.end_depth
        sc_bmode = self.analysis_obj.image_data.sc_bmode
        scStruct, _, _ = scanConvert(self.window_idx_map, width, tilt,
                                        start_depth, end_depth, desiredHeight=sc_bmode.shape[0])
        self.sc_window_idx_map = scStruct.scArr
        
    def draw_paramaps(self):
        """Generates parametric maps for midband fit, spectral slope, and spectral intercept.
        """
        super().draw_paramaps()
        self.scanConvertParamaps()

    def format_paramaps(self):
        """Adds colorbars to the parametric maps."""
        del self.legend_paramaps
        self.legend_paramaps = []
        
        for param_ix, paramap in enumerate(self.scParamaps):
            fig = plt.figure()
            gs = GridSpec(1, 2, width_ratios=[20, 1])  # Adjust width ratios as needed

            # Main image subplot
            ax = fig.add_subplot(gs[0, 0])
            width = paramap.shape[1]*self.analysis_obj.image_data.sc_lateral_res
            height = paramap.shape[0]*self.analysis_obj.image_data.sc_axial_res
            aspect = width/height
            im = ax.imshow(paramap)
            extent =  im.get_extent()
            ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
            ax.axis('off')

            # Create a separate mappable for the colorbar
            norm = mpl.colors.Normalize(vmin=0, vmax=255)
            cmap_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap_names[param_ix % len(self.cmap_names)])

            # Colorbar subplot with exact same height
            cax = fig.add_subplot(gs[0, 1])
            cbar = plt.colorbar(cmap_mappable, cax=cax, orientation='vertical')
            custom_tick_locations = [0, 255/5, 2*255/5, 3*255/5, 4*255/5, 255]  # Example: min, middle, max
            min_val = self.min_param_vals[param_ix]; max_val = self.max_param_vals[param_ix]
            val_range = max_val - min_val
            custom_tick_labels = [np.round(min_val, decimals=2), 
                                np.round(min_val + val_range/5, decimals=2), 
                                np.round(min_val + 2*val_range/5, decimals=2),
                                np.round(min_val + 3*val_range/5, decimals=2), 
                                np.round(min_val + 4*val_range/5, decimals=2), 
                                np.round(max_val, decimals=2)]       # Example: custom labels

            cbar.set_ticks(custom_tick_locations)
            cbar.set_ticklabels(custom_tick_labels)
            cbar.set_label(self.params[param_ix], fontweight='bold', fontsize=14)
            fig.tight_layout()
            self.legend_paramaps.append(fig)