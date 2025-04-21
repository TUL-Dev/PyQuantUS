from pathlib import Path

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ...analysis._obj.paramap_base import ParamapAnalysis
from .._obj.base import BaseVisualizations
        

class EntryClass(BaseVisualizations):
    """Facilitate parametric map-centric analysis of ultrasound images.
    """
    
    def __init__(self, analysis_obj: ParamapAnalysis, **kwargs):
        super().__init__()
        
        self.ps_plot_output_path = kwargs.get('ps_plot_output_path', None)
        self.paramap_folder_path = kwargs.get('paramap_folder_path', None)
        
        self.analysis_obj = analysis_obj
        self.paramaps = []
        self.legend_paramaps = []
        self.ps_plot = None
        
    def convert_images_to_rgb(self):
        """Converts grayscale images to RGB for colormap application.
        """
        self.analysis_obj.image_data.bmode = cv2.cvtColor(
            np.array(self.analysis_obj.image_data.bmode, dtype=np.uint8),
            cv2.COLOR_GRAY2RGB
        )
        
    def draw_paramaps(self):
        """Generates parametric maps for midband fit, spectral slope, and spectral intercept.
        """
        if not len(self.analysis_obj.roi_windows):
            print("No analyzed windows to color")
            return
        if not len(self.analysis_obj.image_data.bmode.shape) == 3:
            self.convert_images_to_rgb()
        if hasattr(self, 'paramaps'):
            del self.paramaps
        
        params = self.analysis_obj.roi_windows[0].results.__dict__.keys()
        self.min_param_vals = []; self.max_param_vals = []; self.paramaps = []
        self.params = []
        
        for param in params:
            if isinstance(getattr(self.analysis_obj.roi_windows[0].results, param), (str, list, np.ndarray)):
                continue
            param_arr = [getattr(window.results, param) for window in self.analysis_obj.roi_windows]
            self.min_param_vals.append(min(param_arr))
            self.max_param_vals.append(max(param_arr))
            self.paramaps.append(self.analysis_obj.image_data.bmode.copy())
            self.params.append(param)
            
            for window in self.analysis_obj.roi_windows:
                param_color_idx = int((255 / (self.max_param_vals[-1]-self.min_param_vals[-1])
                                     )*(getattr(window.results, param)-self.min_param_vals[-1])
                                    ) if self.min_param_vals[-1] != self.max_param_vals[-1] else 125
        
                self.paramaps[-1][window.top: window.bottom+1, window.left: window.right+1] = self.cmaps[len(self.params)-1 % len(self.cmaps)][param_color_idx]*255
                
        bmode_shape = self.analysis_obj.image_data.bmode.shape
        self.window_idx_map = np.zeros((bmode_shape[0], bmode_shape[1]), dtype=int)
        for i, window in enumerate(self.analysis_obj.roi_windows):
            self.window_idx_map[window.top: window.bottom+1, window.left: window.right+1] = i+1

    def format_paramaps(self):
        """Adds colorbars to the parametric maps."""
        del self.legend_paramaps
        self.legend_paramaps = []
        
        for param_ix, paramap in enumerate(self.paramaps):
            fig = plt.figure()
            gs = GridSpec(1, 2, width_ratios=[20, 1])  # Adjust width ratios as needed

            # Main image subplot
            ax = fig.add_subplot(gs[0, 0])
            width = paramap.shape[1]*self.analysis_obj.image_data.lateral_res
            height = paramap.shape[0]*self.analysis_obj.image_data.axial_res
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


    def plot_ps_data(self) -> plt.Figure:
        """Plots the power spectrum data for each window in the ROI.
        
        The power spectrum data is plotted along with the average power spectrum and a line of best fit
        used for the midband fit, spectral slope, and spectral intercept calculations. Also plots the 
        frequency band used for analysis.
        """
        ss_arr = [window.results.ss for window in self.analysis_obj.roi_windows]
        si_arr = [window.results.si for window in self.analysis_obj.roi_windows]
        nps_arr = [window.results.nps for window in self.analysis_obj.roi_windows]
        
        fig, ax = plt.subplots()

        ss_mean = np.mean(np.array(ss_arr)/1e6)
        si_mean = np.mean(si_arr)
        nps_arr = [window.results.nps for window in self.analysis_obj.roi_windows]
        av_nps = np.mean(nps_arr, axis=0)
        f = self.analysis_obj.roi_windows[0].results.f
        x = np.linspace(min(f), max(f), 100)
        y = ss_mean*x + si_mean

        for nps in nps_arr[:-1]:
            ax.plot(f/1e6, nps, c="b", alpha=0.2)
        ax.plot(f/1e6, nps_arr[-1], c="b", alpha=0.2, label="Window NPS")
        ax.plot(f/1e6, av_nps, color="r", label="Av NPS")
        ax.plot(x/1e6, y, c="orange", label="Av LOBF")
        ax.plot(2*[self.analysis_obj.config.analysis_freq_band[0]/1e6], [np.amin(nps_arr), np.amax(nps_arr)], c="purple")
        ax.plot(2*[self.analysis_obj.config.analysis_freq_band[1]/1e6], [np.amin(nps_arr), np.amax(nps_arr)], c="purple", label="Analysis Band")
        ax.set_title("Normalized Power Spectra")
        ax.legend()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power (dB)")
        ax.set_ylim([np.amin(nps_arr), np.amax(nps_arr)])
        ax.set_xlim([min(f)/1e6, max(f)/1e6])
        self.ps_plot = fig

    def export_visualizations(self):
        """Used to specify which visualizations to export and where.
        """
        assert self.ps_plot_output_path.endswith('.png'), "Power spectrum plot output path must end with .png"
        
        # Save the power spectrum plot
        if self.ps_plot_output_path:
            self.ps_plot.savefig(self.ps_plot_output_path)
        
        # Save the parametric maps
        if self.paramap_folder_path:
            paramap_folder_path = Path(self.paramap_folder_path)
            paramap_folder_path.mkdir(parents=True, exist_ok=True)
            for param_ix, param in enumerate(self.params):
                self.legend_paramaps[param_ix].savefig(paramap_folder_path / f'{param}.png')

    def compute_visualizations(self):
        """Used to specify which visualizations to compute.
        """
        self.draw_paramaps()
        self.format_paramaps()
        self.plot_ps_data()
        self.export_visualizations()
