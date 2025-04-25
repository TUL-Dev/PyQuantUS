import pickle
from pathlib import Path

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ...analysis._obj.paramap_base_3d import ParamapAnalysis3d
from .._obj.base import BaseVisualizations
        

class EntryClass(BaseVisualizations):
    """Facilitate parametric map-centric analysis of ultrasound images.
    """
    
    def __init__(self, analysis_obj: ParamapAnalysis3d, **kwargs):
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
        self.analysis_obj.image_data.bmode -= np.amin(self.analysis_obj.image_data.bmode)
        self.analysis_obj.image_data.bmode *= 255/np.amax(self.analysis_obj.image_data.bmode)
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
            self.paramaps.append(np.zeros(self.analysis_obj.image_data.bmode.shape[:3]))
            self.params.append(param)
            
            for window in self.analysis_obj.voi_windows:
                param_color_idx = int((255 / (self.max_param_vals[-1]-self.min_param_vals[-1])
                                     )*(getattr(window.results, param)-self.min_param_vals[-1])
                                    ) if self.min_param_vals[-1] != self.max_param_vals[-1] else 125
        
                self.paramaps[-1][window.cor_min: window.cor_max+1, window.lat_min: window.lat_min+1,
                                  window.ax_min: window.ax_max+1] = param_color_idx+1
                
        bmode_shape = self.analysis_obj.image_data.bmode.shape
        self.window_idx_map = np.zeros((bmode_shape[0], bmode_shape[1]), dtype=int)
        for i, window in enumerate(self.analysis_obj.voi_windows):
            self.window_idx_map[window.cor_min: window.cor_max+1, window.lat_min: window.lat_max+1,
                                window.ax_min: window.ax_max+1] = i+1
            
    def color_paramap(self, paramap: np.ndarray, paramap_ix: int) -> np.ndarray:
        """Applies a colormap to the parametric map.
        
        Args:
            paramap (np.ndarray): The parametric map to color.
            paramap_ix (int): The index of the parametric map.
        
        Returns:
            np.ndarray: The colored parametric map.
        """
        cmap = self.cmaps[paramap_ix % len(self.cmaps)]
        colored_paramap = np.zeros((*paramap.shape, 3), dtype=np.uint8)
        
        window_points = np.transpose(np.where(paramap > 0))
        colored_paramap = np.zeros(paramap.shape + (4,), dtype=np.uint8)
        for point in window_points:
            cmap_idx = int(np.round(paramap[point[0], point[1], point[2]])-1)
            colored_paramap[point[0], point[1], point[2], :3] = (np.array(cmap[cmap_idx])*255).astype(np.uint8)
            colored_paramap[point[0], point[1], point[2], 3] = 255
        
        return colored_paramap
            
    def save_paramaps(self, paramaps: list):
        """Saves the parametric maps to the specified folder.
        """
        if not self.paramap_folder_path:
            print("No paramap folder path specified")
            return
        paramap_folder_path = Path(self.paramap_folder_path)
        paramap_folder_path.mkdir(parents=True, exist_ok=True)
        
        for param_ix, param in enumerate(self.params):
            cmap = self.cmaps[param_ix % len(self.cmaps)]
            paramap = paramaps[param_ix]
            colored_paramap = self.color_paramap(paramap, param_ix)
            with open(paramap_folder_path / f'{param}.pkl', 'wb') as f:
                pickle.dump(colored_paramap, f)
            
            # Plot the paramap legend
            fig, ax = plt.subplots(figsize=(2, 10))
            gradient = np.linspace(0, 1, len(cmap)).reshape(-1, 1)
            ax.imshow(gradient, aspect='auto', cmap=mpl.colors.ListedColormap(cmap))
            ax.set_yticks(np.linspace(0, len(cmap), 6))
            ax.set_yticklabels([f"{self.min_param_vals[param_ix] + i*((self.max_param_vals[param_ix]-self.min_param_vals[param_ix])/5)}"
                                for i in np.linspace(0, 100, 6)])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            fig.savefig(paramap_folder_path / f'{param}_legend.png', bbox_inches='tight')
            del fig, ax

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

    def export_visualizations(self, paramaps: list):
        """Used to specify which visualizations to export and where.
        """
        assert self.ps_plot_output_path.endswith('.png'), "Power spectrum plot output path must end with .png"
        
        # Save the power spectrum plot
        if self.ps_plot_output_path:
            self.ps_plot.savefig(self.ps_plot_output_path)
        
        # Save the parametric maps
        if self.paramap_folder_path:
            self.save_paramaps(paramaps)

    def compute_visualizations(self):
        """Used to specify which visualizations to compute.
        """
        self.draw_paramaps()
        self.plot_ps_data()
        self.export_visualizations(self.paramaps)
