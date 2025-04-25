from abc import ABC, abstractmethod

import numpy as np
from typing import List
from PIL import Image, ImageDraw

from ...seg_loaders._obj.us_rf_3d import BmodeSeg3d
from ...scan_loaders._obj.us_rf_3d import UltrasoundRfImage3d
from ...config_loaders._obj.us_rf_3d import RfAnalysisConfig3d

class Window3d:
    """Class to store window data for 3D UTC analysis.
    
    Args:
        ResultsClass (type): Class type to store analysis results.
    """
    def __init__(self, ResultsClass: type):
        self.ax_min: int
        self.ax_max: int
        self.lat_min: int
        self.lat_max: int
        self.cor_min: int
        self.cor_max: int
        self.results = ResultsClass()


class ParamapAnalysis3d(ABC):
    """Facilitate parametric map-centric analysis of ultrasound images.
    """
    
    def __init__(self, ResultsClass: type):
        self.image_data: UltrasoundRfImage3d
        self.config: RfAnalysisConfig3d
        self.seg_data: BmodeSeg3d
        
        self.voi_windows: List[Window3d] = []
        self.single_window: Window3d
        self.ResultsClass = ResultsClass

    def generate_seg_windows(self):
        """Generate windows for UTC analysis based on user-defined spline."""
        # Checks
        assert self.image_data is not None, "Image data not loaded"
        assert self.seg_data is not None, "Segmentation data not loaded"
        assert self.config is not None, "Config data not loaded"
        assert isinstance(self.image_data, UltrasoundRfImage3d), "Image data not of type UltrasoundRfImage"
        assert isinstance(self.seg_data, BmodeSeg3d), "Segmentation data not of type BmodeSeg"
        assert isinstance(self.config, RfAnalysisConfig3d), "Config data not of type RfAnalysisConfig"
        
        # Some axial/lateral/coronal dims
        axial_pix_size = round(self.config.ax_win_size / self.image_data.axial_res)  # mm/(mm/pix)
        lateral_pix_size = round(self.config.lat_win_size / self.image_data.lateral_res)  # mm(mm/pix)
        coronal_pix_size = round(self.config.cor_win_size / self.image_data.coronal_res)  # mm/(mm/pix)

        # Overlap fraction determines the incremental distance between windows
        axial_increment = axial_pix_size * (1 - self.config.axial_overlap)
        lateral_increment = lateral_pix_size * (1 - self.config.lateral_overlap)
        coronal_increment = coronal_pix_size * (1 - self.config.coronal_overlap)

        # Determine windows - Find Volume to Iterate Over
        axial_start = np.min(np.where(np.any(self.seg_data.seg_mask, axis=(0, 1)))[0])
        axial_end = np.max(np.where(np.any(self.seg_data.seg_mask, axis=(0, 1)))[0])
        lateral_start = np.min(np.where(np.any(self.seg_data.seg_mask, axis=(0, 2)))[0])
        lateral_end = np.max(np.where(np.any(self.seg_data.seg_mask, axis=(0, 2)))[0])
        coronal_start = np.min(np.where(np.any(self.seg_data.seg_mask, axis=(1, 2)))[0])
        coronal_end = np.max(np.where(np.any(self.seg_data.seg_mask, axis=(1, 2)))[0])

        self.voi_windows = []

        for axial_pos in np.arange(axial_start, axial_end, axial_increment):
            for lateral_pos in np.arange(lateral_start, lateral_end, lateral_increment):
                for coronal_pos in np.arange(coronal_start, coronal_end, coronal_increment):
                    # Convert axial, lateral, and coronal positions to indices
                    axial_ind = np.round(axial_pos).astype(int)
                    lateral_ind = np.round(lateral_pos).astype(int)
                    coronal_ind = np.round(coronal_pos).astype(int)
                    
                    # Determine if window is inside analysis volume
                    mask_vals = self.seg_data.seg_mask[
                        coronal_ind : (coronal_ind + coronal_pix_size),
                        lateral_ind : (lateral_ind + lateral_pix_size),
                        axial_ind : (axial_ind + axial_pix_size),
                    ]
                    
                    # Define Percentage Threshold
                    total_number_of_elements_in_region = mask_vals.size
                    number_of_ones_in_region = len(np.where(mask_vals == True)[0])
                    percentage_ones = number_of_ones_in_region / total_number_of_elements_in_region
                    
                    if percentage_ones > self.config.window_thresh:
                        # Add ROI to output structure, quantize back to valid distances
                        new_window = Window3d(self.ResultsClass)
                        new_window.ax_min = int(axial_pos)
                        new_window.ax_max = int(axial_pos + axial_pix_size)
                        new_window.lat_min = int(lateral_pos)
                        new_window.lat_max = int(lateral_pos + lateral_pix_size)
                        new_window.cor_min = int(coronal_pos)
                        new_window.cor_max = int(coronal_pos + coronal_pix_size)
                        self.voi_windows.append(new_window)
                    
    @abstractmethod
    def compute_window_vals(self, scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, window: Window3d):
        """Compute parametric map values for a single window.
        
        Args:
            scan_rf_window (np.ndarray): RF data of the window in the scan image.
            phantom_rf_window (np.ndarray): RF data of the window in the phantom image.
            window (Window): Window object to store results.
        """
        pass

    def compute_paramaps(self):
        """Compute UTC parameters for each window in the ROI, creating a parametric map.
        """
        if not len(self.roi_windows):
            self.generate_seg_windows()
            assert len(self.roi_windows) > 0, "No windows generated"

        for window in self.voi_windows:
            img_window = self.image_data.rf_data[window.cor_min: window.cor_max+1,
                                                window.lat_min: window.lat_max+1,
                                                window.ax_min: window.ax_max+1]
            ref_window = self.image_data.phantom_rf_data[window.cor_min: window.cor_max+1,
                                                        window.lat_min: window.lat_max+1,
                                                        window.ax_min: window.ax_max+1]
            self.compute_window_vals(img_window, ref_window, window)
    
    def compute_single_window(self):
        """Compute UTC parameters for a single window capturing the entire analysis volume.
        """
        min_ax = min([window.ax_min for window in self.voi_windows])
        max_ax = max([window.ax_max for window in self.voi_windows])
        min_lat = min([window.lat_min for window in self.voi_windows])
        max_lat = max([window.lat_max for window in self.voi_windows])
        min_cor = min([window.cor_min for window in self.voi_windows])
        max_cor = max([window.cor_max for window in self.voi_windows])
        
        self.single_window = Window3d(self.ResultsClass)
        self.single_window.ax_min = min_ax; self.single_window.ax_max = max_ax
        self.single_window.lat_min = min_lat; self.single_window.lat_max = max_lat
        self.single_window.cor_min = min_cor; self.single_window.cor_max = max_cor
        
        img_window = self.image_data.rf_data[min_cor: max_cor+1, min_lat: max_lat+1, min_ax: max_ax+1]
        ref_window = self.image_data.phantom_rf_data[min_cor: max_cor+1, min_lat: max_lat+1, min_ax: max_ax+1]
        self.compute_window_vals(img_window, ref_window, self.single_window)
