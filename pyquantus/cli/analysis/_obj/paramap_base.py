from abc import ABC, abstractmethod

import numpy as np
from typing import List
from PIL import Image, ImageDraw

from ...seg_loaders._obj.us_rf import BmodeSeg
from ...scan_loaders._obj.us_rf import UltrasoundRfImage
from ...config_loaders._obj.us_rf import RfAnalysisConfig

class Window:
    """Class to store window data for UTC analysis.
    
    Args:
        ResultsClass (type): Class type to store analysis results.
    """
    def __init__(self, ResultsClass: type):
        self.left = 0
        self.right = 0
        self.top = 0
        self.bottom = 0
        self.results = ResultsClass()
        

class ParamapAnalysis(ABC):
    """Facilitate parametric map-centric analysis of ultrasound images.
    """
    
    def __init__(self, ResultsClass: type):
        self.image_data: UltrasoundRfImage
        self.config: RfAnalysisConfig
        self.seg_data: BmodeSeg
        
        self.roi_windows: List[Window] = []
        self.single_window: Window
        self.ResultsClass = ResultsClass
        self.spline_x: np.ndarray
        self.spline_y: np.ndarray

    def generate_seg_windows(self):
        """Generate windows for UTC analysis based on user-defined spline."""
        # Checks
        assert self.image_data is not None, "Image data not loaded"
        assert self.seg_data is not None, "Segmentation data not loaded"
        assert self.config is not None, "Config data not loaded"
        assert isinstance(self.image_data, UltrasoundRfImage), "Image data not of type UltrasoundRfImage"
        assert isinstance(self.seg_data, BmodeSeg), "Segmentation data not of type BmodeSeg"
        assert isinstance(self.config, RfAnalysisConfig), "Config data not of type RfAnalysisConfig"
        
        # Some axial/lateral dims
        ax_pix_size = round(self.config.ax_win_size / self.image_data.axial_res)  # mm/(mm/pix)
        lat_pix_size = round(self.config.lat_win_size / self.image_data.lateral_res)  # mm/(mm/pix)
        axial = list(range(self.image_data.rf_data.shape[0]))
        lateral = list(range(self.image_data.rf_data.shape[1]))

        # Overlap fraction determines the incremental distance between ROIs
        axial_increment = ax_pix_size * (1 - self.config.axial_overlap)
        lateral_increment = lat_pix_size * (1 - self.config.lateral_overlap)

        # Determine ROIS - Find Region to Iterate Over
        axial_start = max(min(self.spline_y), axial[0])
        axial_end = min(max(self.spline_y), axial[-1] - ax_pix_size)
        lateral_start = max(min(self.spline_x), lateral[0])
        lateral_end = min(max(self.spline_x), lateral[-1] - lat_pix_size)

        self.roi_windows = []
        # Determine all points inside the user-defined polygon that defines analysis region
        # The 'mask' matrix - "1" inside region and "0" outside region
        # Pair x and y spline coordinates
        spline = []
        if len(self.spline_x) != len(self.spline_y):
            print("Spline has unequal amount of x and y coordinates")
            return
        for i in range(len(self.spline_x)):
            spline.append((self.spline_x[i], self.spline_y[i]))

        img = Image.new("L", (self.image_data.rf_data.shape[1], self.image_data.rf_data.shape[0]), 0)
        ImageDraw.Draw(img).polygon(spline, outline=1, fill=1)
        mask = np.array(img)

        for axial_pos in np.arange(axial_start, axial_end, axial_increment):
            for lateral_pos in np.arange(lateral_start, lateral_end, lateral_increment):
                # Convert axial and lateral positions in mm to Indices
                axial_abs_ar = abs(axial - axial_pos)
                axial_ind = np.where(axial_abs_ar == min(axial_abs_ar))[0][0]
                lateral_abs_ar = abs(lateral - lateral_pos)
                lateral_ind = np.where(lateral_abs_ar == min(lateral_abs_ar))[0][0]

                # Determine if ROI is Inside Analysis Region
                mask_vals = mask[
                    axial_ind : (axial_ind + ax_pix_size),
                    lateral_ind : (lateral_ind + lat_pix_size),
                ]

                # Define Percentage Threshold
                total_elements_in_region = mask_vals.size
                ones_in_region = len(np.where(mask_vals == 1)[0])
                percentage_ones = ones_in_region / total_elements_in_region

                if percentage_ones > self.config.window_thresh:
                    # Add ROI to output structure, quantize back to valid distances
                    new_window = Window(self.ResultsClass)
                    new_window.left = int(lateral[lateral_ind])
                    new_window.right = int(lateral[lateral_ind + lat_pix_size - 1])
                    new_window.top = int(axial[axial_ind])
                    new_window.bottom = int(axial[axial_ind + ax_pix_size - 1])
                    self.roi_windows.append(new_window)
                    
    @abstractmethod
    def compute_window_vals(self, scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, window: Window):
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

        for window in self.roi_windows:
            img_window = self.image_data.rf_data[window.top: window.bottom+1, window.left: window.right+1]
            ref_window = self.image_data.phantom_rf_data[window.top: window.bottom+1, window.left: window.right+1]
            self.compute_window_vals(img_window, ref_window, window)
    
    def compute_single_window(self):
        min_left = min([window.left for window in self.roi_windows])
        max_right = max([window.right for window in self.roi_windows])
        min_top = min([window.top for window in self.roi_windows])
        max_bottom = max([window.bottom for window in self.roi_windows])
        
        self.single_window = Window(self.ResultsClass)
        img_window = self.image_data.rf_data[min_top: max_bottom+1, min_left: max_right+1]
        ref_window = self.image_data.phantom_rf_data[min_top: max_bottom+1, min_left: max_right+1]
        self.compute_window_vals(img_window, ref_window, self.single_window)
