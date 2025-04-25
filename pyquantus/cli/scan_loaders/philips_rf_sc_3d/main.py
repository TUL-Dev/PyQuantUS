import math
import numpy as np

from .._obj.us_rf_3d import ScUltrasoundRfImage3d
from .philipsRf3d import philipsRfParser3d, getVolume
from .philipsRf import philipsRfParser

class EntryClass(ScUltrasoundRfImage3d):
    """
    Class for Terason RF image data.
    This class is used to parse RF data from Terason ultrasound machines.
    """
    
    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        super().__init__(scan_path, phantom_path)
        
        frame = kwargs.get("frame", 0)
        assert isinstance(frame, int)
        
        img_data, img_info, ref_data, ref_info = philipsRfParser3d(scan_path, phantom_path)
        
        self.rf_data = img_data.rf
        self.phantom_rf_data = ref_data.rf
        self.bmode = img_data.bMode
        self.sc_bmode = img_data.scBmode
        self.axial_res = img_info.xResRF
        self.lateral_res = img_info.yResRF
        self.coronal_res = img_info.zResRF
        self.sc_axial_res = img_info.axialRes
        self.sc_lateral_res = img_info.lateralRes
        self.sc_coronal_res = img_info.coronalRes
        self.coord_map_3d = img_data.coordMap3d
        self.depth = img_info.scParams.VDB_2D_ECHO_STOP_DEPTH_SIP # mm
        self.apex_dist = img_info.scParams.VDB_2D_ECHO_APEX_TO_SKINLINE*180/np.pi # mm
        self.azim_steer_angle_start = img_info.scParams.VDB_2D_ECHO_START_WIDTH_GC*180/np.pi # deg
        self.azim_steer_angle_end = img_info.scParams.VDB_2D_ECHO_STOP_WIDTH_GC # deg
        self.elev_steer_angle_start = img_info.scParams.VDB_THREED_START_ELEVATION_ACTUAL*180/np.pi # deg
        self.elev_steer_angle_end = img_info.scParams.VDB_THREED_STOP_ELEVATION_ACTUAL*180/np.pi # deg
        self.vol_height = img_info.scParams.VDB_2D_ECHO_STOP_DEPTH_SIP * (
            abs(math.sin(math.radians(self.elev_steer_angle_start))) + 
            abs(math.sin(math.radians(self.elev_steer_angle_end)))) # Axial
        self.vol_width = img_info.scParams.VDB_2D_ECHO_STOP_DEPTH_SIP * (
            abs(math.sin(math.radians(self.azim_steer_angle_start))) + 
            abs(math.sin(math.radians(self.azim_steer_angle_end)))) # Lateral
        self.vol_depth = img_info.scParams.VDB_2D_ECHO_STOP_DEPTH_SIP - img_info.scParams.VDB_2D_ECHO_START_DEPTH_SIP # Coronal
        self.pix_per_mm = img_info.scParams.pixPerMm # pixels/mm
