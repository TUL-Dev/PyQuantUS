from .._obj.us_rf import ScUltrasoundRfImage
from .parser import verasonicsRfParser

class EntryClass(ScUltrasoundRfImage):
    """
    Class for Terason RF image data.
    This class is used to parse RF data from Terason ultrasound machines.
    """
    
    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        super().__init__()
        
        param_path = kwargs.get('param_path', None)
        if param_path is None:
            raise ValueError("Parameter path is required.")
        frame = kwargs.get('frame', None)
        if frame is None:
            raise ValueError("Frame is required.")
        
        img_data, img_info, ref_data, ref_info = verasonicsRfParser(scan_path, phantom_path, param_path)
        self.rf_data = img_data.rf[frame]
        self.phantom_rf_data = ref_data.rf[frame]
        self.bmode = img_data.bMode[frame]
        self.sc_bmode = img_data.scBmode[frame]
        self.sc_axial_res = img_info.axialRes
        self.sc_lateral_res = img_info.lateralRes
        self.xmap = img_data.scBmodeStruct.xmap
        self.ymap = img_data.scBmodeStruct.ymap
        self.axial_res = img_info.yResRF
        self.lateral_res = img_info.xResRF
        self.width = img_info.width1
        self.tilt = img_info.tilt1
        self.start_depth = img_info.startDepth1
        self.end_depth = img_info.endDepth1
